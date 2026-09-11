#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""A :class:`ProxyNDSource` backed by an immutable Zarr array."""

from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from urllib.parse import urlsplit

import numpy as np

import blosc2
from blosc2.core import storage_options_fingerprint
from blosc2.proxy_source import REMOTE_MAX_CONCURRENCY, ProxyNDSource, Traffic

# Zarr's synchronous bridge keeps a process-global loop, thread, and executor.
# The HDF5 translator resets that state to recover from a wedged loop, so every
# read through it must hold this lock to avoid having its runtime yanked away.
ZARR_SYNC_LOCK = threading.RLock()


def owned_fsspec_store(zarr, filesystem, path):
    """Adapt an owned filesystem without Zarr reconstructing another client."""
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    wrapped = AsyncFileSystemWrapper(filesystem, asynchronous=True)
    return zarr.storage.FsspecStore(wrapped, path=path, read_only=True)


def counting_store(zarr, store, traffic, *, metadata=None):
    if getattr(store, "_blosc2_traffic", None) is traffic:
        return store

    class CountingStore(zarr.storage.WrapperStore):
        _blosc2_traffic = traffic

        async def get(self, key, prototype, byte_range=None):
            is_metadata = (
                metadata is not None
                and key.rsplit("/", 1)[-1] in {".zarray", ".zgroup", ".zattrs", ".zmetadata", "zarr.json"}
                and byte_range is None
            )
            if is_metadata and key in metadata:
                data = metadata[key]
                return None if data is None else prototype.buffer.from_bytes(data)
            value = await super().get(key, prototype, byte_range)
            if value is not None:
                traffic.charge(len(value))
            if is_metadata:
                metadata[key] = None if value is None else value.to_bytes()
            return value

        async def get_partial_values(self, prototype, key_ranges):
            values = await super().get_partial_values(prototype, key_ranges)
            received = sum(len(value) for value in values if value is not None)
            if received:
                traffic.charge(received)
            return values

    return CountingStore(store)


_counting_store = counting_store


def zarr_chunk_to_blosc2(
    array,
    nchunk: int,
    shape: tuple,
    chunks: tuple,
    blocks: tuple,
    dtype: np.dtype,
    cparams,
) -> bytes:
    """Read a Zarr chunk slice and return it as Blosc2 compressed bytes."""
    grid = tuple(math.ceil(size / chunk) for size, chunk in zip(shape, chunks, strict=True))
    total = math.prod(grid)
    if isinstance(nchunk, bool) or not isinstance(nchunk, int) or nchunk < 0 or nchunk >= total:
        raise IndexError(f"nchunk must be in range [0, {total}), got {nchunk}")
    coords = np.unravel_index(nchunk, grid)
    selection = tuple(
        slice(int(coord) * chunk, min((int(coord) + 1) * chunk, size))
        for coord, chunk, size in zip(coords, chunks, shape, strict=True)
    )
    with ZARR_SYNC_LOCK:
        values = np.asarray(array[selection], dtype=dtype)
    buffer = np.zeros(chunks, dtype=dtype)
    if shape:
        values = np.ascontiguousarray(values)
        buffer[tuple(slice(0, size) for size in values.shape)] = values
    else:
        buffer[()] = values
    converted = blosc2.asarray(buffer, chunks=chunks, blocks=blocks, cparams=cparams)
    return converted.schunk.get_chunk(0)


_zarr_chunk_to_blosc2 = zarr_chunk_to_blosc2


class ZarrNDSource(ProxyNDSource):
    """Read an immutable Zarr array as Blosc2-compressed logical chunks.

    Replacing data beneath the same store identity violates this adapter's
    contract and may leave previously converted chunks stale.  Peak working
    memory includes concurrently decoded Zarr chunks and their Blosc2
    conversion buffers; ``max_cache_bytes`` only limits retained compressed
    chunks.
    """

    serves_blocks = False
    encoding_version = 1

    def __init__(
        self,
        store,
        *,
        storage_options: dict | None = None,
        max_concurrency: int = REMOTE_MAX_CONCURRENCY,
        blocks=None,
        cparams=None,
        _traffic: Traffic | None = None,
        _urlpath: str | None = None,
        _path: str | None = None,
    ):
        try:
            import zarr
        except ImportError as exc:
            raise ImportError(
                "ZarrNDSource requires Zarr-Python; install it with 'pip install blosc2[zarr]'"
            ) from exc

        if isinstance(store, os.PathLike):
            store = os.fspath(store)
        self.urlpath = _urlpath if _urlpath is not None else store if isinstance(store, str) else None
        self.max_concurrency = max_concurrency
        remote = isinstance(store, str) and bool(urlsplit(store).scheme)
        self.traffic = _traffic if _traffic is not None else Traffic() if remote else None
        open_store = store
        if remote:
            try:
                import fsspec
            except ImportError as exc:
                raise ImportError(
                    "remote Zarr sources require fsspec; install with 'pip install blosc2[zarr,fsspec]'"
                ) from exc

            source = zarr.storage.FsspecStore.from_mapper(
                fsspec.get_mapper(store, **(storage_options or {})), read_only=True
            )
            open_store = source
        if self.traffic is not None:
            open_store = _counting_store(zarr, open_store, self.traffic)
        try:
            with ZARR_SYNC_LOCK:
                self.array = zarr.open_array(
                    store=open_store,
                    path=_path,
                    mode="r",
                )
        except Exception as exc:
            if type(exc).__name__ in {"ContainsGroupError", "NodeTypeValidationError"}:
                raise ValueError(f"{store!r} is a Zarr group; pass the path of an array") from exc
            raise

        self._shape = tuple(int(value) for value in self.array.shape)
        self._chunks = tuple(int(value) for value in self.array.chunks)
        try:
            self._dtype = np.dtype(self.array.dtype)
        except TypeError as exc:
            raise TypeError(f"ZarrNDSource only supports fixed-size dtypes, got {self.array.dtype}") from exc
        self._validate_metadata()
        _, computed_blocks = blosc2.compute_chunks_blocks(
            self._shape, chunks=self._chunks, blocks=blocks, dtype=self._dtype, cparams=cparams
        )
        self._blocks = tuple(computed_blocks)
        self._cparams = (
            blosc2.CParams(typesize=self._dtype.itemsize)
            if cparams is None
            else blosc2.CParams(**cparams)
            if isinstance(cparams, dict)
            else cparams
        )
        identity = {
            "encoding_version": self.encoding_version,
            "urlpath": self.urlpath,
            "shape": self._shape,
            "chunks": self._chunks,
            "blocks": self._blocks,
            "dtype": self._dtype.str,
        }
        fingerprint = storage_options_fingerprint(storage_options) if remote else ""
        if fingerprint:
            # The same store path through another endpoint/account may hold
            # different bytes, so it is a different source identity.
            identity["storage_options"] = fingerprint
        self.stamp = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _validate_metadata(self) -> None:
        if len(self._shape) > blosc2.MAX_DIM:
            raise ValueError(f"Zarr arrays may have at most {blosc2.MAX_DIM} dimensions")
        if len(self._chunks) != len(self._shape) or any(size <= 0 for size in self._chunks):
            raise ValueError("Zarr chunk extents must be positive and match the array dimensions")
        if self._dtype.hasobject or self._dtype.itemsize == 0:
            raise TypeError(f"ZarrNDSource only supports fixed-size dtypes, got {self._dtype}")
        chunk_nbytes = math.prod(self._chunks) * self._dtype.itemsize
        if chunk_nbytes > blosc2.MAX_BUFFERSIZE:
            raise ValueError(
                f"Zarr chunks must be at most {blosc2.MAX_BUFFERSIZE} bytes, got {chunk_nbytes}"
            )

    @property
    def shape(self) -> tuple:
        return self._shape

    @property
    def chunks(self) -> tuple:
        return self._chunks

    @property
    def blocks(self) -> tuple:
        return self._blocks

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def cparams(self):
        return self._cparams

    @property
    def attrs(self) -> dict:
        """The user attributes of the remote array."""
        return self.vlmeta

    @property
    def vlmeta(self) -> dict:
        try:
            return dict(self.array.attrs)
        except Exception:
            return {}

    def get_chunk(self, nchunk: int) -> bytes:
        return zarr_chunk_to_blosc2(
            self.array,
            nchunk,
            self.shape,
            self.chunks,
            self.blocks,
            self.dtype,
            self.cparams,
        )
