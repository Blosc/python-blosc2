#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""A :class:`ProxyNDSource` backed by an immutable HDF5 dataset via kerchunk."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from urllib.parse import urlsplit

import numpy as np

import blosc2
from blosc2.proxy_source import REMOTE_MAX_CONCURRENCY, ProxyNDSource, Traffic
from blosc2.zarr_source import counting_store, zarr_chunk_to_blosc2


def check_hdf5_dependencies() -> None:
    """Validate that kerchunk and h5py are available."""
    try:
        import kerchunk.hdf  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HDF5 support requires kerchunk; install it with 'pip install blosc2[hdf5]'"
        ) from exc

    try:
        import h5py  # noqa: F401
    except ImportError as exc:
        raise ImportError("HDF5 support requires h5py; install it with 'pip install blosc2[hdf5]'") from exc

    with contextlib.suppress(ImportError):
        import hdf5plugin  # noqa: F401

    _ensure_blosc2_filter_registered()


def _register_numcodecs_blosc2() -> None:
    try:
        import numcodecs
        import numcodecs.abc
    except ImportError:
        return

    if hasattr(numcodecs, "Blosc2"):
        return

    class Blosc2Codec(numcodecs.abc.Codec):
        codec_id = "blosc2"

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def encode(self, buf):
            return buf

        def decode(self, buf, out=None):
            try:
                decomp = blosc2.decompress(buf)
            except Exception:
                decomp = blosc2.from_cframe(buf)[:].tobytes()
            if out is not None:
                np.frombuffer(out, dtype=np.uint8)[:] = np.frombuffer(decomp, dtype=np.uint8)
                return out
            return decomp

        def get_config(self):
            return {"id": self.codec_id, **self.kwargs}

    numcodecs.register_codec(Blosc2Codec)
    numcodecs.Blosc2 = Blosc2Codec


def _patch_kerchunk_decode_filters() -> None:
    try:
        import kerchunk.hdf
        import numcodecs
    except ImportError:
        return

    if getattr(kerchunk.hdf.SingleHdf5ToZarr, "_blosc2_patched", False):
        return

    orig_decode = kerchunk.hdf.SingleHdf5ToZarr._decode_filters

    def patched_decode(self, h5obj):
        filters = []
        saved = {}
        for filter_id, props in list(h5obj._filters.items()):
            if str(filter_id) == "32026":
                saved[filter_id] = props
                filters.append(numcodecs.Blosc2())
        for fid in saved:
            h5obj._filters.pop(fid, None)
        try:
            filters.extend(orig_decode(self, h5obj))
        finally:
            h5obj._filters.update(saved)
        return filters

    kerchunk.hdf.SingleHdf5ToZarr._decode_filters = patched_decode
    kerchunk.hdf.SingleHdf5ToZarr._blosc2_patched = True


def _ensure_blosc2_filter_registered() -> None:
    _register_numcodecs_blosc2()
    _patch_kerchunk_decode_filters()


def check_zarr_fsspec_dependencies() -> None:
    """Validate that zarr and fsspec are available."""
    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HDF5NDSource requires Zarr-Python; install it with 'pip install blosc2[zarr]'"
        ) from exc

    try:
        import fsspec  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "HDF5NDSource requires fsspec; install it with 'pip install blosc2[fsspec]'"
        ) from exc


def available_datasets(url, storage_options: dict | None = None) -> list[str]:
    """Return all dataset paths within an HDF5 file or reference dictionary.

    Parameters
    ----------
    url : str, os.PathLike, or dict
        Path or URL to an HDF5 file, a JSON reference file, or an in-memory
        kerchunk reference dictionary.
    storage_options : dict, optional
        Options passed to fsspec or kerchunk for remote URLs.

    Returns
    -------
    list[str]
        Sorted list of dataset paths (e.g. ``['d0/a0', 'd0/d1/a2']``).
    """
    if isinstance(url, dict):
        ref_dict = url.get("refs", url)
    elif isinstance(url, (str, os.PathLike)):
        url_str = os.fspath(url)
        if "::" in url_str:
            parts = url_str.split("::", 1)
            if "://" not in parts[1]:
                url_str = parts[0].rstrip("/")
        lower = url_str.lower()
        for ext in (".h5/", ".hdf5/"):
            idx = lower.find(ext)
            if idx != -1:
                url_str = url_str[: idx + len(ext) - 1]
                break
        if url_str.endswith(".json"):
            try:
                import fsspec

                with fsspec.open(url_str, "r", **(storage_options or {})) as f:
                    refs = json.load(f)
            except Exception:
                with open(url_str) as f:
                    refs = json.load(f)
            ref_dict = refs.get("refs", refs)
        else:
            check_hdf5_dependencies()
            import kerchunk.hdf

            refs = kerchunk.hdf.SingleHdf5ToZarr(url_str, storage_options=storage_options or {}).translate()
            ref_dict = refs.get("refs", refs)
    else:
        raise TypeError("url must be a URL string, path-like object, or reference dict")

    datasets = []
    for k in ref_dict:
        if k.endswith("/.zarray"):
            datasets.append(k[: -len("/.zarray")])
        elif k == ".zarray":
            datasets.append("/")
    return sorted(datasets)


class HDF5NDSource(ProxyNDSource):
    """Read an immutable HDF5 dataset as Blosc2-compressed logical chunks via kerchunk.

    Replacing data beneath the same store identity violates this adapter's
    contract and may leave previously converted chunks stale. Peak working
    memory includes concurrently decoded Zarr chunks and their Blosc2
    conversion buffers; ``max_cache_bytes`` only limits retained compressed
    chunks.

    Parameters
    ----------
    urlpath : str or path-like
        URL or file path to the HDF5 file.
    dataset : str
        Path to the dataset within the HDF5 file (e.g. ``"d0/d1/a2"``).
    refs : dict, str, or path-like, optional
        Pre-computed kerchunk reference dictionary or path to a JSON reference
        file. If omitted, the HDF5 metadata will be scanned using kerchunk.
    storage_options : dict, optional
        Parameters passed to fsspec or kerchunk when accessing remote files.
    max_concurrency : int, optional
        Maximum number of concurrent remote requests.
    blocks : tuple, optional
        Blosc2 block shape for chunk caching.
    cparams : dict or CParams, optional
        Blosc2 compression parameters for chunk conversion.
    _traffic : Traffic, optional
        Traffic monitor instance.
    """

    serves_blocks = False
    encoding_version = 1

    def __init__(
        self,
        urlpath,
        dataset: str,
        *,
        refs: dict | str | os.PathLike | None = None,
        storage_options: dict | None = None,
        max_concurrency: int = REMOTE_MAX_CONCURRENCY,
        blocks=None,
        cparams=None,
        _traffic: Traffic | None = None,
    ):
        check_hdf5_dependencies()
        check_zarr_fsspec_dependencies()

        if isinstance(urlpath, os.PathLike):
            urlpath = os.fspath(urlpath)
        if isinstance(urlpath, str):
            if "::" in urlpath:
                parts = urlpath.split("::", 1)
                if "://" not in parts[1]:
                    if dataset is not None and dataset != parts[1].strip("/"):
                        raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
                    urlpath = parts[0].rstrip("/")
                    dataset = parts[1].strip("/")
            lower = urlpath.lower()
            for ext in (".h5/", ".hdf5/"):
                idx = lower.find(ext)
                if idx != -1:
                    base_len = idx + len(ext) - 1
                    sub = urlpath[base_len + 1 :].strip("/")
                    if dataset is not None and dataset != sub:
                        raise ValueError("Cannot specify dataset in both URL path and dataset parameter")
                    dataset = sub
                    urlpath = urlpath[:base_len]
                    break

        if dataset is None:
            raise ValueError("HDF5 sources require a dataset path (e.g., dataset='d0/d1/a2')")
        if not isinstance(dataset, str):
            raise TypeError("dataset must be a string")

        self.urlpath = urlpath if isinstance(urlpath, str) else str(urlpath)
        self.dataset = dataset.strip("/")
        self.max_concurrency = max_concurrency

        remote = isinstance(self.urlpath, str) and bool(urlsplit(self.urlpath).scheme)
        self.traffic = _traffic if _traffic is not None else Traffic() if remote else None

        self._refs = self._load_or_scan_refs(refs, storage_options)
        self._validate_dataset_presence(dataset)
        self.array = self._open_array(storage_options)

        self._shape = tuple(int(value) for value in self.array.shape)
        self._chunks = tuple(int(value) for value in self.array.chunks)
        try:
            self._dtype = np.dtype(self.array.dtype)
        except TypeError as exc:
            raise TypeError(f"HDF5NDSource only supports fixed-size dtypes, got {self.array.dtype}") from exc

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
            "dataset": self.dataset,
            "shape": self._shape,
            "chunks": self._chunks,
            "blocks": self._blocks,
            "dtype": self._dtype.str,
        }
        self.stamp = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _load_or_scan_refs(self, refs, storage_options) -> dict:
        if refs is not None:
            if isinstance(refs, (str, os.PathLike)):
                refs_str = os.fspath(refs)
                if isinstance(refs_str, str) and bool(urlsplit(refs_str).scheme):
                    import fsspec

                    with fsspec.open(refs_str, "r", **(storage_options or {})) as f:
                        return json.load(f)
                with open(refs_str) as f:
                    return json.load(f)
            if isinstance(refs, dict):
                return refs
            raise TypeError("refs must be a dict, string, or path-like object")

        import kerchunk.hdf

        return kerchunk.hdf.SingleHdf5ToZarr(self.urlpath, storage_options=storage_options or {}).translate()

    def _validate_dataset_presence(self, raw_dataset: str) -> None:
        ref_dict = self._refs.get("refs", self._refs)
        clean = self.dataset

        is_group = f"{clean}/.zgroup" in ref_dict or (clean == "" and ".zgroup" in ref_dict)
        if is_group:
            available = available_datasets(self._refs)
            raise ValueError(
                f"{raw_dataset!r} is an HDF5 group; pass the path of a dataset. "
                f"Available datasets: {available}"
            )

        is_array = f"{clean}/.zarray" in ref_dict or (clean == "" and ".zarray" in ref_dict)
        if not is_array:
            available = available_datasets(self._refs)
            raise ValueError(
                f"dataset {raw_dataset!r} not found in {self.urlpath!r}. Available datasets: {available}"
            )

    def _open_array(self, storage_options):
        import fsspec
        import zarr

        rfs_kwargs = {}
        if storage_options:
            rfs_kwargs["target_options"] = storage_options
            rfs_kwargs["remote_options"] = storage_options
        fs = fsspec.filesystem("reference", fo=self._refs, **rfs_kwargs)
        mapper = fs.get_mapper(self.dataset)
        try:
            open_store = zarr.storage.FsspecStore.from_mapper(mapper, read_only=True)
        except ValueError:
            from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

            wrapped_fs = AsyncFileSystemWrapper(fs, asynchronous=True)
            open_store = zarr.storage.FsspecStore(wrapped_fs, path=mapper.root, read_only=True)
        if self.traffic is not None:
            open_store = counting_store(zarr, open_store, self.traffic)
        return zarr.open_array(store=open_store, mode="r")

    def _validate_metadata(self) -> None:
        if len(self._shape) > blosc2.MAX_DIM:
            raise ValueError(f"HDF5 arrays may have at most {blosc2.MAX_DIM} dimensions")
        if len(self._chunks) != len(self._shape) or any(size <= 0 for size in self._chunks):
            raise ValueError("HDF5 chunk extents must be positive and match the array dimensions")
        if self._dtype.hasobject or self._dtype.itemsize == 0:
            raise TypeError(f"HDF5NDSource only supports fixed-size dtypes, got {self._dtype}")
        chunk_nbytes = math.prod(self._chunks) * self._dtype.itemsize
        if chunk_nbytes > blosc2.MAX_BUFFERSIZE:
            raise ValueError(
                f"HDF5 chunks must be at most {blosc2.MAX_BUFFERSIZE} bytes, got {chunk_nbytes}"
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
