#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import copy
import pathlib
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import blosc2
from blosc2._msgpack_utils import msgpack_packb, msgpack_unpackb

_BATCHARRAY_META = {"version": 1, "serializer": "msgpack", "format": "vlblocks"}


def _check_serialized_size(buffer: bytes) -> None:
    if len(buffer) > blosc2.MAX_BUFFERSIZE:
        raise ValueError(f"Serialized objects cannot be larger than {blosc2.MAX_BUFFERSIZE} bytes")


class Batch(Sequence[Any]):
    """A lazy sequence of Python objects stored in one BatchArray chunk."""

    def __init__(self, parent: BatchArray, nchunk: int, lazychunk: bytes) -> None:
        self._parent = parent
        self._nchunk = nchunk
        self._lazychunk = lazychunk
        self._payloads: list[bytes] | None = None
        self._nbytes, self._cbytes, self._nblocks = blosc2.get_cbuffer_sizes(lazychunk)

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("Batch indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Batch index out of range")
        return index

    def _decode_payloads(self) -> list[bytes]:
        if self._payloads is None:
            self._payloads = self._parent._decode_payloads(self._nchunk)
        return self._payloads

    def __getitem__(self, index: int | slice) -> Any | list[Any]:
        payloads = self._decode_payloads()
        if isinstance(index, slice):
            return [msgpack_unpackb(payload) for payload in payloads[index]]
        index = self._normalize_index(index)
        return msgpack_unpackb(payloads[index])

    def __len__(self) -> int:
        return self._nblocks

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    @property
    def lazychunk(self) -> bytes:
        return self._lazychunk

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def cbytes(self) -> int:
        return self._cbytes

    @property
    def cratio(self) -> float:
        return self._nbytes / self._cbytes

    def __repr__(self) -> str:
        return f"Batch(len={len(self)}, nbytes={self.nbytes}, cbytes={self.cbytes})"


class BatchArray:
    """A batched variable-length array backed by an :class:`blosc2.SChunk`."""

    @staticmethod
    def _set_typesize_one(cparams: blosc2.CParams | dict | None) -> blosc2.CParams | dict:
        if cparams is None:
            cparams = blosc2.CParams()
        elif isinstance(cparams, blosc2.CParams):
            cparams = copy.deepcopy(cparams)
        else:
            cparams = dict(cparams)

        if isinstance(cparams, blosc2.CParams):
            cparams.typesize = 1
        else:
            cparams["typesize"] = 1
        return cparams

    @staticmethod
    def _coerce_storage(storage: blosc2.Storage | dict | None, kwargs: dict[str, Any]) -> blosc2.Storage:
        if storage is not None:
            storage_keys = set(blosc2.Storage.__annotations__)
            storage_kwargs = storage_keys.intersection(kwargs)
            if storage_kwargs:
                unexpected = ", ".join(sorted(storage_kwargs))
                raise AttributeError(
                    f"Cannot pass both `storage` and other kwargs already included in Storage: {unexpected}"
                )
            if isinstance(storage, blosc2.Storage):
                return copy.deepcopy(storage)
            return blosc2.Storage(**storage)

        storage_kwargs = {
            name: kwargs.pop(name) for name in list(blosc2.Storage.__annotations__) if name in kwargs
        }
        return blosc2.Storage(**storage_kwargs)

    @staticmethod
    def _validate_storage(storage: blosc2.Storage) -> None:
        if storage.mmap_mode not in (None, "r"):
            raise ValueError("For BatchArray containers, mmap_mode must be None or 'r'")
        if storage.mmap_mode == "r" and storage.mode != "r":
            raise ValueError("For BatchArray containers, mmap_mode='r' requires mode='r'")

    def _attach_schunk(self, schunk: blosc2.SChunk) -> None:
        self.schunk = schunk
        self.urlpath = schunk.urlpath
        self.mode = schunk.mode
        self.mmap_mode = getattr(schunk, "mmap_mode", None)
        self._validate_tag()

    def _maybe_open_existing(self, storage: blosc2.Storage) -> bool:
        urlpath = storage.urlpath
        if urlpath is None or storage.mode not in ("r", "a") or not pathlib.Path(urlpath).exists():
            return False

        schunk = blosc2.blosc2_ext.open(urlpath, mode=storage.mode, offset=0, mmap_mode=storage.mmap_mode)
        self._attach_schunk(schunk)
        return True

    def _make_storage(self) -> blosc2.Storage:
        meta = {name: self.meta[name] for name in self.meta}
        return blosc2.Storage(
            contiguous=self.schunk.contiguous,
            urlpath=self.urlpath,
            mode=self.mode,
            mmap_mode=self.mmap_mode,
            meta=meta,
        )

    def __init__(
        self,
        chunksize: int | None = None,
        _from_schunk: blosc2.SChunk | None = None,
        **kwargs: Any,
    ) -> None:
        if _from_schunk is not None:
            if chunksize is not None:
                raise ValueError("Cannot pass `chunksize` together with `_from_schunk`")
            if kwargs:
                unexpected = ", ".join(sorted(kwargs))
                raise ValueError(f"Cannot pass {unexpected} together with `_from_schunk`")
            self._attach_schunk(_from_schunk)
            return

        cparams = kwargs.pop("cparams", None)
        dparams = kwargs.pop("dparams", None)
        storage = kwargs.pop("storage", None)
        storage = self._coerce_storage(storage, kwargs)

        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise ValueError(f"Unsupported BatchArray keyword argument(s): {unexpected}")

        self._validate_storage(storage)
        cparams = self._set_typesize_one(cparams)

        if dparams is None:
            dparams = blosc2.DParams()

        if self._maybe_open_existing(storage):
            return

        fixed_meta = dict(storage.meta or {})
        fixed_meta["batcharray"] = dict(_BATCHARRAY_META)
        storage.meta = fixed_meta
        if chunksize is None:
            chunksize = -1
        schunk = blosc2.SChunk(
            chunksize=chunksize, data=None, cparams=cparams, dparams=dparams, storage=storage
        )
        self._attach_schunk(schunk)

    def _validate_tag(self) -> None:
        if "batcharray" not in self.schunk.meta:
            raise ValueError("The supplied SChunk is not tagged as a BatchArray")

    def _check_writable(self) -> None:
        if self.mode == "r":
            raise ValueError("Cannot modify a BatchArray opened in read-only mode")

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("BatchArray indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("BatchArray index out of range")
        return index

    def _normalize_insert_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("BatchArray indices must be integers")
        if index < 0:
            index += len(self)
            if index < 0:
                return 0
        if index > len(self):
            return len(self)
        return index

    def _slice_indices(self, index: slice) -> list[int]:
        return list(range(*index.indices(len(self))))

    def _copy_meta(self) -> dict[str, Any]:
        return {name: self.meta[name] for name in self.meta}

    def _normalize_batch(self, value: object) -> list[Any]:
        if isinstance(value, (str, bytes, bytearray, memoryview)):
            raise TypeError("BatchArray entries must be sequences of Python objects")
        if not isinstance(value, Sequence):
            raise TypeError("BatchArray entries must be sequences of Python objects")
        values = list(value)
        if len(values) == 0:
            raise ValueError("BatchArray entries cannot be empty")
        return values

    def _serialize_batch(self, value: object) -> list[bytes]:
        payloads = []
        for item in self._normalize_batch(value):
            payload = msgpack_packb(item)
            _check_serialized_size(payload)
            payloads.append(payload)
        return payloads

    def _vl_cparams_kwargs(self) -> dict[str, Any]:
        return asdict(self.schunk.cparams)

    def _vl_dparams_kwargs(self) -> dict[str, Any]:
        return asdict(self.schunk.dparams)

    def _compress_batch(self, payloads: list[bytes]) -> bytes:
        return blosc2.blosc2_ext.vlcompress(payloads, **self._vl_cparams_kwargs())

    def _decode_payloads(self, nchunk: int) -> list[bytes]:
        return blosc2.blosc2_ext.vldecompress(self.schunk.get_chunk(nchunk), **self._vl_dparams_kwargs())

    def _get_batch(self, index: int) -> Batch:
        return Batch(self, index, self.schunk.get_lazychunk(index))

    def append(self, value: object) -> int:
        """Append one batch and return the new number of entries."""
        self._check_writable()
        chunk = self._compress_batch(self._serialize_batch(value))
        return self.schunk.append_chunk(chunk)

    def insert(self, index: int, value: object) -> int:
        """Insert one batch at ``index`` and return the new number of entries."""
        self._check_writable()
        index = self._normalize_insert_index(index)
        chunk = self._compress_batch(self._serialize_batch(value))
        return self.schunk.insert_chunk(index, chunk)

    def delete(self, index: int | slice) -> int:
        """Delete the batch at ``index`` and return the new number of entries."""
        self._check_writable()
        if isinstance(index, slice):
            for idx in reversed(self._slice_indices(index)):
                self.schunk.delete_chunk(idx)
            return len(self)
        index = self._normalize_index(index)
        return self.schunk.delete_chunk(index)

    def pop(self, index: int = -1) -> list[Any]:
        """Remove and return the batch at ``index``."""
        self._check_writable()
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for BatchArray")
        index = self._normalize_index(index)
        value = self[index][:]
        self.schunk.delete_chunk(index)
        return value

    def extend(self, values: object) -> None:
        """Append all batches from an iterable."""
        self._check_writable()
        for value in values:
            chunk = self._compress_batch(self._serialize_batch(value))
            self.schunk.append_chunk(chunk)

    def clear(self) -> None:
        """Remove all entries from the container."""
        self._check_writable()
        storage = self._make_storage()
        if storage.urlpath is not None:
            blosc2.remove_urlpath(storage.urlpath)
        schunk = blosc2.SChunk(
            chunksize=-1,
            data=None,
            cparams=copy.deepcopy(self.cparams),
            dparams=copy.deepcopy(self.dparams),
            storage=storage,
        )
        self._attach_schunk(schunk)

    def __getitem__(self, index: int | slice) -> Batch | list[Batch]:
        if isinstance(index, slice):
            return [self[i] for i in self._slice_indices(index)]
        index = self._normalize_index(index)
        return self._get_batch(index)

    def __setitem__(self, index: int | slice, value: object) -> None:
        if isinstance(index, slice):
            self._check_writable()
            indices = self._slice_indices(index)
            values = list(value)
            step = 1 if index.step is None else index.step
            if step == 1:
                start = self._normalize_insert_index(0 if index.start is None else index.start)
                for idx in reversed(indices):
                    self.schunk.delete_chunk(idx)
                for offset, item in enumerate(values):
                    chunk = self._compress_batch(self._serialize_batch(item))
                    self.schunk.insert_chunk(start + offset, chunk)
                return
            if len(values) != len(indices):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values)} to extended slice of size {len(indices)}"
                )
            for idx, item in zip(indices, values, strict=True):
                chunk = self._compress_batch(self._serialize_batch(item))
                self.schunk.update_chunk(idx, chunk)
            return
        self._check_writable()
        index = self._normalize_index(index)
        chunk = self._compress_batch(self._serialize_batch(value))
        self.schunk.update_chunk(index, chunk)

    def __delitem__(self, index: int | slice) -> None:
        self.delete(index)

    def __len__(self) -> int:
        return self.schunk.nchunks

    def __iter__(self) -> Iterator[Batch]:
        for i in range(len(self)):
            yield self[i]

    @property
    def meta(self):
        return self.schunk.meta

    @property
    def vlmeta(self):
        return self.schunk.vlmeta

    @property
    def cparams(self):
        return self.schunk.cparams

    @property
    def dparams(self):
        return self.schunk.dparams

    @property
    def chunksize(self) -> int:
        return self.schunk.chunksize

    @property
    def nbytes(self) -> int:
        return self.schunk.nbytes

    @property
    def cbytes(self) -> int:
        return self.schunk.cbytes

    @property
    def cratio(self) -> float:
        return self.schunk.cratio

    def to_cframe(self) -> bytes:
        return self.schunk.to_cframe()

    def copy(self, **kwargs: Any) -> BatchArray:
        """Create a copy of the container with optional constructor overrides."""
        if "meta" in kwargs:
            raise ValueError("meta should not be passed to copy")

        kwargs["cparams"] = kwargs.get("cparams", copy.deepcopy(self.cparams))
        kwargs["dparams"] = kwargs.get("dparams", copy.deepcopy(self.dparams))
        kwargs["chunksize"] = kwargs.get("chunksize", -1)

        if "storage" not in kwargs:
            kwargs["meta"] = self._copy_meta()
            kwargs["contiguous"] = kwargs.get("contiguous", self.schunk.contiguous)
            if "urlpath" in kwargs and "mode" not in kwargs:
                kwargs["mode"] = "w"

        out = BatchArray(**kwargs)
        out.extend(self)
        return out

    def __enter__(self) -> BatchArray:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __repr__(self) -> str:
        return f"BatchArray(len={len(self)}, urlpath={self.urlpath!r})"


def batcharray_from_cframe(cframe: bytes, copy: bool = True) -> BatchArray:
    """Deserialize a CFrame buffer into a :class:`BatchArray`."""

    schunk = blosc2.schunk_from_cframe(cframe, copy=copy)
    return BatchArray(_from_schunk=schunk)
