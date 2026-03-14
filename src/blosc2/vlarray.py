#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import copy
import pathlib
from typing import TYPE_CHECKING, Any

from msgpack import packb, unpackb

import blosc2
from blosc2 import blosc2_ext

if TYPE_CHECKING:
    from collections.abc import Iterator

    from blosc2.schunk import SChunk

_VLARRAY_META = {"version": 1, "serializer": "msgpack"}


def _check_serialized_size(buffer: bytes) -> None:
    if len(buffer) > blosc2.MAX_BUFFERSIZE:
        raise ValueError(f"Serialized objects cannot be larger than {blosc2.MAX_BUFFERSIZE} bytes")


class VLArray:
    """A variable-length array backed by an :class:`blosc2.SChunk`."""

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
            raise ValueError("For VLArray containers, mmap_mode must be None or 'r'")
        if storage.mmap_mode == "r" and storage.mode != "r":
            raise ValueError("For VLArray containers, mmap_mode='r' requires mode='r'")

    def _attach_schunk(self, schunk: SChunk) -> None:
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
        _from_schunk: SChunk | None = None,
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
            raise ValueError(f"Unsupported VLArray keyword argument(s): {unexpected}")

        self._validate_storage(storage)
        cparams = self._set_typesize_one(cparams)

        if dparams is None:
            dparams = blosc2.DParams()

        if self._maybe_open_existing(storage):
            return

        fixed_meta = dict(storage.meta or {})
        fixed_meta["vlarray"] = dict(_VLARRAY_META)
        storage.meta = fixed_meta
        if chunksize is None:
            chunksize = -1
        schunk = blosc2.SChunk(
            chunksize=chunksize, data=None, cparams=cparams, dparams=dparams, storage=storage
        )
        self._attach_schunk(schunk)

    def _validate_tag(self) -> None:
        if "vlarray" not in self.schunk.meta:
            raise ValueError("The supplied SChunk is not tagged as a VLArray")

    def _check_writable(self) -> None:
        if self.mode == "r":
            raise ValueError("Cannot modify a VLArray opened in read-only mode")

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("VLArray indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("VLArray index out of range")
        return index

    def _normalize_insert_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("VLArray indices must be integers")
        if index < 0:
            index += len(self)
            if index < 0:
                return 0
        if index > len(self):
            return len(self)
        return index

    def _serialize(self, value: Any) -> bytes:
        payload = packb(value, default=blosc2_ext.encode_tuple, strict_types=True, use_bin_type=True)
        _check_serialized_size(payload)
        return payload

    def _compress(self, payload: bytes) -> bytes:
        return blosc2.compress2(payload, cparams=self.schunk.cparams)

    def append(self, value: Any) -> int:
        """Append one value and return the new number of entries."""
        self._check_writable()
        chunk = self._compress(self._serialize(value))
        return self.schunk.append_chunk(chunk)

    def insert(self, index: int, value: Any) -> int:
        """Insert one value at ``index`` and return the new number of entries."""
        self._check_writable()
        index = self._normalize_insert_index(index)
        chunk = self._compress(self._serialize(value))
        return self.schunk.insert_chunk(index, chunk)

    def delete(self, index: int) -> int:
        """Delete the value at ``index`` and return the new number of entries."""
        self._check_writable()
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for VLArray")
        index = self._normalize_index(index)
        return self.schunk.delete_chunk(index)

    def pop(self, index: int = -1) -> Any:
        """Remove and return the value at ``index``."""
        self._check_writable()
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for VLArray")
        index = self._normalize_index(index)
        value = self[index]
        self.schunk.delete_chunk(index)
        return value

    def extend(self, values: object) -> None:
        """Append all values from an iterable."""
        self._check_writable()
        for value in values:
            chunk = self._compress(self._serialize(value))
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

    def __getitem__(self, index: int) -> Any:
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for VLArray")
        index = self._normalize_index(index)
        payload = self.schunk.decompress_chunk(index)
        return unpackb(payload, list_hook=blosc2_ext.decode_tuple)

    def __setitem__(self, index: int, value: Any) -> None:
        if isinstance(index, slice):
            raise NotImplementedError("Slicing is not supported for VLArray")
        self._check_writable()
        index = self._normalize_index(index)
        chunk = self._compress(self._serialize(value))
        self.schunk.update_chunk(index, chunk)

    def __delitem__(self, index: int) -> None:
        self.delete(index)

    def __len__(self) -> int:
        return self.schunk.nchunks

    def __iter__(self) -> Iterator[Any]:
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

    def to_cframe(self) -> bytes:
        return self.schunk.to_cframe()

    def __enter__(self) -> VLArray:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __repr__(self) -> str:
        return f"VLArray(len={len(self)}, urlpath={self.urlpath!r})"


def vlarray_from_cframe(cframe: bytes, copy: bool = False) -> VLArray:
    """Deserialize a CFrame buffer into a :class:`VLArray`."""

    schunk = blosc2.schunk_from_cframe(cframe, copy=copy)
    return VLArray(_from_schunk=schunk)
