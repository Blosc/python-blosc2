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
from blosc2.info import InfoReporter, format_nbytes_info

_OBJECTARRAY_META = {"version": 2, "serializer": "msgpack", "format": "batched_vlblocks"}
_OBJECTARRAY_LAYOUT_KEY = "objectarray"


def _check_serialized_size(buffer: bytes) -> None:
    if len(buffer) > blosc2.MAX_BUFFERSIZE:
        raise ValueError(f"Serialized objects cannot be larger than {blosc2.MAX_BUFFERSIZE} bytes")


class Batch(Sequence[Any]):
    """A lazy sequence of Python objects stored in one ObjectArray chunk."""

    def __init__(self, parent: ObjectArray, nchunk: int, lazychunk: bytes) -> None:
        self._parent = parent
        self._nchunk = nchunk
        self._lazychunk = lazychunk
        self._blocks: list[list[Any]] | None = None
        self._nbytes, self._cbytes, self._nblocks = blosc2.get_cbuffer_sizes(lazychunk)

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("Batch indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Batch index out of range")
        return index

    def _decode_blocks(self) -> list[list[Any]]:
        if self._blocks is None:
            self._blocks = self._parent._decode_blocks(self._nchunk)
        return self._blocks

    def __getitem__(self, index: int | slice) -> Any | list[Any]:
        blocks = self._decode_blocks()
        if isinstance(index, slice):
            flat_items = [item for block in blocks for item in block]
            return flat_items[index]
        index = self._normalize_index(index)
        blocksize = self._parent.blocksize
        if blocksize is None:
            raise RuntimeError("ObjectArray blocksize is not initialized")
        block_index, item_index = divmod(index, blocksize)
        return blocks[block_index][item_index]

    def __len__(self) -> int:
        chunksize = self._parent.chunksize
        if chunksize is None:
            return self._nblocks
        return chunksize

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


class ObjectArray:
    """A batched variable-length array backed by an :class:`blosc2.SChunk`."""

    @staticmethod
    def _set_typesize_one(cparams: blosc2.CParams | dict | None) -> blosc2.CParams | dict:
        auto_use_dict = cparams is None
        if cparams is None:
            cparams = blosc2.CParams()
        elif isinstance(cparams, blosc2.CParams):
            cparams = copy.deepcopy(cparams)
        else:
            cparams = dict(cparams)
            auto_use_dict = "use_dict" not in cparams

        if isinstance(cparams, blosc2.CParams):
            cparams.typesize = 1
            if auto_use_dict and cparams.codec == blosc2.Codec.ZSTD and cparams.clevel > 0:
                # ObjectArray stores many small serialized payloads, where Zstd dicts help materially.
                cparams.use_dict = True
        else:
            cparams["typesize"] = 1
            codec = cparams.get("codec", blosc2.Codec.ZSTD)
            clevel = cparams.get("clevel", 5)
            if auto_use_dict and codec == blosc2.Codec.ZSTD and clevel > 0:
                # ObjectArray stores many small serialized payloads, where Zstd dicts help materially.
                cparams["use_dict"] = True
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
            raise ValueError("For ObjectArray containers, mmap_mode must be None or 'r'")
        if storage.mmap_mode == "r" and storage.mode != "r":
            raise ValueError("For ObjectArray containers, mmap_mode='r' requires mode='r'")

    def _attach_schunk(self, schunk: blosc2.SChunk) -> None:
        self.schunk = schunk
        self.mode = schunk.mode
        self.mmap_mode = getattr(schunk, "mmap_mode", None)
        self._validate_tag()
        self._load_layout()

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
        blocksize: int | None = None,
        _from_schunk: blosc2.SChunk | None = None,
        **kwargs: Any,
    ) -> None:
        self._chunksize: int | None = chunksize
        self._blocksize: int | None = blocksize
        self._layout_format: str | None = None
        if _from_schunk is not None:
            if chunksize is not None or blocksize is not None:
                raise ValueError("Cannot pass `chunksize` or `blocksize` together with `_from_schunk`")
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
            raise ValueError(f"Unsupported ObjectArray keyword argument(s): {unexpected}")

        self._validate_storage(storage)
        cparams = self._set_typesize_one(cparams)

        if dparams is None:
            dparams = blosc2.DParams()

        if self._maybe_open_existing(storage):
            return

        fixed_meta = dict(storage.meta or {})
        fixed_meta["objectarray"] = dict(_OBJECTARRAY_META)
        storage.meta = fixed_meta
        schunk = blosc2.SChunk(chunksize=-1, data=None, cparams=cparams, dparams=dparams, storage=storage)
        self._attach_schunk(schunk)
        if self._chunksize is not None or self._blocksize is not None:
            self._store_layout()

    def _validate_tag(self) -> None:
        if "objectarray" not in self.schunk.meta:
            raise ValueError("The supplied SChunk is not tagged as an ObjectArray")

    def _load_layout(self) -> None:
        layout = None
        self._layout_format = None
        if _OBJECTARRAY_LAYOUT_KEY in self.vlmeta:
            layout = self.vlmeta[_OBJECTARRAY_LAYOUT_KEY]
        if isinstance(layout, dict):
            self._chunksize = layout.get("chunksize")
            self._blocksize = layout.get("blocksize")
            self._layout_format = layout.get("format", "batched_vlblocks")
            return
        if len(self) == 0:
            return
        raise ValueError("ObjectArray layout metadata is missing")

    def _store_layout(self) -> None:
        if self._chunksize is None or self.mode == "r":
            return
        layout = {
            "version": 1,
            "chunksize": self._chunksize,
            "blocksize": self._blocksize,
            "format": self._layout_format or "batched_vlblocks",
            "sizing_policy": "l2_cache_prefix",
        }
        self.vlmeta[_OBJECTARRAY_LAYOUT_KEY] = layout

    def _check_writable(self) -> None:
        if self.mode == "r":
            raise ValueError("Cannot modify an ObjectArray opened in read-only mode")

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("ObjectArray indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("ObjectArray index out of range")
        return index

    def _normalize_insert_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("ObjectArray indices must be integers")
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
            raise TypeError("ObjectArray entries must be sequences of Python objects")
        if not isinstance(value, Sequence):
            raise TypeError("ObjectArray entries must be sequences of Python objects")
        values = list(value)
        if len(values) == 0:
            raise ValueError("ObjectArray entries cannot be empty")
        return values

    def _ensure_layout_for_batch(self, batch: list[Any]) -> None:
        if self._chunksize is None:
            self._chunksize = len(batch)
        if len(batch) != self._chunksize:
            raise ValueError(f"ObjectArray entries must contain exactly {self._chunksize} objects")
        if self._blocksize is None:
            payload_sizes = [len(msgpack_packb(item)) for item in batch]
            self._blocksize = self._guess_blocksize(payload_sizes)
        self._store_layout()

    def _guess_blocksize(self, payload_sizes: list[int]) -> int:
        if not payload_sizes:
            raise ValueError("ObjectArray entries cannot be empty")
        l2_cache_size = blosc2.cpu_info.get("l2_cache_size")
        if not isinstance(l2_cache_size, int) or l2_cache_size <= 0:
            return len(payload_sizes)
        total = 0
        count = 0
        for payload_size in payload_sizes:
            if count > 0 and total + payload_size > l2_cache_size:
                break
            total += payload_size
            count += 1
        if count == 0:
            count = 1
        return min(count, len(payload_sizes))

    def _serialize_batch(self, value: object) -> list[Any]:
        batch = self._normalize_batch(value)
        self._ensure_layout_for_batch(batch)
        return batch

    def _serialize_block(self, items: list[Any]) -> bytes:
        payload = msgpack_packb(items)
        _check_serialized_size(payload)
        return payload

    def _vl_cparams_kwargs(self) -> dict[str, Any]:
        return asdict(self.schunk.cparams)

    def _vl_dparams_kwargs(self) -> dict[str, Any]:
        return asdict(self.schunk.dparams)

    def _compress_batch(self, batch: list[Any]) -> bytes:
        if self._blocksize is None:
            raise RuntimeError("ObjectArray blocksize is not initialized")
        blocks = [
            self._serialize_block(batch[i : i + self._blocksize])
            for i in range(0, len(batch), self._blocksize)
        ]
        return blosc2.blosc2_ext.vlcompress(blocks, **self._vl_cparams_kwargs())

    def _decode_blocks(self, nchunk: int) -> list[list[Any]]:
        block_payloads = blosc2.blosc2_ext.vldecompress(
            self.schunk.get_chunk(nchunk), **self._vl_dparams_kwargs()
        )
        return [msgpack_unpackb(payload) for payload in block_payloads]

    def _get_batch(self, index: int) -> Batch:
        return Batch(self, index, self.schunk.get_lazychunk(index))

    def _batch_lengths(self) -> list[int]:
        if self.chunksize is not None:
            return [self.chunksize for _ in range(len(self))]
        return [len(self[i]) for i in range(len(self))]

    def append(self, value: object) -> int:
        """Append one batch and return the new number of entries."""
        self._check_writable()
        batch = self._serialize_batch(value)
        chunk = self._compress_batch(batch)
        return self.schunk.append_chunk(chunk)

    def insert(self, index: int, value: object) -> int:
        """Insert one batch at ``index`` and return the new number of entries."""
        self._check_writable()
        index = self._normalize_insert_index(index)
        batch = self._serialize_batch(value)
        chunk = self._compress_batch(batch)
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
            raise NotImplementedError("Slicing is not supported for ObjectArray")
        index = self._normalize_index(index)
        value = self[index][:]
        self.schunk.delete_chunk(index)
        return value

    def extend(self, values: object) -> None:
        """Append all batches from an iterable."""
        self._check_writable()
        for value in values:
            batch = self._serialize_batch(value)
            chunk = self._compress_batch(batch)
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
        self._store_layout()

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
                    batch = self._serialize_batch(item)
                    chunk = self._compress_batch(batch)
                    self.schunk.insert_chunk(start + offset, chunk)
                return
            if len(values) != len(indices):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values)} to extended slice of size {len(indices)}"
                )
            for idx, item in zip(indices, values, strict=True):
                batch = self._serialize_batch(item)
                chunk = self._compress_batch(batch)
                self.schunk.update_chunk(idx, chunk)
            return
        self._check_writable()
        index = self._normalize_index(index)
        batch = self._serialize_batch(value)
        chunk = self._compress_batch(batch)
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
        return self._chunksize

    @property
    def blocksize(self) -> int:
        return self._blocksize

    @property
    def typesize(self) -> int:
        return self.schunk.typesize

    @property
    def nbytes(self) -> int:
        return self.schunk.nbytes

    @property
    def cbytes(self) -> int:
        return self.schunk.cbytes

    @property
    def cratio(self) -> float:
        return self.schunk.cratio

    @property
    def urlpath(self) -> str | None:
        return self.schunk.urlpath

    @property
    def contiguous(self) -> bool:
        return self.schunk.contiguous

    @property
    def info(self) -> InfoReporter:
        """Print information about this ObjectArray."""
        return InfoReporter(self)

    @property
    def info_items(self) -> list:
        """A list of tuples with summary information about this ObjectArray."""
        batch_lengths = self._batch_lengths()
        nitems = sum(batch_lengths)
        avg_batch_len = nitems / len(batch_lengths) if batch_lengths else 0.0
        return [
            ("type", f"{self.__class__.__name__}"),
            ("nbatches", len(self)),
            ("chunksize", self.chunksize),
            ("blocksize", self.blocksize),
            ("nitems", nitems),
            ("batch_len_min", min(batch_lengths) if batch_lengths else 0),
            ("batch_len_max", max(batch_lengths) if batch_lengths else 0),
            ("batch_len_avg", f"{avg_batch_len:.2f}"),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.2f}"),
            ("cparams", self.cparams),
            ("dparams", self.dparams),
        ]

    def to_cframe(self) -> bytes:
        return self.schunk.to_cframe()

    def copy(self, **kwargs: Any) -> ObjectArray:
        """Create a copy of the container with optional constructor overrides."""
        if "meta" in kwargs:
            raise ValueError("meta should not be passed to copy")

        kwargs["cparams"] = kwargs.get("cparams", copy.deepcopy(self.cparams))
        kwargs["dparams"] = kwargs.get("dparams", copy.deepcopy(self.dparams))
        kwargs["chunksize"] = kwargs.get("chunksize", self.chunksize)
        kwargs["blocksize"] = kwargs.get("blocksize", self.blocksize)

        if "storage" not in kwargs:
            kwargs["meta"] = self._copy_meta()
            kwargs["contiguous"] = kwargs.get("contiguous", self.schunk.contiguous)
            if "urlpath" in kwargs and "mode" not in kwargs:
                kwargs["mode"] = "w"

        out = ObjectArray(**kwargs)
        if "storage" not in kwargs and len(self.vlmeta) > 0:
            for key, value in self.vlmeta.getall().items():
                out.vlmeta[key] = value
        out.extend(self)
        return out

    def __enter__(self) -> ObjectArray:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __repr__(self) -> str:
        return f"ObjectArray(len={len(self)}, urlpath={self.urlpath!r})"


def objectarray_from_cframe(cframe: bytes, copy: bool = True) -> ObjectArray:
    """Deserialize a CFrame buffer into a :class:`ObjectArray`."""

    schunk = blosc2.schunk_from_cframe(cframe, copy=copy)
    return ObjectArray(_from_schunk=schunk)
