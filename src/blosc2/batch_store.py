#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import copy
import pathlib
import statistics
from collections.abc import Iterator, Sequence
from dataclasses import asdict
from typing import Any

import blosc2
from blosc2._msgpack_utils import msgpack_packb, msgpack_unpackb
from blosc2.info import InfoReporter, format_nbytes_info

_BATCHSTORE_META = {"version": 1, "serializer": "msgpack", "max_blocksize": None}


def _check_serialized_size(buffer: bytes) -> None:
    if len(buffer) > blosc2.MAX_BUFFERSIZE:
        raise ValueError(f"Serialized objects cannot be larger than {blosc2.MAX_BUFFERSIZE} bytes")


class Batch(Sequence[Any]):
    """A lazy sequence of Python objects stored in one BatchStore batch."""

    def __init__(self, parent: BatchStore, nbatch: int, lazybatch: bytes) -> None:
        self._parent = parent
        self._nbatch = nbatch
        self._lazybatch = lazybatch
        self._items: list[Any] | None = None
        self._cached_block_index: int | None = None
        self._cached_block: list[Any] | None = None
        self._nbytes, self._cbytes, self._nblocks = blosc2.get_cbuffer_sizes(lazybatch)

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("Batch indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Batch index out of range")
        return index

    def _decode_items(self) -> list[Any]:
        if self._items is None:
            blocks = self._parent._decode_blocks(self._nbatch)
            self._items = [item for block in blocks for item in block]
        return self._items

    def _get_block(self, block_index: int) -> list[Any]:
        if self._cached_block_index == block_index and self._cached_block is not None:
            return self._cached_block
        block = msgpack_unpackb(self._parent.schunk.get_vlblock(self._nbatch, block_index))
        self._cached_block_index = block_index
        self._cached_block = block
        return block

    def __getitem__(self, index: int | slice) -> Any | list[Any]:
        if isinstance(index, slice):
            items = self._decode_items()
            return items[index]
        if index < 0:
            items = self._decode_items()
            index = self._normalize_index(index)
            return items[index]
        max_blocksize = self._parent.max_blocksize
        if max_blocksize is not None:
            block_index, item_index = divmod(index, max_blocksize)
            if block_index >= self._nblocks:
                raise IndexError("Batch index out of range")
            block = self._get_block(block_index)
            try:
                return block[item_index]
            except IndexError as exc:
                raise IndexError("Batch index out of range") from exc
        items = self._decode_items()
        index = self._normalize_index(index)
        return items[index]

    def __len__(self) -> int:
        return len(self._decode_items())

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    @property
    def lazybatch(self) -> bytes:
        return self._lazybatch

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


class BatchStore:
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
            raise ValueError("For BatchStore containers, mmap_mode must be None or 'r'")
        if storage.mmap_mode == "r" and storage.mode != "r":
            raise ValueError("For BatchStore containers, mmap_mode='r' requires mode='r'")

    def _attach_schunk(self, schunk: blosc2.SChunk) -> None:
        self.schunk = schunk
        self.mode = schunk.mode
        self.mmap_mode = getattr(schunk, "mmap_mode", None)
        try:
            batchstore_meta = self.schunk.meta["batchstore"]
        except KeyError:
            batchstore_meta = {}
        self._max_blocksize = batchstore_meta.get("max_blocksize", self._max_blocksize)
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
        max_blocksize: int | None = None,
        _from_schunk: blosc2.SChunk | None = None,
        **kwargs: Any,
    ) -> None:
        if max_blocksize is not None and max_blocksize <= 0:
            raise ValueError("max_blocksize must be a positive integer")
        self._max_blocksize: int | None = max_blocksize
        if _from_schunk is not None:
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
            raise ValueError(f"Unsupported BatchStore keyword argument(s): {unexpected}")

        self._validate_storage(storage)
        cparams = self._set_typesize_one(cparams)

        if dparams is None:
            dparams = blosc2.DParams()

        if self._maybe_open_existing(storage):
            return

        fixed_meta = dict(storage.meta or {})
        fixed_meta["batchstore"] = {**_BATCHSTORE_META, "max_blocksize": self._max_blocksize}
        storage.meta = fixed_meta
        schunk = blosc2.SChunk(chunksize=-1, data=None, cparams=cparams, dparams=dparams, storage=storage)
        self._attach_schunk(schunk)

    def _validate_tag(self) -> None:
        if "batchstore" not in self.schunk.meta:
            raise ValueError("The supplied SChunk is not tagged as a BatchStore")

    def _check_writable(self) -> None:
        if self.mode == "r":
            raise ValueError("Cannot modify a BatchStore opened in read-only mode")

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("BatchStore indices must be integers")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("BatchStore index out of range")
        return index

    def _normalize_insert_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("BatchStore indices must be integers")
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
            raise TypeError("BatchStore entries must be sequences of Python objects")
        if not isinstance(value, Sequence):
            raise TypeError("BatchStore entries must be sequences of Python objects")
        values = list(value)
        if len(values) == 0:
            raise ValueError("BatchStore entries cannot be empty")
        return values

    def _ensure_layout_for_batch(self, batch: list[Any]) -> None:
        if self._max_blocksize is None:
            payload_sizes = [len(msgpack_packb(item)) for item in batch]
            self._max_blocksize = self._guess_blocksize(payload_sizes)
            self._persist_max_blocksize()

    def _persist_max_blocksize(self) -> None:
        if self._max_blocksize is None or len(self) > 0:
            return
        storage = self._make_storage()
        fixed_meta = dict(storage.meta or {})
        fixed_meta["batchstore"] = {
            **dict(fixed_meta.get("batchstore", {})),
            "max_blocksize": self._max_blocksize,
        }
        storage.meta = fixed_meta
        schunk = blosc2.SChunk(
            chunksize=-1,
            data=None,
            cparams=copy.deepcopy(self.cparams),
            dparams=copy.deepcopy(self.dparams),
            storage=storage,
        )
        self._attach_schunk(schunk)

    def _guess_blocksize(self, payload_sizes: list[int]) -> int:
        if not payload_sizes:
            raise ValueError("BatchStore entries cannot be empty")
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
        if self._max_blocksize is None:
            raise RuntimeError("BatchStore max_blocksize is not initialized")
        blocks = [
            self._serialize_block(batch[i : i + self._max_blocksize])
            for i in range(0, len(batch), self._max_blocksize)
        ]
        return blosc2.blosc2_ext.vlcompress(blocks, **self._vl_cparams_kwargs())

    def _decode_blocks(self, nbatch: int) -> list[list[Any]]:
        block_payloads = blosc2.blosc2_ext.vldecompress(
            self.schunk.get_chunk(nbatch), **self._vl_dparams_kwargs()
        )
        return [msgpack_unpackb(payload) for payload in block_payloads]

    def _get_batch(self, index: int) -> Batch:
        return Batch(self, index, self.schunk.get_lazychunk(index))

    def append(self, value: object) -> int:
        """Append one batch and return the new number of entries."""
        self._check_writable()
        batch = self._serialize_batch(value)
        batch_payload = self._compress_batch(batch)
        return self.schunk.append_chunk(batch_payload)

    def insert(self, index: int, value: object) -> int:
        """Insert one batch at ``index`` and return the new number of entries."""
        self._check_writable()
        index = self._normalize_insert_index(index)
        batch = self._serialize_batch(value)
        batch_payload = self._compress_batch(batch)
        return self.schunk.insert_chunk(index, batch_payload)

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
            raise NotImplementedError("Slicing is not supported for BatchStore")
        index = self._normalize_index(index)
        value = self[index][:]
        self.schunk.delete_chunk(index)
        return value

    def extend(self, values: object) -> None:
        """Append all batches from an iterable."""
        self._check_writable()
        for value in values:
            batch = self._serialize_batch(value)
            batch_payload = self._compress_batch(batch)
            self.schunk.append_chunk(batch_payload)

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
                    batch = self._serialize_batch(item)
                    batch_payload = self._compress_batch(batch)
                    self.schunk.insert_chunk(start + offset, batch_payload)
                return
            if len(values) != len(indices):
                raise ValueError(
                    f"attempt to assign sequence of size {len(values)} to extended slice of size {len(indices)}"
                )
            for idx, item in zip(indices, values, strict=True):
                batch = self._serialize_batch(item)
                batch_payload = self._compress_batch(batch)
                self.schunk.update_chunk(idx, batch_payload)
            return
        self._check_writable()
        index = self._normalize_index(index)
        batch = self._serialize_batch(value)
        batch_payload = self._compress_batch(batch)
        self.schunk.update_chunk(index, batch_payload)

    def __delitem__(self, index: int | slice) -> None:
        self.delete(index)

    def __len__(self) -> int:
        return self.schunk.nchunks

    def iter_batches(self) -> Iterator[Batch]:
        for i in range(len(self)):
            yield self[i]

    def iter_objects(self) -> Iterator[Any]:
        for batch in self.iter_batches():
            yield from batch

    def __iter__(self) -> Iterator[Batch]:
        yield from self.iter_batches()

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
    def max_blocksize(self) -> int | None:
        return self._max_blocksize

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
        """Print information about this BatchStore."""
        return InfoReporter(self)

    @property
    def info_items(self) -> list:
        """A list of tuples with summary information about this BatchStore."""
        batch_sizes = [len(batch) for batch in self.iter_batches()]
        if batch_sizes:
            batch_stats = (
                f"mean={statistics.fmean(batch_sizes):.2f}, max={max(batch_sizes)}, min={min(batch_sizes)}"
            )
        else:
            batch_stats = "n/a"
        return [
            ("type", f"{self.__class__.__name__}"),
            ("nbatches", len(self)),
            ("batch stats", batch_stats),
            ("max_blocksize", self.max_blocksize),
            ("nitems", sum(batch_sizes)),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.2f}"),
            ("cparams", self.cparams),
            ("dparams", self.dparams),
        ]

    def to_cframe(self) -> bytes:
        return self.schunk.to_cframe()

    def copy(self, **kwargs: Any) -> BatchStore:
        """Create a copy of the container with optional constructor overrides."""
        if "meta" in kwargs:
            raise ValueError("meta should not be passed to copy")
        kwargs["cparams"] = kwargs.get("cparams", copy.deepcopy(self.cparams))
        kwargs["dparams"] = kwargs.get("dparams", copy.deepcopy(self.dparams))
        kwargs["max_blocksize"] = kwargs.get("max_blocksize", self.max_blocksize)

        if "storage" not in kwargs:
            kwargs["meta"] = self._copy_meta()
            kwargs["contiguous"] = kwargs.get("contiguous", self.schunk.contiguous)
            if "urlpath" in kwargs and "mode" not in kwargs:
                kwargs["mode"] = "w"

        out = BatchStore(**kwargs)
        if "storage" not in kwargs and len(self.vlmeta) > 0:
            for key, value in self.vlmeta.getall().items():
                out.vlmeta[key] = value
        out.extend(self.iter_batches())
        return out

    def __enter__(self) -> BatchStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __repr__(self) -> str:
        return f"BatchStore(len={len(self)}, urlpath={self.urlpath!r})"
