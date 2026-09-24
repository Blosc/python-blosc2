"""Internal read-only BatchArray adapter over remote compressed batches."""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import blosc2
from blosc2.batch_array import (
    _BATCHARRAY_VLMETA_KEY,
    Batch,
    BatchArray,
    BatchArrayItems,
)


class _RemoteBatch(Batch):
    def __getitem__(self, index):
        self._parent._check_open()
        return super().__getitem__(index)

    def __len__(self):
        self._parent._check_open()
        return super().__len__()

    def _payloads(self):
        self._parent._check_open()
        payloads = getattr(self, "_remote_payloads", None)
        if payloads is None:
            payloads = blosc2.blosc2_ext.vldecompress(self._lazybatch)
            self._remote_payloads = payloads
        return payloads

    def _decode_items(self):
        if self._items is None:
            self._items = [
                item for payload in self._payloads() for item in self._parent._deserialize_block(payload)
            ]
        return self._items

    def _get_block(self, block_index):
        if self._cached_block_index != block_index or self._cached_block is None:
            self._cached_block = self._parent._deserialize_block(self._payloads()[block_index])
            self._cached_block_index = block_index
        return self._cached_block

    def _get_block_item(self, block_index, item_index):
        if self._parent._serializer == "arrow":
            return self._parent._deserialize_arrow_block_item(self._payloads()[block_index], item_index)
        return self._get_block(block_index)[item_index]


class _RemoteBatchArray(BatchArray):
    """The BatchArray read surface needed by remote CTable wrappers."""

    def __init__(self, source, column, check_open=None):
        self._remote = True
        self._owner_check = check_open or (lambda: None)
        try:
            metadata = source.meta["batcharray"]
        except KeyError as exc:
            raise ValueError(f"Remote batch column {column!r} is not tagged as a BatchArray") from exc
        self._serializer = metadata.get("serializer", "msgpack")
        self._items_per_block = metadata.get("items_per_block")
        self._arrow_schema = metadata.get("arrow_schema")
        self._arrow_schema_obj = None
        self._source = source
        self.schunk = _RemoteBatchSChunk(source)
        self.mode = "r"
        self.mmap_mode = None
        self._batch_lengths = self._validated_lengths(column)
        self._items = BatchArrayItems(self)
        self._item_prefix_sums = None
        self._validate_tag()

    def _check_open(self):
        self._owner_check()
        self._source._check()

    def _validated_lengths(self, column):
        metadata = self.schunk.vlmeta.get(_BATCHARRAY_VLMETA_KEY, {})
        lengths = metadata.get("batch_lengths")
        count = self.schunk.nchunks
        if count == 0 and lengths in (None, []):
            return []
        if not isinstance(lengths, list) or len(lengths) != count:
            raise ValueError(f"Remote batch column {column!r} requires one persisted batch length per batch")
        if any(isinstance(length, bool) or not isinstance(length, int) or length < 0 for length in lengths):
            raise ValueError(f"Remote batch column {column!r} has invalid persisted batch lengths")
        if sum(lengths) > self.schunk.nbytes:
            raise ValueError(f"Remote batch column {column!r} has invalid persisted batch length bounds")
        return lengths

    def _get_batch(self, index):
        self._check_open()
        return _RemoteBatch(self, index, self._source.get_chunk(index))

    def _deserialize_msgpack_block(self, payload):
        from blosc2.msgpack_utils import _safe_msgpack_unpackb

        return _safe_msgpack_unpackb(payload)

    def _check_writable(self):
        self._check_open()
        raise ValueError("Remote CTable batch columns are read-only")

    def __len__(self):
        self._check_open()
        return super().__len__()

    @property
    def meta(self):
        self._check_open()
        return super().meta

    @property
    def vlmeta(self):
        self._check_open()
        return super().vlmeta

    @property
    def nbytes(self):
        self._check_open()
        return super().nbytes

    @property
    def cbytes(self):
        self._check_open()
        return super().cbytes

    @property
    def cratio(self):
        self._check_open()
        return super().cratio


class _RemoteBatchCache:
    """Compressed batch retention using the owner's aggregate cache budget."""

    def __init__(self, source, key, coordinator, path=None, *, read_only=False, artifact=None):
        self._source = source
        self._cache_key = key
        self._cache_coordinator = coordinator
        self._path = None if path is None else Path(path)
        self._read_only = read_only
        self._artifact = artifact
        self._memory = {}
        self._cache_sizes = {}
        self._cache_lru = OrderedDict()
        if self._path is not None:
            if not read_only:
                self._path.mkdir(parents=True, exist_ok=True)
            for file in sorted(self._path.glob("*.chunk"), key=lambda item: int(item.stem)):
                index = int(file.stem)
                if 0 <= index < len(source.offsets):
                    self._cache_sizes[index] = file.stat().st_size
                    self._cache_lru[index] = None
        if artifact is not None:
            for index, info in artifact[1].items():
                if not 0 <= index < len(source.offsets):
                    raise ValueError("Invalid cached batch index")
                self._cache_sizes[index] = info["length"]
                self._cache_lru[index] = None
        coordinator.register(self)

    def __getattr__(self, name):
        return getattr(self._source, name)

    def _file(self, index):
        return self._path / f"{index}.chunk"

    def read_cached_chunk(self, index):
        """Read retained bytes without fetching or changing cache recency."""
        if self._artifact is not None:
            path, offsets = self._artifact
            info = offsets[index]
            with open(path, "rb") as file:
                file.seek(info["offset"])
                data = file.read(info["length"])
            if len(data) != info["length"]:
                raise ValueError("Truncated cached batch")
            return data
        return self._file(index).read_bytes() if self._path is not None else self._memory[index]

    def get_chunk(self, index):
        self._source._check()
        if index in self._cache_sizes:
            chunk = self.read_cached_chunk(index)
        else:
            if self._cache_coordinator.cached_only:
                from blosc2.remote_store import CacheMiss

                raise CacheMiss
            chunk = self._source.get_chunk(index)
            if self._read_only:
                return chunk
            if self._path is None:
                self._memory[index] = chunk
            else:
                from blosc2.remote_store_cache import atomic_write

                atomic_write(self._file(index), chunk)
            self._cache_sizes[index] = len(chunk)
        self._cache_lru.pop(index, None)
        self._cache_lru[index] = None
        self._cache_coordinator.touch(self, index)
        self._cache_coordinator.enforce()
        return chunk

    def _sync_evictions(self):
        pass

    def _retained_cache_bytes(self):
        return sum(self._cache_sizes.values())

    def _trim_cache(self, target_bytes, *, max_chunks=None):
        if self._read_only and self._retained_cache_bytes() > target_bytes:
            raise ValueError("Cannot trim an immutable batch cache")
        removed = []
        while self._retained_cache_bytes() > target_bytes and self._cache_lru:
            if max_chunks is not None and len(removed) >= max_chunks:
                break
            index = next(iter(self._cache_lru))
            if self._path is None:
                self._memory.pop(index, None)
            else:
                os.unlink(self._file(index))
            self._cache_lru.pop(index)
            self._cache_sizes.pop(index)
            self._cache_coordinator.forget(self, index)
            removed.append(index)
        return tuple(removed)


class _RemoteBatchSChunk:
    def __init__(self, source):
        from blosc2.remote_array import RemoteMetadataMapping

        self._source = source
        self.meta = RemoteMetadataMapping(source.meta)
        self.vlmeta = RemoteMetadataMapping(source.vlmeta)
        self.mode = "r"
        self.mmap_mode = None
        self.nchunks = len(source.offsets)
        self.nbytes = int(source.header[4])
        self.cbytes = int(source.header[5])
        self.typesize = 1
        self.urlpath = None
        self.contiguous = True
        self.cparams = blosc2.CParams(typesize=1)
        self.dparams = blosc2.DParams()

    @property
    def cratio(self):
        return self.nbytes / self.cbytes if self.cbytes else 0.0

    def get_chunk(self, index):
        return self._source.get_chunk(index)

    get_lazychunk = get_chunk

    def get_vlblock(self, chunk, block):
        return blosc2.blosc2_ext.vldecompress(self.get_chunk(chunk))[block]
