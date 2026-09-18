"""Internal read-only BatchArray adapter over remote compressed batches."""

from __future__ import annotations

import blosc2
from blosc2.batch_array import (
    _BATCHARRAY_VLMETA_KEY,
    Batch,
    BatchArray,
    BatchArrayItems,
)


class _RemoteBatch(Batch):
    def _payloads(self):
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

    def __init__(self, source, column):
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
        return _RemoteBatch(self, index, self._source.get_chunk(index))

    def _check_writable(self):
        raise ValueError("Cannot modify a remote BatchArray")


class _RemoteBatchSChunk:
    def __init__(self, source):
        self._source = source
        self.meta = source.meta
        self.vlmeta = source.vlmeta
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
