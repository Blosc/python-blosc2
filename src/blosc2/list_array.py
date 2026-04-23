#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import copy
from bisect import bisect_right
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Any

import numpy as np

import blosc2
from blosc2.batch_array import BatchArray
from blosc2.info import InfoReporter, format_nbytes_info
from blosc2.schema import ListSpec, SchemaSpec
from blosc2.schema import list as list_spec_builder
from blosc2.vlarray import VLArray

_SUPPORTED_SERIALIZERS = {"msgpack", "arrow"}
_SUPPORTED_STORAGES = {"batch", "vl"}


def _spec_label(spec: SchemaSpec) -> str:
    if isinstance(spec, ListSpec):
        return spec.display_label()
    meta = spec.to_metadata_dict()
    kind = meta.get("kind", type(spec).__name__)
    if kind == "string":
        return "string"
    if kind == "bytes":
        return "bytes"
    return str(kind)


@lru_cache(maxsize=1)
def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError("ListArray serializer='arrow' requires pyarrow") from exc
    return pa


def _validate_list_spec(spec: ListSpec) -> None:
    if spec.storage not in _SUPPORTED_STORAGES:
        raise ValueError(f"Unsupported list storage: {spec.storage!r}")
    if spec.serializer not in _SUPPORTED_SERIALIZERS:
        raise ValueError(f"Unsupported list serializer: {spec.serializer!r}")
    if spec.storage == "vl" and spec.serializer != "msgpack":
        raise ValueError("ListArray storage='vl' only supports serializer='msgpack'")
    if spec.serializer == "arrow" and spec.storage != "batch":
        raise ValueError("ListArray serializer='arrow' requires storage='batch'")
    if isinstance(spec.item_spec, ListSpec):
        raise TypeError("Nested list item specs are not supported in V1")
    if spec.batch_rows is not None and spec.batch_rows <= 0:
        raise ValueError("batch_rows must be a positive integer")
    if spec.items_per_block is not None and spec.items_per_block <= 0:
        raise ValueError("items_per_block must be a positive integer")


def _coerce_scalar_item(spec: SchemaSpec, value: Any) -> Any:
    if value is None:
        raise ValueError("ListArray does not support nullable items inside a list in V1")

    if getattr(spec, "python_type", None) is str:
        if not isinstance(value, str):
            value = str(value)
    elif getattr(spec, "python_type", None) is bytes:
        if isinstance(value, str):
            value = value.encode()
        elif not isinstance(value, (bytes, bytearray, memoryview)):
            value = bytes(value)
        value = bytes(value)
    else:
        dtype = getattr(spec, "dtype", None)
        if dtype is None:
            raise TypeError(f"Unsupported list item spec {type(spec).__name__!r}")
        value = np.array(value, dtype=dtype).item()

    ge = getattr(spec, "ge", None)
    if ge is not None and value < ge:
        raise ValueError(f"List item {value!r} violates ge={ge}")
    gt = getattr(spec, "gt", None)
    if gt is not None and value <= gt:
        raise ValueError(f"List item {value!r} violates gt={gt}")
    le = getattr(spec, "le", None)
    if le is not None and value > le:
        raise ValueError(f"List item {value!r} violates le={le}")
    lt = getattr(spec, "lt", None)
    if lt is not None and value >= lt:
        raise ValueError(f"List item {value!r} violates lt={lt}")

    max_length = getattr(spec, "max_length", None)
    min_length = getattr(spec, "min_length", None)
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"List item {value!r} exceeds max_length={max_length}")
    if min_length is not None and len(value) < min_length:
        raise ValueError(f"List item {value!r} is shorter than min_length={min_length}")
    return value


def coerce_list_cell(spec: ListSpec, value: Any) -> list[Any] | None:
    _validate_list_spec(spec)
    if value is None:
        if not spec.nullable:
            raise ValueError("Null list cells are not allowed for this column")
        return None
    if isinstance(value, (str, bytes, bytearray, memoryview)):
        raise TypeError("ListArray cells must be list-like, not strings or bytes")
    if not isinstance(value, Iterable):
        raise TypeError("ListArray cells must be list-like")
    return [_coerce_scalar_item(spec.item_spec, item) for item in list(value)]


class ListArray:
    """A row-oriented container for list-valued data.

    Backed internally by either :class:`blosc2.VLArray` or
    :class:`blosc2.BatchArray`.
    """

    def __init__(
        self,
        spec: ListSpec | None = None,
        *,
        item_spec: SchemaSpec | None = None,
        nullable: bool = False,
        storage: str = "batch",
        serializer: str = "msgpack",
        batch_rows: int | None = None,
        items_per_block: int | None = None,
        _from_schunk=None,
        **kwargs: Any,
    ) -> None:
        if _from_schunk is not None:
            if spec is not None or item_spec is not None or kwargs:
                raise ValueError("Cannot pass schema/storage arguments together with _from_schunk")
            self._init_from_schunk(_from_schunk)
            return

        if spec is None:
            if item_spec is None:
                raise ValueError("ListArray requires either spec=... or item_spec=...")
            spec = list_spec_builder(
                item_spec,
                nullable=nullable,
                storage=storage,
                serializer=serializer,
                batch_rows=batch_rows,
                items_per_block=items_per_block,
            )
        self.spec = spec
        _validate_list_spec(self.spec)
        self._pending_cells: list[list[Any] | None] = []
        self._persisted_row_count = 0
        self._persisted_prefix_cache: list[int] | None = None
        self._cached_batch_index: int | None = None
        self._cached_batch_values: list[list[Any] | None] | None = None

        storage_obj = self._coerce_storage(kwargs)
        fixed_meta = dict(storage_obj.meta or {})
        fixed_meta["listarray"] = self.spec.to_listarray_metadata()
        storage_obj.meta = fixed_meta

        if self.spec.storage == "vl":
            self._backend = VLArray(storage=storage_obj, **kwargs)
        else:
            self._backend = BatchArray(
                storage=storage_obj,
                serializer=self.spec.serializer,
                items_per_block=self.spec.items_per_block,
                **kwargs,
            )
            self._persisted_row_count = self._persisted_rows_count()

    @staticmethod
    def _coerce_storage(kwargs: dict[str, Any]) -> blosc2.Storage:
        storage = kwargs.pop("storage", None)
        if storage is None:
            storage_kwargs = {
                name: kwargs.pop(name) for name in list(blosc2.Storage.__annotations__) if name in kwargs
            }
            return blosc2.Storage(**storage_kwargs)
        if isinstance(storage, blosc2.Storage):
            return copy.deepcopy(storage)
        return blosc2.Storage(**storage)

    def _init_from_schunk(self, schunk) -> None:
        meta = schunk.meta
        if "listarray" not in meta:
            raise ValueError("The supplied SChunk is not tagged as a ListArray")
        la_meta = meta["listarray"]
        self.spec = ListSpec.from_metadata_dict(la_meta)
        self._pending_cells = []
        self._persisted_prefix_cache = None
        self._cached_batch_index = None
        self._cached_batch_values = None
        if self.spec.storage == "vl":
            if "vlarray" not in meta:
                raise ValueError("ListArray metadata says backend='vl' but VLArray tag is missing")
            self._backend = VLArray(_from_schunk=schunk)
            self._persisted_row_count = len(self._backend)
        else:
            if "batcharray" not in meta:
                raise ValueError("ListArray metadata says backend='batch' but BatchArray tag is missing")
            self._backend = BatchArray(_from_schunk=schunk)
            self._persisted_row_count = self._persisted_rows_count()

    def _invalidate_batch_caches(self) -> None:
        self._persisted_prefix_cache = None
        self._cached_batch_index = None
        self._cached_batch_values = None

    def _persisted_rows_count(self) -> int:
        if self.spec.storage == "vl":
            return len(self._backend)
        lengths = self._backend._load_or_compute_batch_lengths()
        return int(sum(lengths))

    def _persisted_prefix_sums(self) -> list[int]:
        if self._persisted_prefix_cache is not None:
            return self._persisted_prefix_cache
        lengths = self._backend._load_or_compute_batch_lengths()
        prefix = [0]
        total = 0
        for length in lengths:
            total += int(length)
            prefix.append(total)
        self._persisted_prefix_cache = prefix
        return prefix

    def _get_batch_values(self, batch_index: int) -> list[list[Any] | None]:
        if self._cached_batch_index == batch_index and self._cached_batch_values is not None:
            return self._cached_batch_values
        batch_values = self._backend[batch_index][:]
        self._cached_batch_index = batch_index
        self._cached_batch_values = batch_values
        return batch_values

    def _normalize_index(self, index: int) -> int:
        if not isinstance(index, int):
            raise TypeError("ListArray indices must be integers")
        n = len(self)
        if index < 0:
            index += n
        if index < 0 or index >= n:
            raise IndexError("ListArray index out of range")
        return index

    def _locate_persisted_row(self, row_index: int) -> tuple[int, int]:
        prefix = self._persisted_prefix_sums()
        batch_index = bisect_right(prefix, row_index) - 1
        inner_index = row_index - prefix[batch_index]
        return batch_index, inner_index

    def _flush_full_batches(self) -> None:
        if self.spec.storage != "batch":
            return
        batch_rows = self.batch_rows
        if batch_rows is None:
            return
        while len(self._pending_cells) >= batch_rows:
            batch = self._pending_cells[:batch_rows]
            self._backend.append(batch)
            self._pending_cells = self._pending_cells[batch_rows:]
            self._persisted_row_count += len(batch)
            self._invalidate_batch_caches()

    def append(self, value: Any) -> int:
        cell = coerce_list_cell(self.spec, value)
        if self.spec.storage == "vl":
            self._backend.append(cell)
            self._persisted_row_count = len(self._backend)
            return len(self)
        self._pending_cells.append(cell)
        self._flush_full_batches()
        return len(self)

    def extend(self, values: Iterable[Any]) -> None:
        if self.spec.storage == "vl":
            self._backend.extend(coerce_list_cell(self.spec, value) for value in values)
            self._persisted_row_count = len(self._backend)
            return
        for value in values:
            self._pending_cells.append(coerce_list_cell(self.spec, value))
            self._flush_full_batches()

    def flush(self) -> None:
        if self.spec.storage != "batch":
            return
        if self._pending_cells:
            batch = list(self._pending_cells)
            self._backend.append(batch)
            self._persisted_row_count += len(batch)
            self._pending_cells.clear()
            self._invalidate_batch_caches()

    def close(self) -> None:
        self.flush()

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self._normalize_index(index)
        if self.spec.storage == "vl":
            return self._backend[index]
        if index >= self._persisted_row_count:
            return self._pending_cells[index - self._persisted_row_count]
        batch_index, inner_index = self._locate_persisted_row(index)
        return self._get_batch_values(batch_index)[inner_index]

    def __setitem__(self, index: int, value: Any) -> None:
        cell = coerce_list_cell(self.spec, value)
        index = self._normalize_index(index)
        if self.spec.storage == "vl":
            self._backend[index] = cell
            return
        if index >= self._persisted_row_count:
            self._pending_cells[index - self._persisted_row_count] = cell
            return
        batch_index, inner_index = self._locate_persisted_row(index)
        batch = self._get_batch_values(batch_index).copy()
        batch[inner_index] = cell
        self._backend[batch_index] = batch
        self._cached_batch_index = batch_index
        self._cached_batch_values = batch

    def __len__(self) -> int:
        if self.spec.storage == "vl":
            return len(self._backend)
        return self._persisted_row_count + len(self._pending_cells)

    def __iter__(self) -> Iterator[Any]:
        for i in range(len(self)):
            yield self[i]

    def copy(self, **kwargs: Any) -> ListArray:
        out = ListArray(spec=self.spec, **kwargs)
        out.extend(self)
        if self.spec.storage == "batch":
            out.flush()
        return out

    @property
    def schunk(self):
        return self._backend.schunk

    @property
    def meta(self):
        return self._backend.meta

    @property
    def vlmeta(self):
        return self._backend.vlmeta

    @property
    def cparams(self):
        return self._backend.cparams

    @property
    def dparams(self):
        return self._backend.dparams

    @property
    def urlpath(self) -> str | None:
        return self._backend.urlpath

    @property
    def contiguous(self) -> bool:
        return self._backend.contiguous

    @property
    def batch_rows(self) -> int | None:
        if self.spec.batch_rows is not None:
            return self.spec.batch_rows
        return None

    @property
    def nbytes(self) -> int:
        return self._backend.nbytes

    @property
    def cbytes(self) -> int:
        return self._backend.cbytes

    @property
    def cratio(self) -> float:
        return self._backend.cratio

    @property
    def info(self) -> InfoReporter:
        return InfoReporter(self)

    @property
    def info_items(self) -> list:
        return [
            ("type", "ListArray"),
            ("logical_type", self.spec.display_label()),
            ("backend", self.spec.storage),
            ("serializer", self.spec.serializer),
            ("rows", len(self)),
            ("pending_rows", len(self._pending_cells) if self.spec.storage == "batch" else 0),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.2f}"),
        ]

    def to_cframe(self) -> bytes:
        self.flush()
        return self._backend.to_cframe()

    def _arrow_item_type(self):
        pa = _require_pyarrow()
        kind = self.spec.item_spec.to_metadata_dict()["kind"]
        mapping = {
            "int8": pa.int8(),
            "int16": pa.int16(),
            "int32": pa.int32(),
            "int64": pa.int64(),
            "uint8": pa.uint8(),
            "uint16": pa.uint16(),
            "uint32": pa.uint32(),
            "uint64": pa.uint64(),
            "float32": pa.float32(),
            "float64": pa.float64(),
            "bool": pa.bool_(),
            "string": pa.string(),
            "bytes": pa.large_binary(),
        }
        return mapping.get(kind)

    def to_arrow(self):
        pa = _require_pyarrow()
        self.flush()
        item_type = self._arrow_item_type()
        if item_type is not None:
            return pa.array(list(self), type=pa.list_(item_type))
        return pa.array(list(self))

    @classmethod
    def from_arrow(
        cls,
        arrow_array,
        *,
        item_spec: SchemaSpec | None = None,
        nullable: bool = True,
        storage: str = "batch",
        serializer: str = "msgpack",
        batch_rows: int | None = None,
        items_per_block: int | None = None,
        **kwargs: Any,
    ) -> ListArray:
        pa = _require_pyarrow()
        if isinstance(arrow_array, pa.ChunkedArray):
            arrow_array = arrow_array.combine_chunks()
        if item_spec is None:
            value_type = arrow_array.type.value_type
            import blosc2.schema as b2s

            mapping = {
                pa.int8(): b2s.int8(),
                pa.int16(): b2s.int16(),
                pa.int32(): b2s.int32(),
                pa.int64(): b2s.int64(),
                pa.uint8(): b2s.uint8(),
                pa.uint16(): b2s.uint16(),
                pa.uint32(): b2s.uint32(),
                pa.uint64(): b2s.uint64(),
                pa.float32(): b2s.float32(),
                pa.float64(): b2s.float64(),
                pa.bool_(): b2s.bool(),
                pa.string(): b2s.string(),
                pa.large_string(): b2s.string(),
                pa.binary(): b2s.bytes(),
                pa.large_binary(): b2s.bytes(),
            }
            item_spec = mapping.get(value_type)
            if item_spec is None:
                raise TypeError(f"Unsupported Arrow list item type {value_type!r}")
        arr = cls(
            item_spec=item_spec,
            nullable=nullable,
            storage=storage,
            serializer=serializer,
            batch_rows=batch_rows,
            items_per_block=items_per_block,
            **kwargs,
        )
        arr.extend(arrow_array.to_pylist())
        return arr

    def __enter__(self) -> ListArray:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return f"ListArray(type={self.spec.display_label()}, len={len(self)}, urlpath={self.urlpath!r})"
