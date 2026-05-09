#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""CTable: a columnar compressed table built on top of blosc2.NDArray."""

from __future__ import annotations

import ast
import contextlib
import contextvars
import copy
import dataclasses
import itertools
import os
import pprint
import re
import shutil
from collections import namedtuple
from collections.abc import Iterable, Mapping
from dataclasses import MISSING, dataclass
from dataclasses import field as dataclass_field
from textwrap import TextWrapper
from typing import Any, Generic, Literal, TypeVar

import numpy as np

import blosc2
from blosc2 import compute_chunks_blocks
from blosc2.ctable_storage import FileTableStorage, InMemoryTableStorage, TableStorage, TreeStoreTableStorage
from blosc2.info import InfoReporter, format_nbytes_info
from blosc2.list_array import ListArray, coerce_list_cell
from blosc2.scalar_array import _ScalarVarLenArray
from blosc2.schema import (
    ListSpec,
    ObjectSpec,
    SchemaSpec,
    StructSpec,
    VLBytesSpec,
    VLStringSpec,
    complex64,
    complex128,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    string,
    uint8,
    uint16,
    uint32,
    uint64,
)
from blosc2.schema import (
    bool as b2_bool,
)
from blosc2.schema import (
    bytes as b2_bytes,
)
from blosc2.schema_compiler import (
    ColumnConfig,
    CompiledColumn,
    CompiledSchema,
    _validate_column_name,
    compile_schema,
    compute_display_width,
    get_blosc2_field_metadata,
    schema_from_dict,
    schema_to_dict,
)


@dataclass(frozen=True)
class NullPolicy:
    """Default sentinels for inferred CTable scalar nulls.

    CTable nullable scalar columns are represented with per-column sentinel
    values. This policy is used when CTable has to infer those sentinels, such
    as when importing nullable scalar Arrow or Parquet columns without an
    explicit column-level null sentinel. The selected sentinel is stored in the
    resulting CTable schema, so existing tables remain self-describing.

    Examples
    --------
    Use :func:`blosc2.null_policy` to apply a policy while creating a CTable
    from data with nullable scalar columns::

        policy = blosc2.NullPolicy(
            signed_int_strategy="max",
            string_value="<NULL>",
            column_null_values={"user_id": -1, "country": "NA"},
        )

        with blosc2.null_policy(policy):
            table = blosc2.CTable.from_parquet("data.parquet")

    The same policy is used for explicit nullable schema specs::

        @dataclass
        class Row:
            user_id: int = blosc2.field(blosc2.int64(nullable=True))
            country: str = blosc2.field(blosc2.string(nullable=True))

        with blosc2.null_policy(policy):
            table = blosc2.CTable(Row)

    ``column_null_values`` takes precedence over the type-wide defaults in the
    policy.  This is useful when a particular column needs a sentinel that is
    known not to collide with its real values.
    """

    string_value: str = "__BLOSC2_NULL__"
    bytes_value: bytes = b"__BLOSC2_NULL__"
    float_value: float = float("nan")
    bool_value: int = 255
    signed_int_strategy: Literal["min", "max"] = "min"
    unsigned_int_strategy: Literal["min", "max"] = "max"
    column_null_values: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def sentinel_for_arrow_type(self, pa, pa_type):
        """Return the default sentinel for *pa_type*, or ``None`` if unsupported."""
        signed_ints = [
            (pa.int8(), np.int8),
            (pa.int16(), np.int16),
            (pa.int32(), np.int32),
            (pa.int64(), np.int64),
        ]
        unsigned_ints = [
            (pa.uint8(), np.uint8),
            (pa.uint16(), np.uint16),
            (pa.uint32(), np.uint32),
            (pa.uint64(), np.uint64),
        ]
        for arrow_type, dtype in signed_ints:
            if pa_type == arrow_type:
                info = np.iinfo(dtype)
                return info.min if self.signed_int_strategy == "min" else info.max
        for arrow_type, dtype in unsigned_ints:
            if pa_type == arrow_type:
                info = np.iinfo(dtype)
                return info.min if self.unsigned_int_strategy == "min" else info.max
        if pa_type in (pa.float32(), pa.float64()):
            return self.float_value
        if pa_type == pa.bool_():
            return self.bool_value
        if pa_type in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
            return self.string_value
        if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
            return self.bytes_value
        return None


DEFAULT_NULL_POLICY = NullPolicy()
_NULL_POLICY = contextvars.ContextVar("blosc2_null_policy", default=DEFAULT_NULL_POLICY)


def get_null_policy() -> NullPolicy:
    """Return the current default null policy."""
    return _NULL_POLICY.get()


@contextlib.contextmanager
def null_policy(policy: NullPolicy):
    """Temporarily set the default policy for CTable null sentinel inference."""
    token = _NULL_POLICY.set(policy)
    try:
        yield
    finally:
        _NULL_POLICY.reset(token)


# ---------------------------------------------------------------------------
# Index proxy
# ---------------------------------------------------------------------------


_DTYPE_SPEC_FACTORIES = {
    np.dtype(np.int8): int8,
    np.dtype(np.int16): int16,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.uint8): uint8,
    np.dtype(np.uint16): uint16,
    np.dtype(np.uint32): uint32,
    np.dtype(np.uint64): uint64,
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
    np.dtype(np.complex64): complex64,
    np.dtype(np.complex128): complex128,
    np.dtype(np.bool_): b2_bool,
}


class _FakeVlMeta:
    """Minimal vlmeta stand-in that accepts writes without touching a real SChunk."""

    def __init__(self):
        self._data: dict = {}

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def get(self, key, default=None):
        return self._data.get(key, default)


class _FakeSchunk:
    """Minimal SChunk stand-in whose vlmeta stores in memory."""

    def __init__(self):
        self.vlmeta = _FakeVlMeta()


class _CTableBuildProxy:
    """Minimal shim that lets the ``indexing`` module build sidecars for a
    CTable column without touching the column's own ``schunk.vlmeta``.

    Attributes mirror those required by the internal build functions:
    ``urlpath``, ``schunk``, ``shape``, ``ndim``, ``dtype``, ``chunks``,
    ``blocks``, and item access via ``__getitem__``.
    """

    def __init__(self, col_array: blosc2.NDArray, anchor_urlpath: str | None) -> None:
        self._col_array = col_array
        self.urlpath = anchor_urlpath  # controls sidecar placement
        self.schunk = _FakeSchunk()
        self.shape = col_array.shape
        self.ndim = col_array.ndim
        self.dtype = col_array.dtype
        self.chunks = col_array.chunks
        self.blocks = col_array.blocks

    def __getitem__(self, key):
        return self._col_array[key]


class _CTableInfoReporter(InfoReporter):
    """Info reporter that also preserves the historic ``t.info()`` call style."""

    def __repr__(self) -> str:
        items = self.obj.info_items
        max_key_len = max(len(k) for k, _ in items)
        parts = []
        for key, value in items:
            if isinstance(value, dict):
                parts.append(f"{key.ljust(max_key_len)} :")
                pretty = pprint.pformat(value, sort_dicts=False)
                parts.extend(f" {line}" for line in pretty.splitlines())
                continue

            wrapper = TextWrapper(
                width=96,
                initial_indent=key.ljust(max_key_len) + " : ",
                subsequent_indent=" " * max_key_len + " : ",
            )
            parts.append(wrapper.fill(str(value)))
        return "\n".join(parts) + "\n"

    def __call__(self) -> None:
        print(repr(self), end="")


class _InfoLiteral:
    """Pretty-printer helper for unquoted literal values inside info dicts."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


# RowT is intentionally left unbound so CTable works with both dataclasses
# and legacy Pydantic models during the transition period.
RowT = TypeVar("RowT")

# Arrays larger than this threshold use blosc2.arange instead of np.arange to
# avoid large transient allocations when mapping logical to physical row positions.
_BLOSC2_ARANGE_THRESHOLD = 1_000_000


def _arange(start, stop=None, step=1) -> blosc2.NDArray | np.ndarray:
    """Return a range array, using blosc2 for large n to save memory."""
    if stop is None:
        start, stop = 0, start
    n = len(range(start, stop, step))
    return (
        blosc2.arange(start, stop, step) if n >= _BLOSC2_ARANGE_THRESHOLD else np.arange(start, stop, step)
    )


# ---------------------------------------------------------------------------
# Legacy Pydantic-compat helpers
# Keep these so existing code that uses Annotated[type, NumpyDtype(...)] or
# Annotated[str, MaxLen(...)] on a pydantic.BaseModel continues to work.
# ---------------------------------------------------------------------------


class NumpyDtype:
    """Metadata tag for Pydantic-based schemas (legacy)."""

    def __init__(self, dtype):
        self.dtype = dtype


class MaxLen:
    """Metadata tag for fixed-width string/bytes columns in Pydantic-based schemas (legacy)."""

    def __init__(self, length: int):
        self.length = int(length)


def _default_display_width(origin) -> int:
    """Return a sensible display column width for a given Python type (legacy)."""
    return {int: 12, float: 15, bool: 6, complex: 25}.get(origin, 20)


def _resolve_field_dtype(field) -> tuple[np.dtype, int]:
    """Return (numpy dtype, display_width) for a Pydantic model field (legacy).

    Extracts dtype from NumpyDtype metadata when present (same class), otherwise
    falls back to a sensible default for each Python primitive type.
    """
    annotation = field.annotation
    origin = getattr(annotation, "__origin__", annotation)

    # str / bytes → look for MaxLen metadata, build fixed-width dtype
    if origin in (str, bytes) or annotation in (str, bytes):
        is_bytes = origin is bytes or annotation is bytes
        max_len = 32
        if hasattr(annotation, "__metadata__"):
            for meta in annotation.__metadata__:
                if isinstance(meta, MaxLen):
                    max_len = meta.length
                    break
        kind = "S" if is_bytes else "U"
        dt = np.dtype(f"{kind}{max_len}")
        display_width = max(10, min(max_len, 50))
        return dt, display_width

    # Check for explicit NumpyDtype metadata (same class as defined here)
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, NumpyDtype):
                dt = np.dtype(meta.dtype)
                display_width = _default_display_width(origin)
                return dt, display_width

    # Primitive defaults
    _PRIMITIVE_MAP = {
        int: (np.int64, 12),
        float: (np.float64, 15),
        bool: (np.bool_, 6),
        complex: (np.complex128, 25),
    }
    if origin in _PRIMITIVE_MAP:
        dt_raw, display_width = _PRIMITIVE_MAP[origin]
        return np.dtype(dt_raw), display_width

    return np.dtype(np.object_), 20


class _LegacySpec(SchemaSpec):
    """Internal compatibility spec wrapping a dtype extracted from a Pydantic schema."""

    def __init__(self, dtype: np.dtype):
        self.dtype = np.dtype(dtype)
        self.python_type = object

    def to_pydantic_kwargs(self) -> dict[str, Any]:
        return {}

    def to_metadata_dict(self) -> dict[str, Any]:
        return {"kind": "legacy", "dtype": str(self.dtype)}


def _compile_pydantic_schema(row_cls: type) -> CompiledSchema:
    """Compatibility adapter: build a CompiledSchema from a Pydantic BaseModel subclass."""
    columns: list[CompiledColumn] = []
    for name, pyd_field in row_cls.model_fields.items():
        dtype, display_width = _resolve_field_dtype(pyd_field)
        spec = _LegacySpec(dtype)
        col = CompiledColumn(
            name=name,
            py_type=object,
            spec=spec,
            dtype=dtype,
            default=MISSING,
            config=ColumnConfig(cparams=None, dparams=None, chunks=None, blocks=None),
            display_width=display_width,
        )
        columns.append(col)
    return CompiledSchema(
        row_cls=row_cls,
        columns=columns,
        columns_by_name={col.name: col for col in columns},
    )


# ---------------------------------------------------------------------------
# ColumnViewIndexer
# ---------------------------------------------------------------------------


class ColumnViewIndexer:
    """Returned by :attr:`Column.view`; indexing returns a Column sub-view.

    Use ``t.price.view[2:10]`` to obtain a writable logical sub-view for
    chained operations (``sum()``, ``[:] = values``, …).
    Use ``t.price[2:10]`` to materialise values as a NumPy array.
    """

    def __init__(self, column: Column) -> None:
        self._column = column

    def __getitem__(self, key) -> Column:
        return self._column._view_from_key(key)

    def __repr__(self) -> str:
        return f"<ColumnViewIndexer col={self._column._col_name!r}>"


# ---------------------------------------------------------------------------
# Internal row/indexing helpers (unchanged)
# ---------------------------------------------------------------------------


def _find_physical_index(arr: blosc2.NDArray, logical_key: int) -> int:
    """Translate a logical (valid-row) index into a physical array index.

    Iterates chunk metadata of the boolean *arr* (valid_rows) to locate the
    *logical_key*-th True value without fully decompressing the array.

    Returns
    -------
    int
        Physical position in the underlying storage array.

    Raises
    ------
    IndexError
        If the logical index is out of range or the array is inconsistent.
    """
    count = 0
    chunk_size = arr.chunks[0]

    for info in arr.iterchunks_info():
        actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
        chunk_start = info.nchunk * chunk_size

        if info.special == blosc2.SpecialValue.ZERO:
            continue

        if info.special == blosc2.SpecialValue.VALUE:
            val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
            if not val:
                continue
            if count + actual_size <= logical_key:
                count += actual_size
                continue
            return chunk_start + (logical_key - count)

        chunk_data = arr[chunk_start : chunk_start + actual_size]
        n_true = int(np.count_nonzero(chunk_data))
        if count + n_true <= logical_key:
            count += n_true
            continue

        return chunk_start + int(np.flatnonzero(chunk_data)[logical_key - count])

    raise IndexError("Unexpected error finding physical index.")


def _make_namedtuple_row_type(col_names: tuple[str, ...]):
    base = namedtuple("CTableRow", col_names, rename=True)
    field_name_map = dict(zip(col_names, base._fields, strict=True))

    class CTableRow(base):
        __slots__ = ()
        _field_name_map = field_name_map
        _original_fields = col_names

        def __getitem__(self, key):
            if isinstance(key, str):
                try:
                    return getattr(self, self._field_name_map[key])
                except KeyError as exc:
                    raise KeyError(
                        f"No field named {key!r}. Available: {list(self._original_fields)}"
                    ) from exc
            return tuple.__getitem__(self, key)

        def as_dict(self) -> dict[str, Any]:
            return {name: self[name] for name in self._original_fields}

    return CTableRow


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


class Column:
    """Column view for a :class:`CTable`, with vectorized operations and reductions."""

    _REPR_PREVIEW_ITEMS = 8

    def __init__(self, table: CTable, col_name: str, mask=None):
        self._table = table
        self._col_name = col_name
        self._mask = mask

    @property
    def _raw_col(self):
        cc = self._table._computed_cols.get(self._col_name)
        if cc is not None:
            return cc["lazy"]
        return self._table._cols[self._col_name]

    @property
    def is_computed(self) -> bool:
        """True if this column is a virtual computed column (read-only)."""
        return self._col_name in self._table._computed_cols

    @property
    def is_list(self) -> bool:
        col = self._table._schema.columns_by_name.get(self._col_name)
        return col is not None and isinstance(col.spec, ListSpec)

    @property
    def is_varlen_scalar(self) -> bool:
        """True if this column holds variable-length scalar strings or bytes."""
        col = self._table._schema.columns_by_name.get(self._col_name)
        return col is not None and isinstance(col.spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec))

    @property
    def _valid_rows(self):
        if self._mask is None:
            return self._table._valid_rows

        return (self._table._valid_rows & self._mask).compute()

    def _lazy_valid_rows(self):
        """Return this column's visible-row mask without forcing lazy evaluation."""
        if self._mask is None:
            return self._table._valid_rows
        return self._table._valid_rows & self._mask

    def __getitem__(self, key: int | slice | list | np.ndarray):
        """Return values for the given logical index.

        - ``int``              → scalar
        - ``slice``            → :class:`numpy.ndarray`
        - ``list / np.ndarray`` → :class:`numpy.ndarray`
        - ``bool np.ndarray``  → :class:`numpy.ndarray`

        For a writable logical sub-view use :attr:`view`.
        """
        return self._values_from_key(key)

    def _values_from_key(self, key):  # noqa: C901
        """Materialise values for a logical index key."""
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            return self._raw_col[int(pos_true)]

        elif isinstance(key, slice):
            valid = self._valid_rows
            real_pos = blosc2.where(valid, _arange(len(valid))).compute()
            start, stop, step = key.indices(len(real_pos))
            if start >= stop:
                return [] if (self.is_list or self.is_varlen_scalar) else np.array([], dtype=self.dtype)
            selected_pos = real_pos[start:stop:step]  # physical row positions
            if self.is_computed:
                lo, hi = int(selected_pos.min()), int(selected_pos.max())
                chunk = np.asarray(self._raw_col[lo : hi + 1])
                return chunk[selected_pos - lo]
            if self.is_list or self.is_varlen_scalar:
                return self._raw_col[selected_pos]
            return np.asarray(self._raw_col[selected_pos])

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            n_live = len(self)
            if len(key) != n_live:
                raise IndexError(
                    f"Boolean mask length {len(key)} does not match number of live rows {n_live}."
                )
            all_pos = np.where(self._valid_rows[:])[0]
            phys_indices = all_pos[key]
            if self.is_computed:
                raw_np = np.asarray(self._raw_col[:])
                return raw_np[phys_indices]
            if self.is_list or self.is_varlen_scalar:
                return self._raw_col[phys_indices]
            return self._raw_col[phys_indices]

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, _arange(len(self._valid_rows))).compute()
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            if self.is_computed:
                raw_np = np.asarray(self._raw_col[:])
                return raw_np[phys_indices]
            if self.is_list or self.is_varlen_scalar:
                return self._raw_col[phys_indices]
            return self._raw_col[phys_indices]

        raise TypeError(f"Invalid index type: {type(key)}")

    def _view_from_key(self, key) -> Column:
        """Build a Column sub-view for the given logical index key.

        Called by :class:`ColumnViewIndexer`.  Supports slice, boolean mask,
        and integer list / array keys.  The returned :class:`Column` shares
        the underlying physical storage and writes through to the table.
        """
        if isinstance(key, slice):
            valid = self._valid_rows
            real_pos = blosc2.where(valid, _arange(len(valid))).compute()
            start, stop, step = key.indices(len(real_pos))
            mask = blosc2.zeros(len(self._table._valid_rows), dtype=np.bool_)
            if start < stop:
                if step == 1:
                    phys_start = real_pos[start]
                    phys_stop = real_pos[stop - 1]
                    mask[phys_start : phys_stop + 1] = True
                else:
                    lindices = _arange(start, stop, step)
                    phys_indices = real_pos[lindices]
                    mask[phys_indices[:]] = True
            return Column(self._table, self._col_name, mask=mask)

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            n_live = len(self)
            if len(key) != n_live:
                raise IndexError(
                    f"Boolean mask length {len(key)} does not match number of live rows {n_live}."
                )
            all_pos = np.where(self._valid_rows[:])[0]
            phys_indices = all_pos[key]
            mask_np = np.zeros(len(self._table._valid_rows), dtype=np.bool_)
            mask_np[phys_indices] = True
            return Column(self._table, self._col_name, mask=blosc2.asarray(mask_np))

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, _arange(len(self._valid_rows))).compute()
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            mask_np = np.zeros(len(self._table._valid_rows), dtype=np.bool_)
            mask_np[phys_indices] = True
            return Column(self._table, self._col_name, mask=blosc2.asarray(mask_np))

        raise TypeError(
            f"Column.view[] does not support key type {type(key).__name__!r}. "
            "Supported: slice, boolean array, list / integer array."
        )

    @property
    def view(self) -> ColumnViewIndexer:
        """Return a :class:`ColumnViewIndexer` for creating logical sub-views.

        Examples
        --------
        Read a sub-view for chained aggregates::

            sub = t.price.view[2:10]
            sub.sum()

        Bulk write through a sub-view::

            t.price.view[0:5][:] = np.zeros(5)
        """
        return ColumnViewIndexer(self)

    def __setitem__(self, key: int | slice | list | np.ndarray, value):  # noqa: C901
        """Set one or more live column values; accepts the same index forms as :meth:`__getitem__`."""
        if self._table._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.is_computed:
            raise ValueError(f"Column {self._col_name!r} is a computed column and cannot be written to.")
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            self._raw_col[int(pos_true)] = value

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            n_live = len(self)
            if len(key) != n_live:
                raise IndexError(
                    f"Boolean mask length {len(key)} does not match number of live rows {n_live}."
                )
            all_pos = np.where(self._valid_rows[:])[0]
            phys_indices = all_pos[key]
            if self.is_list or self.is_varlen_scalar:
                if len(value) != len(phys_indices):
                    raise ValueError("Length mismatch in list-column assignment")
                for pos, cell in zip(phys_indices, value, strict=True):
                    self._raw_col[int(pos)] = cell
            else:
                if isinstance(value, (list, tuple)):
                    value = np.array(value, dtype=self._raw_col.dtype)
                self._raw_col[phys_indices] = value

        elif isinstance(key, (slice, list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, _arange(len(self._valid_rows))).compute()
            if isinstance(key, slice):
                lindices = range(*key.indices(len(real_pos)))
                phys_indices = np.array([real_pos[i] for i in lindices], dtype=np.int64)
            else:
                phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)

            if self.is_list or self.is_varlen_scalar:
                if len(value) != len(phys_indices):
                    raise ValueError("Length mismatch in list-column assignment")
                for pos, cell in zip(phys_indices, value, strict=True):
                    self._raw_col[int(pos)] = cell
            else:
                if isinstance(value, (list, tuple)):
                    value = np.array(value, dtype=self._raw_col.dtype)
                self._raw_col[phys_indices] = value

        else:
            raise TypeError(f"Invalid index type: {type(key)}")
        self._table._root_table._mark_all_indexes_stale()

    def __iter__(self):
        """Iterate over live column values in insertion order, skipping deleted rows."""
        if self.is_computed:
            yield from self._iter_chunks_computed(size=None)
            return
        if self.is_list or self.is_varlen_scalar:
            yield from self._raw_col[np.where(self._valid_rows[:])[0]]
            return
        arr = self._valid_rows
        chunk_size = arr.chunks[0]

        for info in arr.iterchunks_info():
            actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
            chunk_start = info.nchunk * chunk_size

            if info.special == blosc2.SpecialValue.ZERO:
                continue

            if info.special == blosc2.SpecialValue.VALUE:
                val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                if not val:
                    continue
                yield from self._raw_col[chunk_start : chunk_start + actual_size]
                continue

            mask_chunk = arr[chunk_start : chunk_start + actual_size]
            data_chunk = self._raw_col[chunk_start : chunk_start + actual_size]
            yield from data_chunk[mask_chunk]

    def __repr__(self) -> str:
        preview_items = []
        for value in itertools.islice(self, self._REPR_PREVIEW_ITEMS + 1):
            if isinstance(value, np.generic):
                value = value.item()
            preview_items.append(repr(value))

        truncated = len(preview_items) > self._REPR_PREVIEW_ITEMS
        if truncated:
            preview_items = preview_items[: self._REPR_PREVIEW_ITEMS]
            preview_items.append("...")

        preview = ", ".join(preview_items)
        return f"Column({self._col_name!r}, dtype={self.dtype}, len={len(self)}, values=[{preview}])"

    def __len__(self):
        """Return the number of live (non-deleted) values in this column."""
        return blosc2.count_nonzero(self._valid_rows)

    @property
    def shape(self) -> tuple[int]:
        """Logical shape of the live column values."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Number of logical dimensions."""
        return 1

    @property
    def size(self) -> int:
        """Number of live values in the column."""
        return len(self)

    def _ensure_queryable(self) -> None:
        if self.is_varlen_scalar:
            raise NotImplementedError(
                f"Column {self._col_name!r} is a vlstring/vlbytes column; "
                "lazy expressions and vectorized comparisons are not supported yet."
            )

    @staticmethod
    def _unwrap_operand(other):
        if isinstance(other, Column):
            other._ensure_queryable()
            return other._raw_col
        return other

    @property
    def _is_nullable_bool(self) -> bool:
        col = self._table._schema.columns_by_name.get(self._col_name)
        return (
            col is not None
            and col.spec.to_metadata_dict().get("kind") == "bool"
            and getattr(col.spec, "null_value", None) is not None
        )

    def __neg__(self):
        self._ensure_queryable()
        return -self._raw_col

    def __pos__(self):
        self._ensure_queryable()
        return +self._raw_col

    def __abs__(self):
        self._ensure_queryable()
        return abs(self._raw_col)

    def __add__(self, other):
        self._ensure_queryable()
        return self._raw_col + self._unwrap_operand(other)

    def __radd__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) + self._raw_col

    def __sub__(self, other):
        self._ensure_queryable()
        return self._raw_col - self._unwrap_operand(other)

    def __rsub__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) - self._raw_col

    def __mul__(self, other):
        self._ensure_queryable()
        return self._raw_col * self._unwrap_operand(other)

    def __rmul__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) * self._raw_col

    def __truediv__(self, other):
        self._ensure_queryable()
        return self._raw_col / self._unwrap_operand(other)

    def __rtruediv__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) / self._raw_col

    def __floordiv__(self, other):
        self._ensure_queryable()
        return self._raw_col // self._unwrap_operand(other)

    def __rfloordiv__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) // self._raw_col

    def __mod__(self, other):
        self._ensure_queryable()
        return self._raw_col % self._unwrap_operand(other)

    def __rmod__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) % self._raw_col

    def __pow__(self, other):
        self._ensure_queryable()
        return self._raw_col ** self._unwrap_operand(other)

    def __rpow__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) ** self._raw_col

    def __and__(self, other):
        self._ensure_queryable()
        return self._raw_col & self._unwrap_operand(other)

    def __rand__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) & self._raw_col

    def __or__(self, other):
        self._ensure_queryable()
        return self._raw_col | self._unwrap_operand(other)

    def __ror__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) | self._raw_col

    def __xor__(self, other):
        self._ensure_queryable()
        return self._raw_col ^ self._unwrap_operand(other)

    def __rxor__(self, other):
        self._ensure_queryable()
        return self._unwrap_operand(other) ^ self._raw_col

    def __invert__(self):
        self._ensure_queryable()
        if self._is_nullable_bool:
            return self._raw_col == 0
        return ~self._raw_col

    def __lt__(self, other):
        self._ensure_queryable()
        return self._raw_col < self._unwrap_operand(other)

    def __le__(self, other):
        self._ensure_queryable()
        return self._raw_col <= self._unwrap_operand(other)

    def __eq__(self, other):
        self._ensure_queryable()
        if self._is_nullable_bool and isinstance(other, (bool, np.bool_)):
            return self._raw_col == int(other)
        return self._raw_col == self._unwrap_operand(other)

    def __ne__(self, other):
        self._ensure_queryable()
        if self._is_nullable_bool and isinstance(other, (bool, np.bool_)):
            return self._raw_col == int(not other)
        return self._raw_col != self._unwrap_operand(other)

    def __gt__(self, other):
        self._ensure_queryable()
        return self._raw_col > self._unwrap_operand(other)

    def __ge__(self, other):
        self._ensure_queryable()
        return self._raw_col >= self._unwrap_operand(other)

    @property
    def dtype(self):
        """NumPy dtype of the underlying storage, or ``None`` for
        variable-length columns (:func:`~blosc2.vlstring`,
        :func:`~blosc2.vlbytes`, :func:`~blosc2.list`)."""
        return getattr(self._raw_col, "dtype", None)

    def iter_chunks(self, size: int = 65536):
        """Iterate over live column values in chunks of *size* rows.

        Yields numpy arrays of at most *size* elements each, skipping deleted
        rows.  The last chunk may be smaller than *size*.

        Parameters
        ----------
        size:
            Number of live rows per yielded chunk.  Defaults to 65 536.

        Yields
        ------
        numpy.ndarray
            A 1-D array of up to *size* live values with this column's dtype.

        Examples
        --------
        >>> for chunk in t["score"].iter_chunks(size=100_000):
        ...     process(chunk)
        """
        if self.is_computed:
            yield from self._iter_chunks_computed(size=size)
            return
        if self.is_list:
            raise TypeError("Column.iter_chunks() is not supported for list columns in V1.")
        if self.is_varlen_scalar:
            raise TypeError("Column.iter_chunks() is not supported for varlen scalar columns.")
        valid = self._valid_rows
        raw = self._raw_col
        arr_len = len(valid)
        phys_chunk = valid.chunks[0]

        pending: list[np.ndarray] = []
        pending_count = 0

        for info in valid.iterchunks_info():
            actual = min(phys_chunk, arr_len - info.nchunk * phys_chunk)
            start = info.nchunk * phys_chunk

            if info.special == blosc2.SpecialValue.ZERO:
                continue

            if info.special == blosc2.SpecialValue.VALUE:
                val = np.frombuffer(info.repeated_value, dtype=valid.dtype)[0]
                if not val:
                    continue
                segment = raw[start : start + actual]
            else:
                mask = valid[start : start + actual]
                segment = raw[start : start + actual][mask]

            if len(segment) == 0:
                continue

            pending.append(segment)
            pending_count += len(segment)

            while pending_count >= size:
                combined = np.concatenate(pending)
                yield combined[:size]
                rest = combined[size:]
                pending = [rest] if len(rest) > 0 else []
                pending_count = len(rest)

        if pending:
            yield np.concatenate(pending)

    def _iter_chunks_computed(self, size):
        """Yield live values from a computed column, chunk-by-chunk.

        Evaluates the LazyExpr slice-by-slice using the physical chunk layout
        of *valid_rows* and applies the valid-rows mask before accumulating.
        When *size* is None (used by ``__iter__``), each physical chunk is
        yielded directly.
        """
        lazy = self._raw_col  # a LazyExpr
        valid = self._valid_rows
        phys_len = len(valid)
        chunk_size = valid.chunks[0]

        pending: list[np.ndarray] = []
        pending_n = 0

        for chunk_start in range(0, phys_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, phys_len)
            mask = valid[chunk_start:chunk_end]  # numpy bool array
            n_live = int(np.count_nonzero(mask))
            if n_live == 0:
                continue

            # Evaluate the expression only for this physical slice
            data_chunk = np.asarray(lazy[chunk_start:chunk_end])
            segment = data_chunk[mask] if n_live < (chunk_end - chunk_start) else data_chunk

            if size is None:
                # __iter__ path: yield each chunk directly
                yield from segment
                continue

            pending.append(segment)
            pending_n += len(segment)

            while pending_n >= size:
                combined = np.concatenate(pending)
                yield combined[:size]
                rest = combined[size:]
                pending = [rest] if len(rest) > 0 else []
                pending_n = len(rest)

        if size is not None and pending:
            yield np.concatenate(pending)

    def assign(self, data) -> None:
        """Replace all live values in this column with *data*.

        Works on both full tables and views — on a view, only the rows
        visible through the view's mask are overwritten.

        Parameters
        ----------
        data:
            List, numpy array, or any iterable.  Must have exactly as many
            elements as there are live rows in this column.  Values are
            coerced to the column's dtype if possible.

        Raises
        ------
        ValueError
            If ``len(data)`` does not match the number of live rows, or the
            table is opened read-only.
        TypeError
            If values cannot be coerced to the column's dtype.
        """
        if self._table._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.is_computed:
            raise ValueError(f"Column {self._col_name!r} is a computed column and cannot be written to.")
        if self.is_list:
            values = list(data)
            if len(values) != len(self):
                raise ValueError(f"assign() requires {len(self)} values (live rows), got {len(values)}.")
            live_pos = np.where(self._valid_rows[:])[0]
            for pos, cell in zip(live_pos, values, strict=True):
                self._raw_col[int(pos)] = cell
            self._table._root_table._mark_all_indexes_stale()
            return
        n_live = len(self)
        arr = np.asarray(data)
        if len(arr) != n_live:
            raise ValueError(f"assign() requires {n_live} values (live rows), got {len(arr)}.")
        try:
            arr = arr.astype(self.dtype)
        except (ValueError, OverflowError) as exc:
            raise TypeError(f"Cannot coerce data to column dtype {self.dtype!r}: {exc}") from exc
        live_pos = np.where(self._valid_rows[:])[0]
        self._raw_col[live_pos] = arr
        self._table._root_table._mark_all_indexes_stale()

    # ------------------------------------------------------------------
    # Null sentinel support
    # ------------------------------------------------------------------

    @property
    def null_value(self):
        """The sentinel value that represents NULL for this column, or ``None``."""
        col_info = self._table._schema.columns_by_name.get(self._col_name)
        if col_info is None:
            return None
        return getattr(col_info.spec, "null_value", None)

    def _null_mask_for(self, arr: np.ndarray) -> np.ndarray:
        """Return a bool array True where *arr* contains the null sentinel.

        Always returns an array of the same length as *arr*; all False when
        no null_value is configured.
        """
        nv = self.null_value
        if nv is None:
            return np.zeros(len(arr), dtype=np.bool_)
        if isinstance(nv, float) and np.isnan(nv):
            return np.isnan(arr)
        return arr == nv

    def is_null(self) -> np.ndarray:
        """Return a boolean array True where the live value is the null sentinel.

        For varlen scalar columns (vlstring/vlbytes) nullability is represented
        as native ``None`` values, so this returns True wherever the value is
        ``None``.
        """
        if self.is_varlen_scalar:
            return np.array([v is None for v in self], dtype=np.bool_)
        return self._null_mask_for(self[:])

    def notnull(self) -> np.ndarray:
        """Return a boolean array True where the live value is *not* the null sentinel."""
        return ~self.is_null()

    def null_count(self) -> int:
        """Return the number of live rows whose value equals the null sentinel.

        Returns ``0`` in O(1) if no ``null_value`` is configured for this column
        and the column is not a varlen scalar column.
        """
        if self.is_varlen_scalar:
            return sum(1 for v in self if v is None)
        if self.null_value is None:
            return 0
        return int(self.is_null().sum())

    def _nonnull_chunks(self):
        """Yield chunks of live, non-null values.

        Each yielded array has the null sentinel values removed.  If no
        null_value is configured this behaves identically to
        :meth:`iter_chunks`.
        """
        nv = self.null_value
        if nv is None:
            yield from self.iter_chunks()
            return
        is_nan_nv = isinstance(nv, float) and np.isnan(nv)
        for chunk in self.iter_chunks():
            if is_nan_nv:
                mask = ~np.isnan(chunk)
            else:
                mask = chunk != nv
            filtered = chunk[mask]
            if len(filtered) > 0:
                yield filtered

    def unique(self) -> np.ndarray:
        """Return sorted array of unique live, non-null values.

        Null sentinel values are excluded.
        Processes data in chunks — never loads the full column at once.
        """
        seen: set = set()
        for chunk in self._nonnull_chunks():
            seen.update(chunk.tolist())
        return np.array(sorted(seen), dtype=self.dtype)

    def value_counts(self) -> dict:
        """Return a ``{value: count}`` dict sorted by count descending.

        Null sentinel values are excluded.
        Processes data in chunks — never loads the full column at once.

        Example
        -------
        >>> t["active"].value_counts()
        {True: 8432, False: 1568}
        """
        counts: dict = {}
        for chunk in self._nonnull_chunks():
            for val in chunk.tolist():
                counts[val] = counts.get(val, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    def _require_nonempty(self, op: str) -> None:
        if len(self) == 0:
            raise ValueError(f"Column.{op}() called on an empty column.")

    def _require_kind(self, kinds: str, op: str) -> None:
        """Raise TypeError if this column's dtype is not in *kinds*."""
        if self.dtype.kind not in kinds:
            _kind_names = {
                "b": "bool",
                "i": "signed int",
                "u": "unsigned int",
                "f": "float",
                "c": "complex",
                "U": "string",
                "S": "bytes",
            }
            raise TypeError(
                f"Column.{op}() is not supported for dtype {self.dtype!r} "
                f"({_kind_names.get(self.dtype.kind, self.dtype.kind)})."
            )

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def _normalize_sum_where(self, where):
        """Normalize an optional ``sum(where=...)`` predicate to a boolean array/expression."""
        if where is None:
            return None
        if isinstance(where, str):
            self._table._guard_varlen_scalar_expression(where)
            where = blosc2.lazyexpr(where, self._table._where_expression_operands())
        if isinstance(where, np.ndarray) and where.dtype == np.bool_:
            where = blosc2.asarray(where)
        if isinstance(where, Column):
            where = where._raw_col == 1 if where._is_nullable_bool else where._raw_col
        if not (
            isinstance(where, (blosc2.NDArray, blosc2.LazyExpr))
            and getattr(where, "dtype", None) == np.bool_
        ):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(where).__name__}")
        return where

    def _lazy_nonnull_mask(self, where=None):
        """Build a lazy visible-row mask, optionally intersected with non-null values."""
        raw = self._raw_col
        if not isinstance(raw, (blosc2.NDArray, blosc2.LazyExpr)):
            return NotImplemented
        mask = self._lazy_valid_rows()
        if where is not None:
            mask = mask & where
        nv = self.null_value
        if nv is not None:
            if isinstance(nv, (float, np.floating)) and np.isnan(nv):
                nonnull = ~blosc2.isnan(raw)
            else:
                nonnull = raw != nv
            mask = mask & nonnull
        return mask

    def _sum_lazy_fastpath(self, acc_dtype, where=None):
        """Try to compute ``sum`` as a pushed-down lazy masked reduction."""
        if self.is_list or self.is_varlen_scalar or self.dtype is None or self.dtype.kind not in "biufc":
            return NotImplemented

        raw = self._raw_col
        if not isinstance(raw, (blosc2.NDArray, blosc2.LazyExpr)):
            return NotImplemented

        # A lazy masked reduction scans the full physical column.  For very
        # selective filtered views, the existing iterator can skip all-zero mask
        # chunks and is usually faster.  Explicit sum(where=...) is already a
        # direct pushed-down aggregate, so do not apply the density guard there.
        total_rows = len(self._table._valid_rows)
        if (
            where is None
            and self._table.base is not None
            and total_rows
            and self._table._n_rows / total_rows < 0.25
        ):
            return NotImplemented

        mask = self._lazy_nonnull_mask(where=where)
        if mask is NotImplemented:
            return NotImplemented

        zero = acc_dtype(0)
        try:
            return blosc2.where(mask, raw, zero).sum(dtype=acc_dtype)
        except Exception:
            return NotImplemented

    def sum(self, dtype=None, *, where=None):
        """Sum of all live, non-null values.

        Returns zero for an empty column or filtered view.

        Supported dtypes: bool, int, uint, float, complex.
        Bool values are counted as 0 / 1.
        Null sentinel values are skipped.

        Parameters
        ----------
        dtype:
            Optional accumulator dtype.  When omitted, float columns use
            ``np.float64``, complex columns use ``np.complex128``, and integer
            / bool columns use ``np.int64``.
        where:
            Optional boolean predicate. Only rows where the predicate is true,
            the table row is live, and this column is non-null are included.
            This enables direct filtered aggregate pushdown, avoiding creation
            of an intermediate filtered table view.

        Examples
        --------
        Sum values matching a predicate without materializing a filtered view::

            total = t["amount"].sum(where=t.category == 3)

        Combine several column predicates::

            total = t.col2.sum(where=(t.col1 < 300) & (t.col2 < 400))

        Nullable sentinel values are skipped automatically::

            # Equivalent to summing only live rows where predicate is true and
            # t.col2 is not its configured null sentinel.
            total = t.col2.sum(where=t.col1 < 300)
        """
        self._require_kind("biufc", "sum")
        where = self._normalize_sum_where(where)
        # Use a wide accumulator to reduce overflow risk
        acc_dtype = np.dtype(dtype).type if dtype is not None else None
        if acc_dtype is None:
            acc_dtype = (
                np.float64
                if self.dtype.kind == "f"
                else (
                    np.complex128
                    if self.dtype.kind == "c"
                    else np.int64
                    if self.dtype.kind in "biu"
                    else None
                )
            )

        result = self._sum_lazy_fastpath(acc_dtype, where=where)
        if result is NotImplemented:
            if where is not None:
                return self._table.where(where)[self._col_name].sum(dtype=dtype)
            result = acc_dtype(0)
            for chunk in self._nonnull_chunks():
                result += chunk.sum(dtype=acc_dtype)

        # Return in the column's natural dtype when it fits, else keep the requested/wide dtype
        if dtype is None and self.dtype.kind in "biu":
            return int(result)
        return result

    def _lazy_aggregate_fastpath(self, op: str, *, where=None, dtype=None, ddof: int = 0):
        if self.is_list or self.is_varlen_scalar or self.dtype is None or self.dtype.kind not in "biuf":
            return NotImplemented
        raw = self._raw_col
        if not isinstance(raw, (blosc2.NDArray, blosc2.LazyExpr)):
            return NotImplemented
        mask = self._lazy_nonnull_mask(where=where)
        if mask is NotImplemented:
            return NotImplemented
        try:
            count = None
            if op in {"min", "max"}:
                count = int(mask.where(blosc2.ones(raw.shape, dtype=np.int64), 0).sum(dtype=np.int64))
                if count == 0:
                    raise ValueError(f"{op}() called on a column where all values are null.")
            if op == "mean":
                return float(raw.mean(where=mask, dtype=dtype or np.float64))
            if op == "std":
                return float(raw.std(where=mask, dtype=dtype or np.float64, ddof=ddof))
            if op == "min":
                return raw.min(where=mask)
            if op == "max":
                return raw.max(where=mask)
        except ValueError:
            if op in {"mean", "std"}:
                return float("nan")
            raise
        except Exception:
            return NotImplemented
        return NotImplemented

    def min(self, *, where=None):
        """Minimum live, non-null value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        """
        self._require_kind("biufUS", "min")
        where = self._normalize_sum_where(where)
        if where is None:
            self._require_nonempty("min")
        fast = self._lazy_aggregate_fastpath("min", where=where)
        if fast is not NotImplemented:
            return fast
        if where is not None:
            return self._table.where(where)[self._col_name].min()
        result = None
        is_str = self.dtype.kind in "US"
        for chunk in self._nonnull_chunks():
            # numpy .min()/.max() don't support string dtypes in recent NumPy;
            # fall back to Python's built-in min/max which work on any comparable type.
            chunk_min = min(chunk) if is_str else chunk.min()
            if result is None or chunk_min < result:
                result = chunk_min
        if result is None:
            raise ValueError("min() called on a column where all values are null.")
        return result

    def max(self, *, where=None):
        """Maximum live, non-null value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        """
        self._require_kind("biufUS", "max")
        where = self._normalize_sum_where(where)
        if where is None:
            self._require_nonempty("max")
        fast = self._lazy_aggregate_fastpath("max", where=where)
        if fast is not NotImplemented:
            return fast
        if where is not None:
            return self._table.where(where)[self._col_name].max()
        result = None
        is_str = self.dtype.kind in "US"
        for chunk in self._nonnull_chunks():
            chunk_max = max(chunk) if is_str else chunk.max()
            if result is None or chunk_max > result:
                result = chunk_max
        if result is None:
            raise ValueError("max() called on a column where all values are null.")
        return result

    def mean(self, *, where=None) -> float:
        """Arithmetic mean of all live, non-null values.

        Supported dtypes: bool, int, uint, float.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        Always returns a Python float.
        """
        self._require_kind("biuf", "mean")
        where = self._normalize_sum_where(where)
        if where is None and len(self) == 0:
            if self._table.base is not None:
                return float("nan")
            self._require_nonempty("mean")
        fast = self._lazy_aggregate_fastpath("mean", where=where)
        if fast is not NotImplemented:
            return fast
        if where is not None:
            return self._table.where(where)[self._col_name].mean()
        total = np.float64(0)
        count = 0
        for chunk in self._nonnull_chunks():
            total += chunk.sum(dtype=np.float64)
            count += len(chunk)
        if count == 0:
            return float("nan")
        return float(total / count)

    def std(self, ddof: int = 0, *, where=None) -> float:
        """Standard deviation of all live, non-null values (single-pass, Welford's algorithm).

        Parameters
        ----------
        ddof:
            Delta degrees of freedom.  ``0`` (default) gives the population
            std; ``1`` gives the sample std (divides by N-1).
        where:
            Optional boolean predicate. Only rows where the predicate is true,
            the table row is live, and this column is non-null are included.

        Supported dtypes: bool, int, uint, float.
        Null sentinel values are skipped.
        Always returns a Python float.
        """
        self._require_kind("biuf", "std")
        where = self._normalize_sum_where(where)
        if where is None and len(self) == 0:
            if self._table.base is not None:
                return float("nan")
            self._require_nonempty("std")
        fast = self._lazy_aggregate_fastpath("std", where=where, ddof=ddof)
        if fast is not NotImplemented:
            return fast
        if where is not None:
            return self._table.where(where)[self._col_name].std(ddof=ddof)

        # Chan's parallel update — combines per-chunk (n, mean, M2) tuples.
        # This is numerically stable and requires only a single pass.
        n_total = np.int64(0)
        mean_total = np.float64(0)
        M2_total = np.float64(0)

        for chunk in self._nonnull_chunks():
            chunk = chunk.astype(np.float64)
            n_b = np.int64(len(chunk))
            mean_b = chunk.mean()
            M2_b = np.float64(((chunk - mean_b) ** 2).sum())

            if n_total == 0:
                n_total, mean_total, M2_total = n_b, mean_b, M2_b
            else:
                delta = mean_b - mean_total
                n_new = n_total + n_b
                mean_total = (n_total * mean_total + n_b * mean_b) / n_new
                M2_total += M2_b + delta**2 * n_total * n_b / n_new
                n_total = n_new

        divisor = n_total - ddof
        if divisor <= 0:
            return float("nan")
        return float(np.sqrt(M2_total / divisor))

    def any(self) -> bool:
        """Return True if at least one live, non-null value is True.

        Supported dtypes: bool.
        Null sentinel values are skipped.
        Short-circuits on the first True found.
        """
        self._require_kind("b", "any")
        return any(chunk.any() for chunk in self._nonnull_chunks())

    def all(self) -> bool:
        """Return True if every live, non-null value is True.

        Supported dtypes: bool.
        Null sentinel values are skipped.
        Short-circuits on the first False found.
        """
        self._require_kind("b", "all")
        return all(chunk.all() for chunk in self._nonnull_chunks())


# ---------------------------------------------------------------------------
# CTable
# ---------------------------------------------------------------------------


def _fmt_bytes(n: int) -> str:
    """Human-readable byte count (e.g. '1.23 MB')."""
    if n < 1024:
        return f"{n} B"
    if n < 1024**2:
        return f"{n / 1024:.2f} KB"
    if n < 1024**3:
        return f"{n / 1024**2:.2f} MB"
    return f"{n / 1024**3:.2f} GB"


_EXPECTED_SIZE_DEFAULT = 1_048_576
_BATCH_SIZE_DEFAULT = 2048

# ---------------------------------------------------------------------------
# Computed-column definition (virtual columns backed by a LazyExpr)
# ---------------------------------------------------------------------------

# Each entry in CTable._computed_cols maps column name → this dict shape:
#   {
#     "expression": str,        # LazyExpr.expression string (for serialization)
#     "col_deps":  list[str],   # dep column names in operand order (o0=col_deps[0], …)
#     "lazy":      LazyExpr,    # the live lazy expression (holds NDArray refs)
#     "dtype":     np.dtype,    # result dtype
#   }
# We use a plain dict so that nothing extra needs to be imported.


class CTable(Generic[RowT]):
    """Columnar compressed table with typed columns and row-oriented access."""

    #: Ordered list of stored column names.  Computed columns are **not**
    #: included; access those via :attr:`computed_columns`.
    col_names: list[str]

    #: Parent table when this instance is a row-filter or column-projection
    #: view (created by :meth:`where`, :meth:`select`, or :meth:`view`).
    #: ``None`` for top-level tables.  Structural mutations such as
    #: :meth:`add_column` and :meth:`drop_column` are blocked on views.
    base: CTable | None

    def __init__(
        self,
        row_type: type[RowT],
        new_data=None,
        *,
        urlpath: str | None = None,
        mode: str = "a",
        expected_size: int | None = None,
        compact: bool = False,
        validate: bool = True,
        cparams: dict[str, Any] | None = None,
        dparams: dict[str, Any] | None = None,
    ) -> None:
        # Auto-size: if the caller didn't specify expected_size and new_data has a
        # known length, pre-allocate just enough (×2 for headroom, min 64).
        # Fall back to 1 M when new_data has no __len__ or is absent.
        if expected_size is None:
            if new_data is not None and hasattr(new_data, "__len__"):
                expected_size = max(len(new_data) * 2, 64)
            else:
                expected_size = _EXPECTED_SIZE_DEFAULT
        self._row_type = row_type
        self._validate = validate
        self._table_cparams = cparams
        self._table_dparams = dparams
        self._cols: dict[str, blosc2.NDArray | ListArray] = {}
        self._computed_cols: dict[str, dict] = {}  # virtual/computed columns
        self._materialized_cols: dict[str, dict] = {}  # stored columns auto-filled from expressions
        self._expr_index_arrays: dict[str, blosc2.NDArray] = {}
        self._col_widths: dict[str, int] = {}
        self.col_names: list[str] = []
        self.auto_compact = compact
        self.base = None

        # Choose storage backend
        if urlpath is not None:
            if mode == "w" and os.path.exists(urlpath):
                if os.path.isdir(urlpath):
                    shutil.rmtree(urlpath)
                else:
                    os.remove(urlpath)
            storage: TableStorage = FileTableStorage(urlpath, mode)
        else:
            storage = InMemoryTableStorage()
        self._storage = storage
        self._read_only = storage.is_read_only()

        if storage.table_exists() and mode != "w":
            # ---- Open existing persistent table ----
            if new_data is not None:
                raise ValueError(
                    "Cannot pass new_data when opening an existing table. Use mode='w' to overwrite."
                )
            storage.check_kind()
            schema_dict = storage.load_schema()
            self._schema: CompiledSchema = schema_from_dict(schema_dict)
            self._schema = CompiledSchema(
                row_cls=row_type,
                columns=self._schema.columns,
                columns_by_name=self._schema.columns_by_name,
            )
            self.col_names = [c["name"] for c in schema_dict["columns"]]
            self._valid_rows = storage.open_valid_rows()
            for name in self.col_names:
                cc = self._schema.columns_by_name[name]
                if self._is_list_column(cc):
                    col = storage.open_list_column(name)
                elif self._is_varlen_scalar_column(cc):
                    col = storage.open_varlen_scalar_column(name, cc.spec)
                else:
                    col = storage.open_column(name)
                self._cols[name] = col
                self._col_widths[name] = max(len(name), cc.display_width)
            self._n_rows = int(blosc2.count_nonzero(self._valid_rows))
            self._last_pos = None  # resolve lazily on first write
            # ---- Restore computed/materialized column metadata (if any) ----
            self._computed_cols = {}
            self._materialized_cols = {}
            self._expr_index_arrays = {}
            self._load_computed_cols_from_schema(schema_dict)
            self._load_materialized_cols_from_schema(schema_dict)
        else:
            # ---- Create new table ----
            if storage.is_read_only():
                raise FileNotFoundError(f"No CTable found at {urlpath!r}")

            # Build compiled schema from either a dataclass or a legacy Pydantic model
            if dataclasses.is_dataclass(row_type) and isinstance(row_type, type):
                self._schema = compile_schema(row_type)
            else:
                self._schema = _compile_pydantic_schema(row_type)
            self._resolve_nullable_specs(self._schema)

            self._n_rows = 0
            self._last_pos = 0

            default_chunks, default_blocks = compute_chunks_blocks((expected_size,))
            self._valid_rows = storage.create_valid_rows(
                shape=(expected_size,),
                chunks=default_chunks,
                blocks=default_blocks,
            )
            self._init_columns(expected_size, default_chunks, default_blocks, storage)
            storage.save_schema(schema_to_dict(self._schema))

            if new_data is not None:
                self._load_initial_data(new_data)

    def close(self) -> None:
        """Close any persistent backing store held by this table."""
        storage = getattr(self, "_storage", None)
        try:
            self._flush_varlen_columns()
        except Exception:
            with contextlib.suppress(Exception):
                if storage is not None and hasattr(storage, "close"):
                    storage.close()
            raise
        if storage is not None and hasattr(storage, "close"):
            storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            storage = getattr(self, "_storage", None)
            if storage is not None and hasattr(storage, "discard"):
                storage.discard()
            elif storage is not None and hasattr(storage, "close"):
                storage.close()

    @staticmethod
    def _is_list_column(col: CompiledColumn) -> bool:
        return isinstance(col.spec, ListSpec)

    @staticmethod
    def _is_varlen_scalar_column(col: CompiledColumn) -> bool:
        return isinstance(col.spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec))

    @staticmethod
    def _is_list_spec(spec: SchemaSpec) -> bool:
        return isinstance(spec, ListSpec)

    @staticmethod
    def _policy_null_value_for_spec(spec: SchemaSpec, policy: NullPolicy):
        if isinstance(spec, (int8, int16, int32, int64)):
            info = np.iinfo(spec.dtype)
            return info.min if policy.signed_int_strategy == "min" else info.max
        if isinstance(spec, (uint8, uint16, uint32, uint64)):
            info = np.iinfo(spec.dtype)
            return info.min if policy.unsigned_int_strategy == "min" else info.max
        if isinstance(spec, (float32, float64)):
            return policy.float_value
        if isinstance(spec, b2_bool):
            return policy.bool_value
        if isinstance(spec, string):
            return policy.string_value
        if isinstance(spec, b2_bytes):
            return policy.bytes_value
        return None

    @staticmethod
    def _validate_null_value_for_spec(name: str, spec: SchemaSpec, null_value) -> None:
        if isinstance(spec, (int8, int16, int32, int64, uint8, uint16, uint32, uint64)):
            if isinstance(null_value, (bool, np.bool_)) or not isinstance(null_value, (int, np.integer)):
                raise TypeError(f"Null sentinel for column {name!r} must be an integer")
            info = np.iinfo(spec.dtype)
            if not info.min <= int(null_value) <= info.max:
                raise ValueError(
                    f"Null sentinel for column {name!r}={null_value!r} is outside {spec.dtype} range"
                )
            return
        if isinstance(spec, (float32, float64)):
            if not isinstance(null_value, (int, float, np.integer, np.floating)):
                raise TypeError(f"Null sentinel for column {name!r} must be numeric")
            return
        if isinstance(spec, b2_bool):
            if null_value != 255:
                raise ValueError(f"Null sentinel for nullable bool column {name!r} must be 255")
            return
        if isinstance(spec, string):
            if not isinstance(null_value, str):
                raise TypeError(f"Null sentinel for string column {name!r} must be str")
            return
        if isinstance(spec, b2_bytes) and not isinstance(null_value, bytes):
            raise TypeError(f"Null sentinel for bytes column {name!r} must be bytes")

    @classmethod
    def _resolve_nullable_specs(
        cls, schema: CompiledSchema, *, validate_column_null_values: bool = True
    ) -> None:
        policy = get_null_policy()
        schema_names = {col.name for col in schema.columns}
        unknown_null_values = set(policy.column_null_values) - schema_names
        if validate_column_null_values and unknown_null_values:
            names = ", ".join(sorted(unknown_null_values))
            raise KeyError(f"column_null_values contains unknown columns: {names}")
        for col in schema.columns:
            spec = col.spec
            if (
                isinstance(spec, (ListSpec, VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec))
                or getattr(spec, "null_value", None) is not None
            ):
                continue
            if not getattr(spec, "nullable", False):
                continue
            null_value = policy.column_null_values.get(col.name)
            if null_value is None:
                null_value = cls._policy_null_value_for_spec(spec, policy)
            if null_value is None:
                raise TypeError(f"Column {col.name!r} is nullable, but no null policy sentinel is available")
            cls._validate_null_value_for_spec(col.name, spec, null_value)
            spec.null_value = null_value
            if isinstance(spec, string):
                spec.max_length = max(spec.max_length, len(null_value), 1)
                spec.dtype = np.dtype(f"U{spec.max_length}")
            elif isinstance(spec, b2_bytes):
                spec.max_length = max(spec.max_length, len(null_value), 1)
                spec.dtype = np.dtype(f"S{spec.max_length}")
            elif isinstance(spec, b2_bool):
                spec.dtype = np.dtype(np.uint8)
            col.dtype = getattr(spec, "dtype", None)
            col.display_width = compute_display_width(spec)

    def _flush_varlen_columns(self) -> None:
        for col in self._schema.columns:
            if self._is_list_column(col) or self._is_varlen_scalar_column(col):
                self._cols[col.name].flush()

    def _init_columns(
        self, expected_size: int, default_chunks, default_blocks, storage: TableStorage
    ) -> None:
        """Create one physical column per compiled schema column."""
        for col in self._schema.columns:
            self.col_names.append(col.name)
            self._col_widths[col.name] = max(len(col.name), col.display_width)
            col_storage = self._resolve_column_storage(col, default_chunks, default_blocks)
            if self._is_list_column(col):
                self._cols[col.name] = storage.create_list_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
                continue
            if self._is_varlen_scalar_column(col):
                self._cols[col.name] = storage.create_varlen_scalar_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
                continue
            # Recompute chunks/blocks using the actual dtype so that wide
            # string columns (e.g. U183642) don't produce multi-GB chunks.
            chunks = col_storage["chunks"]
            blocks = col_storage["blocks"]
            if col.config.chunks is None and col.config.blocks is None:
                chunks, blocks = compute_chunks_blocks((expected_size,), dtype=col.dtype)
            self._cols[col.name] = storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=(expected_size,),
                chunks=chunks,
                blocks=blocks,
                cparams=col_storage.get("cparams"),
                dparams=col_storage.get("dparams"),
            )

    def _resolve_column_storage(
        self,
        col: CompiledColumn,
        default_chunks,
        default_blocks,
    ) -> dict[str, Any]:
        """Merge table-level and column-level storage settings.

        Column-level settings (from ``b2.field(...)``) take precedence over
        table-level defaults passed to ``CTable.__init__``.
        """
        result: dict[str, Any] = {
            "chunks": col.config.chunks if col.config.chunks is not None else default_chunks,
            "blocks": col.config.blocks if col.config.blocks is not None else default_blocks,
        }
        cparams = col.config.cparams if col.config.cparams is not None else self._table_cparams
        dparams = col.config.dparams if col.config.dparams is not None else self._table_dparams
        if cparams is not None:
            result["cparams"] = cparams
        if dparams is not None:
            result["dparams"] = dparams
        return result

    def _normalize_row_input(self, data: Any) -> dict[str, Any]:
        """Normalize a row input to a ``{col_name: value}`` dict.

        Accepted shapes:
        - list / tuple  → positional, zipped with stored column names (computed columns skipped)
        - dict          → used as-is
        - dataclass     → ``dataclasses.asdict``
        - np.void / structured scalar → field-name access
        """
        stored = self._append_input_col_names
        if isinstance(data, dict):
            return data
        if isinstance(data, (list, tuple)):
            return dict(zip(stored, data, strict=False))
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return dataclasses.asdict(data)
        if isinstance(data, (np.void, np.record)):
            return {name: data[name] for name in stored}
        # Fallback: try positional indexing
        return {name: data[i] for i, name in enumerate(stored)}

    def _coerce_row_to_storage(self, row: dict[str, Any]) -> dict[str, Any]:
        """Coerce each value in *row* to the column's storage representation."""
        result = {}
        for col in self._schema.columns:
            val = row[col.name]
            if self._is_list_column(col):
                result[col.name] = coerce_list_cell(col.spec, val)
            elif self._is_varlen_scalar_column(col):
                # Coercion is handled inside _ScalarVarLenArray.append.
                result[col.name] = val
            else:
                result[col.name] = np.array(val, dtype=col.dtype).item()
        return result

    def _resolve_last_pos(self) -> int:
        """Return the physical index of the next write slot.

        Returns the cached ``_last_pos`` when available.  After a deletion
        ``_last_pos`` is ``None``; this method then walks chunk metadata of
        ``_valid_rows`` from the end (no full decompression) to find the last
        ``True`` position, caches the result, and returns it.
        """
        if self._last_pos is not None:
            return self._last_pos

        arr = self._valid_rows
        chunk_size = arr.chunks[0]
        last_true_pos = -1

        for info in reversed(list(arr.iterchunks_info())):
            actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
            chunk_start = info.nchunk * chunk_size

            if info.special == blosc2.SpecialValue.ZERO:
                continue
            if info.special == blosc2.SpecialValue.VALUE:
                val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                if not val:
                    continue
                last_true_pos = chunk_start + actual_size - 1
                break

            chunk_data = arr[chunk_start : chunk_start + actual_size]
            nonzero = np.flatnonzero(chunk_data)
            if len(nonzero) == 0:
                continue
            last_true_pos = chunk_start + int(nonzero[-1])
            break

        self._last_pos = last_true_pos + 1
        return self._last_pos

    def _grow(self) -> None:
        """Double the scalar-column capacity and the valid_rows mask."""
        c = len(self._valid_rows)
        for name, col_arr in self._cols.items():
            cc = self._schema.columns_by_name[name]
            if self._is_list_column(cc) or self._is_varlen_scalar_column(cc):
                continue
            col_arr.resize((c * 2,))
        self._valid_rows.resize((c * 2,))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_positions(self, head_tail: int = 10):
        nrows = self._n_rows
        hidden = max(0, nrows - head_tail * 2)
        valid_np = self._valid_rows[:]
        all_pos = np.where(valid_np)[0]
        if nrows <= head_tail * 2:
            return all_pos, np.array([], dtype=all_pos.dtype), 0
        return all_pos[:head_tail], all_pos[-head_tail:], hidden

    def _display_widths(self) -> dict[str, int]:
        widths: dict[str, int] = {}
        single_col = len(self.col_names) == 1
        for name in self.col_names:
            spec = self._schema.columns_by_name.get(name)
            dtype_label = self._dtype_info_label(self._col_dtype(name), spec.spec if spec else None)
            widths[name] = max(self._col_widths[name], len(dtype_label))
            if single_col:
                widths[name] = max(widths[name], 80)
        return widths

    @staticmethod
    def _format_cell(value, width: int) -> str:
        s = str(value)
        if len(s) > width:
            s = s[: width - 1] + "…"
        return f" {s:<{width}} "

    def _format_display_row(self, values: dict, widths: dict[str, int]) -> str:
        return "  ".join(self._format_cell(values[n], widths[n]) for n in self.col_names)

    def _rows_to_dicts(self, positions) -> list[dict]:
        if len(positions) == 0:
            return []
        col_data = {n: self._fetch_col_at_positions(n, positions) for n in self.col_names}
        rows = []
        for i in range(len(positions)):
            row = {}
            for n in self.col_names:
                row[n] = self._normalize_scalar_value(col_data[n][i])
            rows.append(row)
        return rows

    def __str__(self) -> str:
        """Pandas-style tabular display with column names, dtypes, and a row count footer."""
        nrows = self._n_rows
        ncols = len(self.col_names)
        head_pos, tail_pos, hidden = self._display_positions()
        widths = self._display_widths()
        sep = "  ".join("─" * (w + 2) for w in widths.values())

        lines = [
            self._format_display_row({n: n for n in self.col_names}, widths),
            self._format_display_row(
                {
                    n: self._dtype_info_label(
                        self._col_dtype(n),
                        self._schema.columns_by_name[n].spec if n in self._schema.columns_by_name else None,
                    )
                    for n in self.col_names
                },
                widths,
            ),
            sep,
        ]
        lines.extend(self._format_display_row(row, widths) for row in self._rows_to_dicts(head_pos))
        if hidden > 0:
            lines.append(self._format_display_row(dict.fromkeys(self.col_names, "..."), widths))
        lines.extend(self._format_display_row(row, widths) for row in self._rows_to_dicts(tail_pos))
        lines.append(sep)
        footer = f"{nrows:,} rows × {ncols} columns"
        if hidden > 0:
            footer += f"  ({hidden:,} rows hidden)"
        lines.append(footer)
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Short ``CTable<cols>(N rows, X compressed)`` summary string."""
        cols = ", ".join(self.col_names)
        return f"CTable<{cols}>({self._n_rows:,} rows, {_fmt_bytes(self.cbytes)} compressed)"

    def __len__(self):
        """Return the number of live (non-deleted) rows."""
        return self._n_rows

    def __iter__(self):
        """Iterate over live rows in insertion order, yielding namedtuple-like row objects."""
        for i in range(self.nrows):
            yield self._materialize_row(i)

    def _row_namedtuple_type(self):
        visible = tuple(self.col_names)
        if getattr(self, "_row_namedtuple_type_cache_cols", None) != visible:
            self._row_namedtuple_type_cache = _make_namedtuple_row_type(visible)
            self._row_namedtuple_type_cache_cols = visible
        return self._row_namedtuple_type_cache

    @staticmethod
    def _normalize_scalar_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value

    def _physical_row_value(self, col_name: str, pos: int):
        cc = self._computed_cols.get(col_name)
        if cc is not None:
            return self._normalize_scalar_value(np.asarray(cc["lazy"][pos]).ravel()[0])
        return self._normalize_scalar_value(self._cols[col_name][pos])

    def _materialize_row(self, index: int):
        n_rows = self.nrows
        if index < 0:
            index += n_rows
        if not (0 <= index < n_rows):
            raise IndexError(f"row index {index} is out of bounds for table with {n_rows} rows")
        pos = _find_physical_index(self._valid_rows, index)
        row_type = self._row_namedtuple_type()
        return row_type(*(self._physical_row_value(name, int(pos)) for name in self.col_names))

    def iter_sorted(
        self,
        cols: str | list[str],
        ascending: bool | list[bool] = True,
        *,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
        batch_size: int = 4096,
    ):
        """Iterate rows in sorted order without materializing a full copy.

        Uses a FULL index when available (no sort needed); otherwise falls
        back to ``np.lexsort`` on live physical positions.  Yields namedtuple-like
        row objects in the same way as ``__iter__``.

        The sorted positions array is stored as a compressed ``blosc2.NDArray``
        to keep RAM usage low for large tables.  ``batch_size`` positions are
        decompressed at a time during iteration.

        Parameters
        ----------
        cols:
            Column name or list of column names to sort by.
        ascending:
            Sort direction.  A single bool applies to all keys; a list must
            have the same length as *cols*.
        start, stop, step:
            Optional slice applied to the sorted sequence before iteration.
            E.g. ``stop=10`` yields only the top-10 rows; ``step=2`` yields
            every other row in sorted order.
        batch_size:
            Number of positions decompressed per iteration step.  Larger
            values reduce decompression overhead; smaller values use less
            transient RAM.  Default is 4096.
        """
        cols, ascending = self._normalise_sort_keys(cols, ascending)

        valid_np = self._valid_rows[:]
        live_pos = np.where(valid_np)[0]
        n = len(live_pos)

        if n == 0:
            return

        sorted_pos = None
        if len(cols) == 1:
            sorted_pos = self._sorted_positions_from_full_index(cols[0], ascending[0])
            if sorted_pos is not None and len(sorted_pos) != n:
                sorted_pos = None

        if sorted_pos is None:
            order = np.lexsort(self._build_lex_keys(cols, ascending, live_pos, n))
            sorted_pos = live_pos[order]

        if start is not None or stop is not None or step is not None:
            sorted_pos = sorted_pos[start:stop:step]

        # Compress positions into an NDArray to reduce RAM usage for large tables.
        # The uncompressed numpy array is released immediately after.
        sorted_pos_nd = blosc2.asarray(np.asarray(sorted_pos, dtype=np.int64))
        del sorted_pos

        # physical → logical index mapping
        phys_to_logical = np.empty(valid_np.shape[0], dtype=np.intp)
        phys_to_logical[live_pos] = np.arange(n, dtype=np.intp)

        total = len(sorted_pos_nd)
        for i in range(0, total, batch_size):
            chunk = sorted_pos_nd[i : i + batch_size]
            for phys in chunk:
                yield self._materialize_row(int(phys_to_logical[phys]))

    # ------------------------------------------------------------------
    # Open existing table (classmethod)
    # ------------------------------------------------------------------

    @classmethod
    def open(cls, urlpath: str, *, mode: str = "r") -> CTable:
        """Open a persistent CTable from *urlpath*.

        Parameters
        ----------
        urlpath:
            Path to the table root directory (created by passing ``urlpath``
            to :class:`CTable`).
        mode:
            ``'r'`` (default) — read-only.
            ``'a'`` — read/write.

        Raises
        ------
        FileNotFoundError
            If *urlpath* does not contain a CTable.
        ValueError
            If the metadata at *urlpath* does not identify a CTable.
        """
        storage = FileTableStorage(urlpath, mode)
        if not storage.table_exists():
            raise FileNotFoundError(f"No CTable found at {urlpath!r}")
        return cls._open_from_storage(storage)

    def to_b2z(self, urlpath: str, *, overwrite: bool = False, compact: bool = False) -> str:
        """Write this table to a compact ``.b2z`` container.

        ``.b2z`` is the compact zip-backed CTable format.  For persistent,
        non-view directory-backed tables and ``compact=False``, this uses a
        fast physical-pack path: the backing :class:`TreeStore` directory is
        zipped with already-compressed leaves stored as-is. This preserves the
        physical layout, including deleted rows and spare capacity, and does
        not recompress columns.  A ``.b2d`` suffix is recommended for
        directory-backed stores, but not required.

        For in-memory tables, views, existing ``.b2z`` tables, or
        ``compact=True``, this falls back to the logical :meth:`save` path,
        materializing only visible/live rows into a new ``.b2z`` store.

        Examples
        --------
        Fast-pack an existing directory-backed table into a compact zip store::

            table = blosc2.CTable.open("data.b2d", mode="r")
            table.to_b2z("data.b2z", overwrite=True)
            table.close()

        Materialize a filtered view into a new compact store::

            view = table.where(table["score"] > 10)
            view.to_b2z("high-score.b2z", overwrite=True)

        Force a logical compacted copy, even for a persistent ``.b2d`` table::

            table.to_b2z("data-compact.b2z", overwrite=True, compact=True)
        """
        if not str(urlpath).endswith(".b2z"):
            raise ValueError("urlpath must have a .b2z extension")

        storage = getattr(self, "_storage", None)
        can_physical_pack = (
            not compact
            and self.base is None
            and isinstance(storage, FileTableStorage)
            and not str(storage._root).endswith(".b2z")
        )
        if can_physical_pack:
            self._flush_varlen_columns()
            store = blosc2.TreeStore(storage._root, mode="r")
            try:
                return store.to_b2z(filename=urlpath, overwrite=overwrite)
            finally:
                store.close()

        if self.base is not None:
            materialized = self.copy(compact=True)
            materialized.save(urlpath, overwrite=overwrite)
        else:
            self.save(urlpath, overwrite=overwrite)
        return os.path.abspath(urlpath)

    def to_b2d(self, urlpath: str, *, overwrite: bool = False, compact: bool = False) -> str:
        """Write this table to a directory-backed store.

        Directory-backed CTable stores may use any path that does not end in
        ``.b2z``; using a ``.b2d`` suffix is recommended for clarity.  For
        persistent, non-view ``.b2z`` tables opened read-only and
        ``compact=False``, this uses a fast physical-unpack path: the zip
        members are extracted as already-compressed leaves. This preserves the
        physical layout, including deleted rows and spare capacity, and does
        not recompress columns.

        For in-memory tables, views, writable ``.b2z`` tables, existing
        directory-backed tables, or ``compact=True``, this falls back to the
        logical :meth:`save` path, materializing only visible/live rows into a
        new directory-backed store.

        Examples
        --------
        Fast-unpack an existing compact zip store into a directory-backed table::

            table = blosc2.CTable.open("data.b2z", mode="r")
            table.to_b2d("data.b2d", overwrite=True)
            table.close()

        Materialize a filtered view into a directory-backed store::

            view = table.where(table["score"] > 10)
            view.to_b2d("high-score.b2d", overwrite=True)

        Force a logical compacted copy, even for a persistent ``.b2z`` table::

            table.to_b2d("data-compact.b2d", overwrite=True, compact=True)
        """
        urlpath = os.fspath(urlpath)
        storage = getattr(self, "_storage", None)
        can_physical_unpack = (
            not compact
            and self.base is None
            and isinstance(storage, FileTableStorage)
            and str(storage._root).endswith(".b2z")
            and storage.open_mode() == "r"
        )
        if can_physical_unpack:
            store = blosc2.TreeStore(storage._root, mode="r")
            try:
                return store.to_b2d(urlpath, overwrite=overwrite)
            finally:
                store.close()

        if self.base is not None:
            materialized = self.copy(compact=True)
            materialized.save(urlpath, overwrite=overwrite)
        else:
            self.save(urlpath, overwrite=overwrite)
        return os.path.abspath(urlpath)

    def _save_to_storage(self, storage: TableStorage) -> None:
        """Write all live rows and columns into *storage*.

        The caller is responsible for calling ``storage.close()`` when done.
        This method does **not** close *storage*.
        """
        self._flush_varlen_columns()

        # Collect live physical positions
        valid_np = self._valid_rows[:]
        live_pos = np.where(valid_np)[0]
        n_live = len(live_pos)
        capacity = max(n_live, 1)

        default_chunks, default_blocks = compute_chunks_blocks((capacity,))

        # --- valid_rows (all True, compacted) ---
        disk_valid = storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        if n_live > 0:
            disk_valid[:n_live] = True

        # --- columns ---
        for col in self._schema.columns:
            name = col.name
            if self._is_list_column(col):
                disk_col = storage.create_list_column(
                    name,
                    spec=col.spec,
                    cparams=col.config.cparams if col.config.cparams is not None else self._table_cparams,
                    dparams=col.config.dparams if col.config.dparams is not None else self._table_dparams,
                )
                if n_live > 0:
                    disk_col.extend((self._cols[name][int(pos)] for pos in live_pos), validate=False)
                    disk_col.flush()
                continue
            if self._is_varlen_scalar_column(col):
                disk_col = storage.create_varlen_scalar_column(
                    name,
                    spec=col.spec,
                    cparams=col.config.cparams if col.config.cparams is not None else self._table_cparams,
                    dparams=col.config.dparams if col.config.dparams is not None else self._table_dparams,
                )
                if n_live > 0:
                    disk_col.extend(self._cols[name][int(pos)] for pos in live_pos)
                    disk_col.flush()
                continue
            dtype_chunks, dtype_blocks = compute_chunks_blocks((capacity,), dtype=col.dtype)
            col_storage = self._resolve_column_storage(col, dtype_chunks, dtype_blocks)
            disk_col = storage.create_column(
                name,
                dtype=col.dtype,
                shape=(capacity,),
                chunks=col_storage["chunks"],
                blocks=col_storage["blocks"],
                cparams=col_storage.get("cparams"),
                dparams=col_storage.get("dparams"),
            )
            if n_live > 0:
                disk_col[:n_live] = self._cols[name][live_pos]

        storage.save_schema(self._schema_dict_with_computed())

    def save(self, urlpath: str, *, overwrite: bool = False) -> None:
        """Persist this table to disk at *urlpath*.

        This writes a standalone copy and returns ``None``; use :meth:`copy`
        directly when the copied :class:`CTable` object is needed.

        Only live rows are written — the on-disk table is always compacted.
        A ``.b2z`` suffix selects the compact zip-backed format; any other
        suffix creates a directory-backed store.  Use a ``.b2d`` suffix for
        directory-backed stores when possible so the format is clear.

        Parameters
        ----------
        urlpath:
            Destination path.  Use a ``.b2z`` suffix for a compact zip-backed
            store; any other suffix creates a directory-backed store.  A
            ``.b2d`` suffix is recommended for directory-backed stores.
        overwrite:
            If ``False`` (default), raise :exc:`ValueError` when *urlpath*
            already exists.  Set to ``True`` to replace an existing table.

        Raises
        ------
        ValueError
            If *urlpath* already exists and ``overwrite=False``.
        """
        if self.base is not None:
            materialized = self.copy(compact=True)
            materialized.save(urlpath, overwrite=overwrite)
            return

        file_storage = FileTableStorage(urlpath, "w")
        target_path = file_storage._root
        if os.path.exists(target_path):
            if not overwrite:
                raise ValueError(f"Path {target_path!r} already exists. Use overwrite=True to replace.")
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)

        self._save_to_storage(file_storage)
        file_storage.close()

    @classmethod
    def _open_from_storage(cls, storage: TableStorage) -> CTable:
        """Construct a :class:`CTable` from an already-configured *storage* backend.

        The caller must have already verified that the storage target exists.
        This is the common open path shared by :meth:`open` and
        :meth:`_open_from_treestore`.
        """
        storage.check_kind()
        schema_dict = storage.load_schema()
        schema = schema_from_dict(schema_dict)
        col_names = [c["name"] for c in schema_dict["columns"]]

        obj = cls.__new__(cls)
        obj._row_type = None
        obj._validate = True
        obj._table_cparams = None
        obj._table_dparams = None
        obj._storage = storage
        obj._read_only = storage.is_read_only()
        obj._schema = schema
        obj._cols = {}
        obj._col_widths = {}
        obj.col_names = col_names
        obj.auto_compact = False
        obj.base = None

        obj._valid_rows = storage.open_valid_rows()
        for name in col_names:
            cc = schema.columns_by_name[name]
            if obj._is_list_column(cc):
                obj._cols[name] = storage.open_list_column(name)
            elif obj._is_varlen_scalar_column(cc):
                obj._cols[name] = storage.open_varlen_scalar_column(name, cc.spec)
            else:
                obj._cols[name] = storage.open_column(name)
            obj._col_widths[name] = max(len(name), cc.display_width)

        obj._n_rows = int(blosc2.count_nonzero(obj._valid_rows))
        obj._last_pos = None
        obj._computed_cols = {}
        obj._materialized_cols = {}
        obj._expr_index_arrays = {}
        obj._load_computed_cols_from_schema(schema_dict)
        obj._load_materialized_cols_from_schema(schema_dict)
        return obj

    def _save_to_treestore(self, store: blosc2.TreeStore, full_key: str) -> None:
        """Save this CTable inline into *store* under *full_key*.

        *full_key* must be the absolute (fully-translated) key within the
        backing DictStore (not a subtree-relative key).
        Internal use only — called by :class:`blosc2.TreeStore`.
        """
        if self.base is not None:
            materialized = self.copy(compact=True)
            materialized._save_to_treestore(store, full_key)
            return
        storage = TreeStoreTableStorage(store, full_key, mode="a", owns_store=False)
        self._save_to_storage(storage)
        # storage is non-owning; outer store handles persistence

    @classmethod
    def _open_from_treestore(cls, store: blosc2.TreeStore, full_key: str) -> CTable:
        """Open an inline CTable from *store* at *full_key*.

        *full_key* must be the absolute key within the backing DictStore.
        Internal use only — called by :class:`blosc2.TreeStore`.
        """
        storage = TreeStoreTableStorage(store, full_key, mode=store.mode, owns_store=False)
        if not storage.table_exists():
            raise FileNotFoundError(f"No inline CTable found at key {full_key!r} in {store.localpath!r}")
        return cls._open_from_storage(storage)

    @classmethod
    def load(cls, urlpath: str) -> CTable:
        """Load a persistent table from *urlpath* into RAM.

        The schema is read from the table's metadata — the original Python
        dataclass is not required.  The returned table is fully in-memory and
        read/write.

        Parameters
        ----------
        urlpath:
            Path to the table root directory.

        Raises
        ------
        FileNotFoundError
            If *urlpath* does not contain a CTable.
        ValueError
            If the metadata at *urlpath* does not identify a CTable.
        """
        file_storage = FileTableStorage(urlpath, "r")
        if not file_storage.table_exists():
            raise FileNotFoundError(f"No CTable found at {urlpath!r}")
        file_storage.check_kind()
        schema_dict = file_storage.load_schema()
        schema = schema_from_dict(schema_dict)
        col_names = [c["name"] for c in schema_dict["columns"]]

        disk_valid = file_storage.open_valid_rows()
        disk_cols = {}
        for col in schema.columns:
            if cls._is_list_column(col):
                disk_cols[col.name] = file_storage.open_list_column(col.name)
            elif cls._is_varlen_scalar_column(col):
                disk_cols[col.name] = file_storage.open_varlen_scalar_column(col.name, col.spec)
            else:
                disk_cols[col.name] = file_storage.open_column(col.name)
        phys_size = len(disk_valid)
        n_live = int(blosc2.count_nonzero(disk_valid))
        capacity = max(phys_size, 1)

        mem_storage = InMemoryTableStorage()
        bool_chunks, bool_blocks = compute_chunks_blocks((capacity,), dtype=np.dtype(np.bool_))

        mem_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=bool_chunks,
            blocks=bool_blocks,
        )
        if phys_size > 0:
            mem_valid[:phys_size] = disk_valid[:]

        mem_cols: dict[str, blosc2.NDArray | ListArray | _ScalarVarLenArray] = {}
        for col in schema.columns:
            name = col.name
            if cls._is_list_column(col):
                mem_col = mem_storage.create_list_column(name, spec=col.spec, cparams=None, dparams=None)
                mem_col.extend(disk_cols[name][:])
                mem_col.flush()
                mem_cols[name] = mem_col
                continue
            if cls._is_varlen_scalar_column(col):
                mem_col = mem_storage.create_varlen_scalar_column(name, spec=col.spec)
                mem_col.extend(iter(disk_cols[name]))
                mem_col.flush()
                mem_cols[name] = mem_col
                continue
            col_chunks, col_blocks = compute_chunks_blocks((capacity,), dtype=col.dtype)
            mem_col = mem_storage.create_column(
                name,
                dtype=col.dtype,
                shape=(capacity,),
                chunks=col_chunks,
                blocks=col_blocks,
                cparams=None,
                dparams=None,
            )
            if phys_size > 0:
                mem_col[:phys_size] = disk_cols[name][:]
            mem_cols[name] = mem_col

        file_storage.close()

        obj = cls.__new__(cls)
        obj._row_type = None
        obj._validate = True
        obj._table_cparams = None
        obj._table_dparams = None
        obj._storage = mem_storage
        obj._read_only = False
        obj._schema = schema
        obj._cols = mem_cols
        obj._col_widths = {col.name: max(len(col.name), col.display_width) for col in schema.columns}
        obj.col_names = col_names
        obj.auto_compact = False
        obj.base = None
        obj._valid_rows = mem_valid
        obj._n_rows = n_live
        obj._last_pos = None  # resolve lazily on first write
        obj._computed_cols = {}
        obj._materialized_cols = {}
        obj._expr_index_arrays = {}
        obj._load_computed_cols_from_schema(schema_dict)
        obj._load_materialized_cols_from_schema(schema_dict)
        return obj

    @classmethod
    def _make_view(cls, parent: CTable, new_valid_rows: blosc2.NDArray) -> CTable:
        """Construct a read-only view sharing *parent*'s columns."""
        obj = cls.__new__(cls)
        obj._row_type = parent._row_type
        obj._validate = parent._validate
        obj._table_cparams = parent._table_cparams
        obj._table_dparams = parent._table_dparams
        obj._storage = None
        obj._read_only = parent._read_only  # inherit: only True for mode="r" disk tables
        obj._schema = parent._schema
        obj._cols = parent._cols  # shared — views cannot change row structure
        obj._computed_cols = parent._computed_cols  # shared — LazyExpr refs remain valid
        obj._materialized_cols = parent._materialized_cols
        obj._expr_index_arrays = parent._expr_index_arrays
        obj._col_widths = parent._col_widths
        obj.col_names = parent.col_names
        obj.auto_compact = parent.auto_compact
        obj.base = parent
        obj._valid_rows = new_valid_rows
        obj._n_rows = int(blosc2.count_nonzero(new_valid_rows))
        obj._last_pos = None
        return obj

    def view(self, new_valid_rows):
        """Return a row-filter view backed by a boolean mask array without copying data."""
        if isinstance(new_valid_rows, np.ndarray) and new_valid_rows.dtype == np.bool_:
            new_valid_rows = blosc2.asarray(new_valid_rows)
        if not (
            isinstance(new_valid_rows, (blosc2.NDArray, blosc2.LazyExpr))
            and (getattr(new_valid_rows, "dtype", None) == np.bool_)
        ):
            raise TypeError(
                f"Expected boolean blosc2.NDArray or LazyExpr, got {type(new_valid_rows).__name__}"
            )

        new_valid_rows = (
            new_valid_rows.compute() if isinstance(new_valid_rows, blosc2.LazyExpr) else new_valid_rows
        )

        if len(self._valid_rows) != len(new_valid_rows):
            raise ValueError()

        return CTable._make_view(self, new_valid_rows)

    def head(self, N: int = 5) -> CTable:
        """Return a view of the first *N* live rows (default 5)."""
        if N <= 0:
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))
        if self._n_rows <= N:
            return self.view(self._valid_rows)

        # Reuse _find_physical_index: physical position of the (N-1)-th live row
        arr = self._valid_rows
        pos_N_true = _find_physical_index(arr, N - 1)

        if pos_N_true < len(arr) // 2:
            mask_arr = blosc2.zeros(shape=len(arr), dtype=np.bool_)
            mask_arr[: pos_N_true + 1] = True
        else:
            mask_arr = blosc2.ones(shape=len(arr), dtype=np.bool_)
            mask_arr[pos_N_true + 1 :] = False

        mask_arr = (mask_arr & self._valid_rows).compute()
        return self.view(mask_arr)

    def tail(self, N: int = 5) -> CTable:
        """Return a view of the last *N* live rows (default 5)."""
        if N <= 0:
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))
        if self._n_rows <= N:
            return self.view(self._valid_rows)

        # Physical position of the first row we want = logical index (nrows - N)
        arr = self._valid_rows
        pos_start = _find_physical_index(arr, self._n_rows - N)

        if pos_start > len(arr) // 2:
            mask_arr = blosc2.zeros(shape=len(arr), dtype=np.bool_)
            mask_arr[pos_start:] = True
        else:
            mask_arr = blosc2.ones(shape=len(arr), dtype=np.bool_)
            if pos_start > 0:
                mask_arr[:pos_start] = False

        mask_arr = (mask_arr & self._valid_rows).compute()
        return self.view(mask_arr)

    def sample(self, n: int, *, seed: int | None = None) -> CTable:
        """Return a read-only view of *n* randomly chosen live rows.

        Parameters
        ----------
        n:
            Number of rows to sample.  If *n* >= number of live rows,
            returns a view of the whole table.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        CTable
            A read-only view sharing columns with this table.
        """
        if n <= 0:
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))
        if n >= self._n_rows:
            return self.view(self._valid_rows)

        rng = np.random.default_rng(seed)
        all_pos = np.where(self._valid_rows[:])[0]
        chosen = rng.choice(all_pos, size=n, replace=False)

        mask = np.zeros(len(self._valid_rows), dtype=np.bool_)
        mask[chosen] = True
        return self.view(blosc2.asarray(mask))

    def select(self, cols: list[str]) -> CTable:
        """Return a column-projection view exposing only *cols*.

        The returned object shares the underlying NDArrays with this table
        (no data is copied).  Row filtering and value writes work as usual;
        structural mutations (add/drop/rename column, append, …) are blocked.

        Parameters
        ----------
        cols:
            Ordered list of column names to keep.

        Raises
        ------
        KeyError
            If any name in *cols* is not a column of this table.
        ValueError
            If *cols* is empty.
        """
        if not cols:
            raise ValueError("select() requires at least one column name.")
        for name in cols:
            if name not in self._cols and name not in self._computed_cols:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")

        obj = CTable.__new__(CTable)
        obj._row_type = self._row_type
        obj._validate = self._validate
        obj._table_cparams = self._table_cparams
        obj._table_dparams = self._table_dparams
        obj._storage = None
        obj._read_only = self._read_only
        obj._valid_rows = self._valid_rows
        obj._n_rows = self._n_rows
        obj._last_pos = self._last_pos
        obj.auto_compact = self.auto_compact
        obj.base = self

        # Stored columns — same NDArray objects, no copy
        obj._cols = {name: self._cols[name] for name in cols if name in self._cols}
        obj.col_names = list(cols)
        obj._materialized_cols = {
            name: dict(self._materialized_cols[name]) for name in cols if name in self._materialized_cols
        }
        obj._expr_index_arrays = self._expr_index_arrays

        # Computed columns — share the same definitions (LazyExpr refs remain valid)
        obj._computed_cols = {
            name: self._computed_cols[name] for name in cols if name in self._computed_cols
        }

        # Rebuild schema for the selected stored columns only
        stored_sel = [n for n in cols if n in self._cols]
        sel_set = set(stored_sel)
        sel_compiled = [c for c in self._schema.columns if c.name in sel_set]
        # Preserve caller-specified order
        order = {name: i for i, name in enumerate(stored_sel)}
        sel_compiled.sort(key=lambda c: order[c.name])
        obj._schema = CompiledSchema(
            columns=sel_compiled,
            columns_by_name={c.name: c for c in sel_compiled},
            row_cls=self._schema.row_cls,
        )
        obj._col_widths = {name: self._col_widths[name] for name in cols if name in self._col_widths}
        return obj

    def describe(self) -> None:
        """Print a per-column statistical summary.

        Numeric columns (int, float): count, mean, std, min, max.
        Bool columns: count, true-count, true-%.
        String columns: count, min (lex), max (lex), n-unique.
        """
        n = self._n_rows
        lines = []
        lines.append(f"CTable  {n:,} rows × {self.ncols} cols")
        lines.append("")

        for name in self.col_names:
            col = self[name]
            dtype = col.dtype
            spec = self._schema.columns_by_name.get(name)
            label = self._dtype_info_label(dtype, spec.spec if spec else None)
            lines.append(f"  {name}  [{label}]")

            if n == 0:
                lines.append("    (empty)")
                lines.append("")
                continue

            nc = col.null_count()
            n_nonnull = n - nc

            if isinstance(spec.spec, ListSpec) if spec is not None else False:
                lines.append(f"    count : {n:,}")
                lines.append("    (stats not available for list columns)")
            elif dtype.kind in "biufc" and dtype.kind != "c":
                # numeric + bool
                if dtype.kind == "b":
                    arr = col[:]
                    # Exclude null sentinels from true/false counts
                    if col.null_value is not None:
                        arr = arr[col.notnull()]
                    true_n = int(arr.sum())
                    lines.append(f"    count : {n:,}")
                    if nc > 0:
                        lines.append(f"    null  : {nc:,}  ({nc / n * 100:.1f} %)")
                    lines.append(f"    true  : {true_n:,}  ({true_n / n * 100:.1f} %)")
                    lines.append(f"    false : {n - true_n - nc:,}  ({(n - true_n - nc) / n * 100:.1f} %)")
                else:
                    fmt = ".4g"
                    lines.append(f"    count : {n:,}")
                    if nc > 0:
                        lines.append(f"    null  : {nc:,}  ({nc / n * 100:.1f} %)")
                    if n_nonnull > 0:
                        mn = col.min()
                        mx = col.max()
                        avg = col.mean()
                        sd = col.std()
                        lines.append(f"    mean  : {avg:{fmt}}")
                        lines.append(f"    std   : {sd:{fmt}}")
                        lines.append(f"    min   : {mn:{fmt}}")
                        lines.append(f"    max   : {mx:{fmt}}")
                    else:
                        lines.append("    (all values are null)")
            elif dtype.kind in "US":
                nu = len(col.unique())
                lines.append(f"    count   : {n:,}")
                if nc > 0:
                    lines.append(f"    null    : {nc:,}  ({nc / n * 100:.1f} %)")
                lines.append(f"    unique  : {nu:,}")
                if n_nonnull > 0:
                    mn = col.min()
                    mx = col.max()
                    lines.append(f"    min     : {str(mn)!r}")
                    lines.append(f"    max     : {str(mx)!r}")
                else:
                    lines.append("    (all values are null)")
            else:
                lines.append(f"    count : {n:,}")
                lines.append(f"    (stats not available for dtype {dtype})")

            lines.append("")

        print("\n".join(lines))

    def cov(self) -> np.ndarray:
        """Return the covariance matrix as a numpy array.

        Only int, float, and bool columns are supported.  Bool columns are
        cast to int (0/1) before computation.  Complex columns raise
        :exc:`TypeError`.

        Returns
        -------
        numpy.ndarray
            Shape ``(ncols, ncols)``.  Column order matches
            :attr:`col_names`.

        Raises
        ------
        TypeError
            If any column has an unsupported dtype (complex, string, …).
        ValueError
            If the table has fewer than 2 live rows (covariance undefined).
        """
        for name in self.col_names:
            dtype = self._col_dtype(name)
            if dtype is None or not (
                np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating) or dtype == np.bool_
            ):
                raise TypeError(
                    f"Column {name!r} has dtype {dtype} which is not supported by cov(). "
                    "Only int, float, and bool columns are allowed."
                )

        if self._n_rows < 2:
            raise ValueError(f"cov() requires at least 2 live rows, got {self._n_rows}.")

        # Build (n_cols, n_rows) matrix — one row per column.
        # Compute a combined null mask: any row that is null in *any* column
        # is excluded from all columns (listwise deletion).
        raw_arrays = []
        null_union = None
        for name in self.col_names:
            col = self[name]
            arr = col[:]
            nm = col._null_mask_for(arr)
            if nm.any():
                null_union = nm if null_union is None else (null_union | nm)
            raw_arrays.append(arr)

        arrays = []
        for arr in raw_arrays:
            if null_union is not None:
                arr = arr[~null_union]
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8)
            arrays.append(arr.astype(np.float64))

        n_valid = len(arrays[0]) if arrays else 0
        if n_valid < 2:
            raise ValueError(
                f"cov() requires at least 2 non-null rows, got {n_valid} after excluding nulls."
            )

        data = np.stack(arrays, axis=0)  # shape (ncols, n_valid)
        return np.atleast_2d(np.cov(data))

    # ------------------------------------------------------------------
    # Arrow interop
    # ------------------------------------------------------------------

    @staticmethod
    def _require_pyarrow(context: str):
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                f"pyarrow is required for {context}. Install it with: pip install pyarrow"
            ) from None
        return pa

    @staticmethod
    def _require_pyarrow_parquet(context: str):
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise ImportError(
                f"pyarrow is required for {context}. Install it with: pip install pyarrow"
            ) from None
        return pq

    @staticmethod
    def _validate_arrow_batch_size(batch_size: int) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

    def _resolve_arrow_columns(self, columns, include_computed: bool = True) -> list[str]:
        if columns is None:
            names = list(self.col_names)
            if not include_computed:
                names = [name for name in names if name not in self._computed_cols]
        else:
            names = list(columns)
        if len(set(names)) != len(names):
            raise ValueError("columns must be unique")
        for name in names:
            if name not in self.col_names:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
        return names

    @staticmethod
    def _pa_type_from_spec(pa, spec):
        if isinstance(spec, VLStringSpec):
            return pa.string()
        if isinstance(spec, VLBytesSpec):
            return pa.large_binary()
        if isinstance(spec, ListSpec):
            return pa.list_(CTable._pa_type_from_spec(pa, spec.item_spec))
        if isinstance(spec, StructSpec):
            return pa.struct(
                [pa.field(name, CTable._pa_type_from_spec(pa, child)) for name, child in spec.fields.items()]
            )
        if isinstance(spec, ObjectSpec):
            raise TypeError(
                "ObjectSpec columns do not have a fixed Arrow type; materialize values explicitly"
            )
        if spec.to_metadata_dict().get("kind") == "bool":
            return pa.bool_()
        dtype = getattr(spec, "dtype", None)
        if dtype is None:
            raise TypeError(f"No Arrow type for blosc2 spec {spec!r}")
        kind = dtype.kind
        if kind == "U":
            return pa.string()
        if kind == "S":
            return pa.large_binary()
        return pa.from_numpy_dtype(dtype)

    def _arrow_schema_for_columns(self, columns=None, *, include_computed: bool = True):
        pa = self._require_pyarrow("to_arrow()/to_parquet()")
        names = self._resolve_arrow_columns(columns, include_computed=include_computed)
        fields = []
        for name in names:
            cc = self._schema.columns_by_name.get(name)
            if cc is not None:
                pa_type = self._pa_type_from_spec(pa, cc.spec)
            else:
                pa_type = pa.from_numpy_dtype(np.asarray(self[name][:0]).dtype)
            fields.append(pa.field(name, pa_type))
        return pa.schema(fields)

    def iter_arrow_batches(
        self,
        *,
        columns: list[str] | None = None,
        batch_size: int = _BATCH_SIZE_DEFAULT,
        include_computed: bool = True,
    ):
        """Yield live rows as bounded-size :class:`pyarrow.RecordBatch` objects."""
        pa = self._require_pyarrow("iter_arrow_batches()")
        self._validate_arrow_batch_size(batch_size)
        self._flush_varlen_columns()
        names = self._resolve_arrow_columns(columns, include_computed=include_computed)

        for start in range(0, self._n_rows, batch_size):
            stop = min(start + batch_size, self._n_rows)
            arrays = []
            for name in names:
                col = self[name]
                if col.is_list:
                    spec = self._schema.columns_by_name[name].spec
                    arrays.append(pa.array(col[start:stop], type=self._pa_type_from_spec(pa, spec)))
                    continue
                if col.is_varlen_scalar:
                    spec = self._schema.columns_by_name[name].spec
                    values = col[start:stop]  # list of str/bytes/None
                    arrays.append(pa.array(values, type=self._pa_type_from_spec(pa, spec)))
                    continue
                arr = np.asarray(col[start:stop])
                nv = col.null_value
                null_mask = col._null_mask_for(arr) if nv is not None else None
                has_nulls = null_mask is not None and bool(null_mask.any())
                if arr.dtype.kind == "U":
                    values = arr.tolist()
                    if has_nulls:
                        values = [None if null_mask[i] else v for i, v in enumerate(values)]
                    arrays.append(pa.array(values, type=pa.string()))
                elif arr.dtype.kind == "S":
                    values = arr.tolist()
                    if has_nulls:
                        values = [None if null_mask[i] else v for i, v in enumerate(values)]
                    arrays.append(pa.array(values, type=pa.large_binary()))
                elif (
                    self._schema.columns_by_name.get(name) is not None
                    and self._schema.columns_by_name[name].spec.to_metadata_dict().get("kind") == "bool"
                ):
                    arrays.append(pa.array(arr == 1, mask=null_mask if has_nulls else None, type=pa.bool_()))
                else:
                    arrays.append(pa.array(arr, mask=null_mask if has_nulls else None))
            yield pa.RecordBatch.from_arrays(arrays, names=names)

    def to_arrow(self):
        """Convert all live rows to a :class:`pyarrow.Table`."""
        pa = self._require_pyarrow("to_arrow()")
        batches = list(self.iter_arrow_batches())
        schema = self._arrow_schema_for_columns()
        return pa.Table.from_batches(batches, schema=schema)

    @staticmethod
    def _auto_null_sentinel(pa, pa_type, *, null_policy: NullPolicy):
        return null_policy.sentinel_for_arrow_type(pa, pa_type)

    @staticmethod
    def _arrow_type_needs_object_fallback(pa, pa_type) -> bool:
        """True when *pa_type* has no typed CTable mapping."""
        if pa_type in (
            pa.int8(),
            pa.int16(),
            pa.int32(),
            pa.int64(),
            pa.uint8(),
            pa.uint16(),
            pa.uint32(),
            pa.uint64(),
            pa.float32(),
            pa.float64(),
            pa.bool_(),
            pa.string(),
            pa.large_string(),
            pa.utf8(),
            pa.large_utf8(),
        ):
            return False
        if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
            return False
        return not (
            pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type) or pa.types.is_struct(pa_type)
        )

    @staticmethod
    def _arrow_type_to_spec(  # noqa: C901
        pa,
        pa_type,
        arrow_col=None,
        *,
        string_max_length=None,
        null_value=None,
        nullable=False,
        object_fallback: bool = False,
    ):
        import blosc2.schema as b2s

        mapping = [
            (pa.int8(), b2s.int8),
            (pa.int16(), b2s.int16),
            (pa.int32(), b2s.int32),
            (pa.int64(), b2s.int64),
            (pa.uint8(), b2s.uint8),
            (pa.uint16(), b2s.uint16),
            (pa.uint32(), b2s.uint32),
            (pa.uint64(), b2s.uint64),
            (pa.float32(), b2s.float32),
            (pa.float64(), b2s.float64),
            (pa.bool_(), b2s.bool),
        ]
        for arrow_t, spec_cls in mapping:
            if pa_type == arrow_t:
                if null_value is not None and hasattr(spec_cls(), "null_value"):
                    return spec_cls(null_value=null_value)
                if null_value is not None and spec_cls is b2s.bool:
                    return spec_cls(null_value=null_value)
                return spec_cls()

        if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            if arrow_col is not None:
                py_values = arrow_col.to_pylist()
                flat_values = [item for cell in py_values if cell is not None for item in cell]
                item_arrow_col = pa.array(flat_values, type=pa_type.value_type)
                nullable = nullable or any(v is None for v in py_values)
            else:
                item_arrow_col = None
                nullable = True
            item_string_max_length = string_max_length
            if pa_type.value_type in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
                item_string_max_length = max(string_max_length or 1, 1_000_000)
            item_spec = CTable._arrow_type_to_spec(
                pa,
                pa_type.value_type,
                item_arrow_col,
                string_max_length=item_string_max_length,
                object_fallback=object_fallback,
            )
            return b2s.list(item_spec, nullable=nullable, storage="batch", serializer="msgpack")

        if pa.types.is_struct(pa_type):
            fields = {}
            for field in pa_type:
                child_col = None
                if arrow_col is not None:
                    combined = (
                        arrow_col.combine_chunks() if hasattr(arrow_col, "combine_chunks") else arrow_col
                    )
                    child_col = combined.field(field.name)
                child_string_max_length = string_max_length
                if field.type in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
                    child_string_max_length = max(string_max_length or 1, 1_000_000)
                fields[field.name] = CTable._arrow_type_to_spec(
                    pa,
                    field.type,
                    child_col,
                    string_max_length=child_string_max_length,
                    nullable=field.nullable,
                    object_fallback=object_fallback,
                )
            return b2s.struct(fields, nullable=nullable)

        if pa_type in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
            if string_max_length is None:
                # No fixed-width threshold given: store as variable-length scalar string.
                return b2s.vlstring(nullable=nullable)
            max_length = max(string_max_length, len(null_value) if null_value is not None else 1, 1)
            return b2s.string(max_length=max_length, null_value=null_value)

        if pa.types.is_binary(pa_type) or pa.types.is_large_binary(pa_type):
            if string_max_length is None:
                # No fixed-width threshold given: store as variable-length scalar bytes.
                return b2s.vlbytes(nullable=nullable)
            max_length = max(string_max_length, len(null_value) if null_value is not None else 1, 1)
            return b2s.bytes(max_length=max_length, null_value=null_value)

        if object_fallback:
            return b2s.object(nullable=nullable)

        raise TypeError(
            f"No blosc2 spec for Arrow type {pa_type!r}. Supported: int8/16/32/64, "
            "uint8/16/32/64, float32/64, bool, string, binary, list, and struct. "
            "Pass object_fallback=True to CTable.from_arrow() to import unsupported Arrow types "
            "as schema-less object columns."
        )

    @staticmethod
    def _string_max_length_for_column(string_max_length, name: str):
        if isinstance(string_max_length, Mapping):
            return string_max_length.get(name)
        return string_max_length

    @classmethod
    def _compiled_columns_from_arrow(
        cls,
        pa,
        schema,
        table_for_inference,
        string_max_length,
        *,
        auto_null_sentinels: bool,
        object_fallback: bool = False,
    ):
        null_policy = get_null_policy()
        column_null_values = null_policy.column_null_values
        schema_names = set(schema.names)
        unknown_null_values = set(column_null_values) - schema_names
        if unknown_null_values:
            names = ", ".join(sorted(unknown_null_values))
            raise KeyError(f"column_null_values contains unknown columns: {names}")
        columns: list[CompiledColumn] = []
        for field in schema:
            name = field.name
            _validate_column_name(name)
            arrow_col = table_for_inference.column(name) if table_for_inference is not None else None
            field_is_list = pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
            field_is_struct = pa.types.is_struct(field.type)
            column_string_max_length = cls._string_max_length_for_column(string_max_length, name)
            field_is_varlen_scalar = (
                not field_is_list
                and not field_is_struct
                and column_string_max_length is None
                and (
                    pa.types.is_string(field.type)
                    or pa.types.is_large_string(field.type)
                    or pa.types.is_binary(field.type)
                    or pa.types.is_large_binary(field.type)
                )
            )
            field_needs_object_fallback = cls._arrow_type_needs_object_fallback(pa, field.type)
            if field_needs_object_fallback and not object_fallback:
                cls._arrow_type_to_spec(pa, field.type, arrow_col, object_fallback=False)
            field_is_object_fallback = object_fallback and field_needs_object_fallback
            null_value = None
            has_null_value_override = name in column_null_values
            if has_null_value_override and (field_is_list or field_is_struct or field_is_object_fallback):
                raise TypeError(f"column_null_values only supports scalar columns; {name!r} is not scalar")
            if has_null_value_override and field_is_varlen_scalar:
                raise TypeError(
                    f"column_null_values is not supported for vlstring/vlbytes column {name!r}; "
                    "these columns represent nulls as native None."
                )
            if has_null_value_override:
                null_value = column_null_values[name]
            elif (
                auto_null_sentinels
                and field.nullable
                and not (
                    field_is_list or field_is_struct or field_is_varlen_scalar or field_is_object_fallback
                )
            ):
                null_value = cls._auto_null_sentinel(pa, field.type, null_policy=null_policy)
            if (
                arrow_col is not None
                and arrow_col.null_count
                and not (
                    field_is_list or field_is_struct or field_is_varlen_scalar or field_is_object_fallback
                )
                and null_value is None
            ):
                raise TypeError(
                    f"Column {name!r} contains Parquet nulls. Provide a CTable schema with a "
                    "null_value sentinel for this column."
                )
            spec = cls._arrow_type_to_spec(
                pa,
                field.type,
                arrow_col,
                string_max_length=column_string_max_length,
                null_value=null_value,
                nullable=field.nullable,
                object_fallback=object_fallback,
            )
            if null_value is not None and not (
                field_is_list or field_is_struct or field_is_varlen_scalar or field_is_object_fallback
            ):
                cls._validate_null_value_for_spec(name, spec, null_value)
            columns.append(cls._compiled_column_from_spec(name, spec))
        return columns

    @classmethod
    def _compiled_column_from_spec(cls, name: str, spec: SchemaSpec) -> CompiledColumn:
        col_config = ColumnConfig(cparams=None, dparams=None, chunks=None, blocks=None)
        return CompiledColumn(
            name=name,
            py_type=spec.python_type,
            spec=spec,
            dtype=getattr(spec, "dtype", None),
            default=MISSING,
            config=col_config,
            display_width=compute_display_width(spec),
        )

    @staticmethod
    def _storage_for_arrow_import(urlpath: str | None, mode: str) -> TableStorage:
        if urlpath is None:
            return InMemoryTableStorage()
        if mode == "w" and os.path.exists(urlpath):
            if os.path.isdir(urlpath):
                shutil.rmtree(urlpath)
            else:
                os.remove(urlpath)
        return FileTableStorage(urlpath, mode)

    @classmethod
    def _create_arrow_import_columns(
        cls, storage: TableStorage, columns: list[CompiledColumn], capacity: int, cparams, dparams
    ):
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        new_valid = storage.create_valid_rows(
            shape=(capacity,), chunks=default_chunks, blocks=default_blocks
        )
        new_cols: dict[str, blosc2.NDArray | ListArray | _ScalarVarLenArray] = {}
        for col in columns:
            if cls._is_list_column(col):
                new_cols[col.name] = storage.create_list_column(
                    col.name, spec=col.spec, cparams=cparams, dparams=dparams
                )
            elif cls._is_varlen_scalar_column(col):
                new_cols[col.name] = storage.create_varlen_scalar_column(
                    col.name, spec=col.spec, cparams=cparams, dparams=dparams
                )
            else:
                chunks, blocks = default_chunks, default_blocks
                if col.dtype is not None:
                    chunks, blocks = compute_chunks_blocks((capacity,), dtype=col.dtype)
                new_cols[col.name] = storage.create_column(
                    col.name,
                    dtype=col.dtype,
                    shape=(capacity,),
                    chunks=chunks,
                    blocks=blocks,
                    cparams=cparams,
                    dparams=dparams,
                )
        return new_cols, new_valid

    @classmethod
    def _new_arrow_import_ctable(
        cls, compiled, storage, new_cols, new_valid, columns, *, cparams, dparams, validate
    ):
        obj = cls.__new__(cls)
        obj._row_type = None
        obj._validate = validate
        obj._table_cparams = cparams
        obj._table_dparams = dparams
        obj._storage = storage
        obj._read_only = storage.is_read_only()
        obj._schema = compiled
        obj._cols = new_cols
        obj._col_widths = {col.name: max(len(col.name), col.display_width) for col in columns}
        obj.col_names = [col.name for col in columns]
        obj.auto_compact = False
        obj.base = None
        obj._computed_cols = {}
        obj._materialized_cols = {}
        obj._expr_index_arrays = {}
        obj._valid_rows = new_valid
        obj._n_rows = 0
        obj._last_pos = 0
        return obj

    @classmethod
    def _write_arrow_batches(cls, obj, batches, columns, new_cols, new_valid) -> None:
        pos = 0
        for batch in batches:
            end = pos + len(batch)
            while end > len(new_valid):
                obj._grow()
                new_valid = obj._valid_rows
            pos = cls._write_arrow_batch(batch, columns, new_cols, new_valid, pos)
        for col in columns:
            if cls._is_list_column(col) or cls._is_varlen_scalar_column(col):
                new_cols[col.name].flush()
        obj._n_rows = pos
        obj._last_pos = pos

    @classmethod
    def _write_arrow_batch(cls, batch, columns, new_cols, new_valid, pos: int) -> int:
        m = len(batch)
        if m == 0:
            return pos
        for col in columns:
            arrow_col = batch.column(batch.schema.get_field_index(col.name))
            if cls._is_list_column(col):
                # Trusted Arrow-import fast path: schema has already been inferred,
                # so avoid Python-level per-item coercion/validation here.
                new_cols[col.name].extend(arrow_col.to_pylist(), validate=False)
            elif cls._is_varlen_scalar_column(col):
                new_cols[col.name].extend(arrow_col.to_pylist())
            else:
                new_cols[col.name][pos : pos + m] = cls._arrow_column_to_numpy(arrow_col, col)
        new_valid[pos : pos + m] = True
        return pos + m

    @staticmethod
    def _arrow_column_to_numpy(arrow_col, col: CompiledColumn) -> np.ndarray:
        nv = getattr(col.spec, "null_value", None)
        if col.spec.to_metadata_dict().get("kind") == "bool" and col.dtype == np.dtype(np.uint8):
            return np.array([nv if v is None else int(v) for v in arrow_col.to_pylist()], dtype=np.uint8)
        if col.dtype.kind in "US":
            values = arrow_col.to_pylist()
            if nv is not None:
                values = [nv if v is None else v for v in values]
            max_len = col.spec.max_length
            too_long = [v for v in values if v is not None and len(v) > max_len]
            if too_long:
                raise ValueError(f"Column {col.name!r} contains values longer than max_length={max_len}.")
            return np.array(values, dtype=col.dtype)
        if arrow_col.null_count:
            if nv is None:
                raise TypeError(
                    f"Column {col.name!r} contains Arrow/Parquet nulls. Provide a CTable schema "
                    "with a null_value sentinel for this column."
                )
            arrow_col = arrow_col.fill_null(nv)
        return arrow_col.to_numpy(zero_copy_only=False).astype(col.dtype)

    @staticmethod
    def _arrow_schema_metadata(schema) -> dict[str, Any]:
        import base64

        try:
            schema_ipc = schema.serialize().to_pybytes()
            schema_ipc_base64 = base64.b64encode(schema_ipc).decode("ascii")
        except Exception:
            schema_ipc_base64 = None
        arrow_meta = {"schema_string": schema.to_string()}
        if schema_ipc_base64 is not None:
            arrow_meta["schema_ipc_base64"] = schema_ipc_base64
        return {"arrow": arrow_meta}

    @classmethod
    def from_arrow(
        cls,
        schema,
        batches,
        *,
        urlpath: str | None = None,
        mode: str = "w",
        cparams=None,
        dparams=None,
        validate: bool = False,
        capacity_hint: int | None = None,
        string_max_length: int | Mapping[str, int] | None = None,
        auto_null_sentinels: bool = True,
        blosc2_batch_size: int | None = _BATCH_SIZE_DEFAULT,
        blosc2_items_per_block: int | None = None,
        object_fallback: bool = False,
    ) -> CTable:
        """Build a :class:`CTable` from an Arrow schema and iterable of record batches.

        When *string_max_length* is ``None`` (the default), scalar Arrow
        ``string`` / ``large_string`` columns are imported as
        :func:`~blosc2.vlstring` columns and ``binary`` / ``large_binary``
        columns are imported as :func:`~blosc2.vlbytes` columns.  Arrow
        ``struct`` columns are imported as :func:`~blosc2.struct` columns backed
        by batched variable-length storage.  Null values for these variable-
        length scalar columns are represented as native ``None`` with no
        sentinel needed.

        When *string_max_length* is set to a positive integer, scalar string
        and binary columns are imported as fixed-width
        :func:`~blosc2.string` / :func:`~blosc2.bytes` columns whose dtype is
        sized to *string_max_length* characters/bytes. It may also be a mapping
        from column name to max length; omitted string/binary columns remain
        :func:`~blosc2.vlstring` / :func:`~blosc2.vlbytes` columns.

        ``blosc2_batch_size`` controls how many rows are buffered before
        BatchArray-backed imported columns (list columns and variable-length
        scalar columns such as ``vlstring``, ``vlbytes``, ``struct``, and
        schema-less ``object`` columns) are flushed to their backend.  Set it to
        ``None`` to keep those columns pending until the final flush.

        Unsupported Arrow types raise by default.  Pass ``object_fallback=True``
        to import such columns as schema-less :func:`~blosc2.object` columns.
        This fallback is intentionally not used by :meth:`from_parquet`.
        """
        pa = cls._require_pyarrow("from_arrow()")
        if blosc2_batch_size is not None and blosc2_batch_size <= 0:
            raise ValueError("blosc2_batch_size must be a positive integer or None")
        if blosc2_items_per_block is not None and blosc2_items_per_block <= 0:
            raise ValueError("blosc2_items_per_block must be a positive integer or None")
        batches = iter(batches)
        first_batch = None
        table_for_inference = None
        if string_max_length is None or isinstance(string_max_length, Mapping):
            first_batch = next(batches, None)
            if first_batch is not None:
                table_for_inference = pa.Table.from_batches([first_batch], schema=schema)
        columns = cls._compiled_columns_from_arrow(
            pa,
            schema,
            table_for_inference,
            string_max_length,
            auto_null_sentinels=auto_null_sentinels,
            object_fallback=object_fallback,
        )
        for col in columns:
            if (
                cls._is_list_column(col) and getattr(col.spec, "storage", None) == "batch"
            ) or cls._is_varlen_scalar_column(col):
                if blosc2_batch_size is not None:
                    col.spec.batch_rows = blosc2_batch_size
                if blosc2_items_per_block is not None:
                    col.spec.items_per_block = blosc2_items_per_block
        compiled = CompiledSchema(
            row_cls=None,
            columns=columns,
            columns_by_name={col.name: col for col in columns},
            metadata=cls._arrow_schema_metadata(schema),
        )
        if first_batch is not None:
            import itertools as _it

            batches = _it.chain([first_batch], batches)
        capacity = max(capacity_hint or 1, 1)
        storage = cls._storage_for_arrow_import(urlpath, mode)
        new_cols, new_valid = cls._create_arrow_import_columns(storage, columns, capacity, cparams, dparams)
        storage.save_schema(schema_to_dict(compiled))
        obj = cls._new_arrow_import_ctable(
            compiled,
            storage,
            new_cols,
            new_valid,
            columns,
            cparams=cparams,
            dparams=dparams,
            validate=validate,
        )
        cls._write_arrow_batches(obj, batches, columns, new_cols, new_valid)
        return obj

    def to_parquet(
        self,
        path,
        *,
        columns: list[str] | None = None,
        batch_size: int = _BATCH_SIZE_DEFAULT,
        compression: str | None = "zstd",
        row_group_size: int | None = None,
        include_computed: bool = True,
        **kwargs,
    ) -> None:
        """Write this table to a Parquet file batch-wise using pyarrow."""
        pq = self._require_pyarrow_parquet("to_parquet()")
        pa = self._require_pyarrow("to_parquet()")
        self._validate_arrow_batch_size(batch_size)
        schema = self._arrow_schema_for_columns(columns, include_computed=include_computed)
        with pq.ParquetWriter(path, schema, compression=compression, **kwargs) as writer:
            for batch in self.iter_arrow_batches(
                columns=columns, batch_size=batch_size, include_computed=include_computed
            ):
                table = pa.Table.from_batches([batch], schema=batch.schema)
                writer.write_table(table, row_group_size=row_group_size or len(batch))

    @classmethod
    def from_parquet(
        cls,
        path,
        *,
        columns: list[str] | None = None,
        batch_size: int = _BATCH_SIZE_DEFAULT,
        urlpath: str | None = None,
        mode: str = "w",
        cparams=None,
        dparams=None,
        validate: bool = False,
        auto_null_sentinels: bool = True,
        blosc2_batch_size: int | None = _BATCH_SIZE_DEFAULT,
        blosc2_items_per_block: int | None = None,
        **kwargs,
    ) -> CTable:
        """Read a Parquet file into a :class:`CTable`.

        The Parquet file is streamed batch by batch through :mod:`pyarrow` and then
        converted into a typed :class:`CTable`. By default, the result is created in
        memory, but you can also persist it on disk via ``urlpath``.

        This method delegates the actual table construction to
        :meth:`CTable.from_arrow`, so Arrow schema handling, nullable-column support,
        and Blosc2 write tuning follow the same rules as that method.  Top-level
        Arrow ``struct<...>`` columns are imported as :func:`~blosc2.struct`
        columns backed by batched variable-length storage.  Unsupported Parquet
        types are not silently imported as schema-less :func:`~blosc2.object`
        columns; they raise so callers can decide how to handle them explicitly.

        Parameters
        ----------
        path : str or path-like
            Path to the source Parquet file.

        columns : list[str] or None, optional
            Subset of columns to read from the Parquet file. If provided, only these
            columns are loaded and their order in the resulting table matches the
            order in this list. Column names must be unique.

        batch_size : int, optional
            Number of rows per Arrow batch read from the Parquet file. This controls
            how much data is pulled from the file at a time before being handed off
            to the CTable builder. Must be greater than 0.

        urlpath : str or None, optional
            Destination storage path for the resulting CTable. If ``None`` (the
            default), the table is created in memory. If provided, the table is backed
            by persistent on-disk storage.

        mode : str, optional
            Storage open mode for ``urlpath``. Defaults to ``"w"``. This is passed
            through to :meth:`CTable.from_arrow`.

        cparams : object, optional
            Compression parameters for the created Blosc2 containers. Passed through
            to :meth:`CTable.from_arrow`.

        dparams : object, optional
            Decompression parameters for the created Blosc2 containers. Passed through
            to :meth:`CTable.from_arrow`.

        validate : bool, optional
            Whether to enable extra internal validation while building the table.
            Defaults to ``False``.

        auto_null_sentinels : bool, optional
            If ``True`` (default), nullable scalar columns imported from Parquet may
            automatically receive per-column null sentinel values when needed. Sentinel
            selection follows the current null-policy rules used by CTable schema
            handling.

        blosc2_batch_size : int or None, optional
            Number of items written to Blosc2 containers per internal write batch.
            Passed through to :meth:`CTable.from_arrow`.

        blosc2_items_per_block : int or None, optional
            Target number of items per internal Blosc2 block. Passed through to
            :meth:`CTable.from_arrow`.

        **kwargs
            Additional keyword arguments forwarded to ``pyarrow.parquet.ParquetFile``.
            Use these for Parquet-reader-specific options supported by PyArrow.

        Returns
        -------
        CTable
            A new :class:`CTable` populated from the Parquet file. The table contains
            all selected columns and all rows from the file. If ``urlpath`` is
            provided, the returned table is disk-backed; otherwise it is in-memory.

        Raises
        ------
        ImportError
            If :mod:`pyarrow` is not installed.
        ValueError
            If ``batch_size`` is not greater than 0.
        ValueError
            If ``columns`` contains duplicate names.
        Exception
            Any exception raised by :mod:`pyarrow` while opening or reading the Parquet
            file, or by :meth:`CTable.from_arrow` while converting Arrow data into a
            CTable.

        Examples
        --------
        Load an entire Parquet file into an in-memory table:

        >>> import blosc2
        >>> t = blosc2.CTable.from_parquet("data.parquet")

        Load only a subset of columns:

        >>> t = blosc2.CTable.from_parquet(
        ...     "data.parquet",
        ...     columns=["user_id", "amount", "country"],
        ... )

        Create a disk-backed table while reading in batches:

        >>> t = blosc2.CTable.from_parquet(
        ...     "data.parquet",
        ...     batch_size=50_000,
        ...     urlpath="data.ctable",
        ... )

        Pass additional options through to PyArrow's Parquet reader:

        >>> t = blosc2.CTable.from_parquet(
        ...     "data.parquet",
        ...     memory_map=True,
        ... )
        """
        pq = cls._require_pyarrow_parquet("from_parquet()")
        pa = cls._require_pyarrow("from_parquet()")
        cls._validate_arrow_batch_size(batch_size)
        string_max_length = kwargs.pop("string_max_length", None)
        pf = pq.ParquetFile(path, **kwargs)
        arrow_schema = pf.schema_arrow
        if columns is not None:
            if len(set(columns)) != len(columns):
                raise ValueError("columns must be unique")
            fields = [arrow_schema.field(name) for name in columns]
            arrow_schema = pa.schema(fields)
        batches = pf.iter_batches(batch_size=batch_size, columns=columns)
        return cls.from_arrow(
            arrow_schema,
            batches,
            urlpath=urlpath,
            mode=mode,
            cparams=cparams,
            dparams=dparams,
            validate=validate,
            capacity_hint=pf.metadata.num_rows if pf.metadata is not None else None,
            string_max_length=string_max_length,
            auto_null_sentinels=auto_null_sentinels,
            blosc2_batch_size=blosc2_batch_size,
            blosc2_items_per_block=blosc2_items_per_block,
        )

    # ------------------------------------------------------------------
    # CSV interop
    # ------------------------------------------------------------------

    def to_csv(self, path: str, *, header: bool = True, sep: str = ",") -> None:
        """Write all live rows to a CSV file.

        Uses Python's stdlib ``csv`` module — no extra dependency required.
        Each column is materialised once via ``col[:]``; rows
        are then written one at a time.

        Parameters
        ----------
        path:
            Destination file path.  Created or overwritten.
        header:
            If ``True`` (default), write column names as the first row.
        sep:
            Field delimiter.  Defaults to ``","``; use ``"\\t"`` for TSV.
        """
        import csv

        arrays = [self[name][:] for name in self.col_names]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            if header:
                writer.writerow(self.col_names)
            for row in zip(*arrays, strict=True):
                writer.writerow(row)

    @staticmethod
    def _csv_col_to_array(raw: list[str], col, nv) -> np.ndarray:
        """Convert a list of raw CSV strings to a numpy array for *col*."""
        if col.dtype == np.bool_:

            def _parse(v, _nv=nv):
                stripped = v.strip()
                if stripped == "" and _nv is not None:
                    return _nv
                return stripped in ("True", "true", "1")

            return np.array([_parse(v) for v in raw], dtype=np.bool_)
        if col.dtype.kind == "S":
            prepared: list = [nv if (v.strip() == "" and nv is not None) else v.encode() for v in raw]
            return np.array(prepared, dtype=col.dtype)
        prepared2 = [nv if (v.strip() == "" and nv is not None) else v for v in raw]
        return np.array(prepared2, dtype=col.dtype)

    @classmethod
    def from_csv(
        cls,
        path: str,
        row_cls,
        *,
        header: bool = True,
        sep: str = ",",
    ) -> CTable:
        """Build a :class:`CTable` from a CSV file.

        Schema comes from *row_cls* (a dataclass) — CTable is always typed.
        All rows are read in a single pass into per-column Python lists, then
        each column is bulk-written into a pre-allocated NDArray (one slice
        assignment per column, no ``extend()``).

        Parameters
        ----------
        path:
            Source CSV file path.
        row_cls:
            A dataclass whose fields define the column names and types.
        header:
            If ``True`` (default), the first row is treated as a header and
            skipped.  Column order in the file must match *row_cls* field
            order regardless.
        sep:
            Field delimiter.  Defaults to ``","``; use ``"\\t"`` for TSV.

        Returns
        -------
        CTable
            A new in-memory CTable containing all rows from the CSV file.

        Raises
        ------
        TypeError
            If *row_cls* is not a dataclass.
        ValueError
            If a row has a different number of fields than the schema.
        """
        import csv

        schema = compile_schema(row_cls)
        cls._resolve_nullable_specs(schema)
        ncols = len(schema.columns)

        # Accumulate values per column as Python lists (one pass through file)
        col_data: list[list] = [[] for _ in range(ncols)]

        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=sep)
            if header:
                next(reader)
            for lineno, row in enumerate(reader, start=2 if header else 1):
                if len(row) != ncols:
                    raise ValueError(f"Line {lineno}: expected {ncols} fields, got {len(row)}.")
                for i, val in enumerate(row):
                    col_data[i].append(val)

        n = len(col_data[0]) if ncols > 0 else 0
        capacity = max(n, 1)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        mem_storage = InMemoryTableStorage()

        new_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        new_cols: dict[str, blosc2.NDArray] = {}
        for col in schema.columns:
            new_cols[col.name] = mem_storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=(capacity,),
                chunks=default_chunks,
                blocks=default_blocks,
                cparams=None,
                dparams=None,
            )

        obj = cls.__new__(cls)
        obj._row_type = row_cls
        obj._validate = True
        obj._table_cparams = None
        obj._table_dparams = None
        obj._storage = mem_storage
        obj._read_only = False
        obj._schema = schema
        obj._cols = new_cols
        obj._col_widths = {col.name: max(len(col.name), col.display_width) for col in schema.columns}
        obj.col_names = [col.name for col in schema.columns]
        obj.auto_compact = False
        obj.base = None
        obj._computed_cols = {}  # from_csv creates no computed columns
        obj._materialized_cols = {}
        obj._expr_index_arrays = {}
        obj._valid_rows = new_valid
        obj._n_rows = 0
        obj._last_pos = 0

        if n > 0:
            for i, col in enumerate(schema.columns):
                nv = getattr(col.spec, "null_value", None)
                arr = cls._csv_col_to_array(col_data[i], col, nv)
                new_cols[col.name][:n] = arr
            new_valid[:n] = True
            obj._n_rows = n
            obj._last_pos = n

        return obj

    # ------------------------------------------------------------------
    # Schema mutations: add / drop / rename columns
    # ------------------------------------------------------------------

    @staticmethod
    def _column_spec_default_and_config(
        spec_or_field: SchemaSpec | dataclasses.Field,
    ) -> tuple[SchemaSpec, Any, ColumnConfig]:
        """Extract the schema spec, default and storage config for ``add_column()``."""
        if isinstance(spec_or_field, dataclasses.Field):
            meta = get_blosc2_field_metadata(spec_or_field)
            if meta is None:
                raise TypeError("add_column() field descriptors must be created with blosc2.field().")
            spec = copy.deepcopy(meta["spec"])
            if spec_or_field.default is not MISSING:
                default = spec_or_field.default
            elif spec_or_field.default_factory is not MISSING:  # type: ignore[misc]
                default = spec_or_field.default_factory()
            else:
                default = MISSING
            config = ColumnConfig(
                cparams=meta.get("cparams"),
                dparams=meta.get("dparams"),
                chunks=meta.get("chunks"),
                blocks=meta.get("blocks"),
            )
        else:
            spec = spec_or_field
            default = MISSING
            config = ColumnConfig(cparams=None, dparams=None, chunks=None, blocks=None)

        if not isinstance(spec, SchemaSpec):
            raise TypeError(f"add_column() requires a SchemaSpec, got {type(spec)!r}.")
        return spec, default, config

    def add_column(
        self,
        name: str,
        spec: SchemaSpec | dataclasses.Field,
    ) -> None:
        """Add a new column filled from the default declared in *spec*.

        Parameters
        ----------
        name:
            Column name.  Must follow the same naming rules as schema fields.
        spec:
            A schema descriptor such as ``b2.int64(ge=0)`` or a field
            descriptor such as ``b2.field(b2.int64(ge=0), default=0)``.
            When the table already has live rows, use ``blosc2.field(...)``
            with a default declared so those rows can be backfilled.

        Raises
        ------
        ValueError
            If the table is read-only, is a view, the column already exists,
            or a non-empty table is given a column with no default declared.
        TypeError
            If a declared default cannot be coerced to *spec*'s dtype.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot add a column to a view.")
        _validate_column_name(name)
        if name in self._cols:
            raise ValueError(f"Column {name!r} already exists.")
        if name in self._computed_cols:
            raise ValueError(f"A computed column named {name!r} already exists.")

        spec, default, column_config = self._column_spec_default_and_config(spec)

        live_pos = np.where(self._valid_rows[:])[0]
        if default is MISSING and len(live_pos) > 0:
            raise ValueError(
                "add_column() requires a default declared as blosc2.field(..., default=...) "
                "when the table has live rows."
            )

        compiled_col = self._compiled_column_from_spec(name, spec)
        compiled_col.config = column_config
        self._resolve_nullable_specs(
            CompiledSchema(row_cls=None, columns=[compiled_col], columns_by_name={name: compiled_col}),
            validate_column_null_values=False,
        )
        spec = compiled_col.spec

        if self._is_varlen_scalar_column(compiled_col):
            # Varlen scalar columns don't use fixed-width NDArray storage.
            col_storage = self._resolve_column_storage(compiled_col, None, None)
            new_col = self._storage.create_varlen_scalar_column(
                name,
                spec=spec,
                cparams=col_storage.get("cparams"),
                dparams=col_storage.get("dparams"),
            )
            for _ in live_pos:
                new_col.append(default)
            new_col.flush()
        elif self._is_list_column(compiled_col):
            raise TypeError(
                "add_column() does not support list columns; use the constructor with a full schema."
            )
        else:
            if default is not MISSING:
                try:
                    default_val = spec.dtype.type(default)
                except (ValueError, OverflowError) as exc:
                    raise TypeError(
                        f"Cannot coerce default {default!r} to dtype {spec.dtype!r}: {exc}"
                    ) from exc
            else:
                default_val = None

            capacity = len(self._valid_rows)
            default_chunks, default_blocks = compute_chunks_blocks((capacity,))
            col_storage = self._resolve_column_storage(compiled_col, default_chunks, default_blocks)
            new_col = self._storage.create_column(
                name,
                dtype=spec.dtype,
                shape=(capacity,),
                chunks=col_storage["chunks"],
                blocks=col_storage["blocks"],
                cparams=col_storage.get("cparams"),
                dparams=col_storage.get("dparams"),
            )
            if len(live_pos) > 0:
                new_col[live_pos] = default_val

        compiled_col.default = default
        self._cols[name] = new_col
        self.col_names.append(name)
        self._col_widths[name] = max(len(name), compiled_col.display_width)

        new_columns = self._schema.columns + [compiled_col]
        self._schema = CompiledSchema(
            row_cls=self._schema.row_cls,
            columns=new_columns,
            columns_by_name={**self._schema.columns_by_name, name: compiled_col},
        )
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def drop_column(self, name: str) -> None:
        """Remove a column from the table.

        On disk tables the corresponding persisted column leaf is deleted.

        Raises
        ------
        ValueError
            If the table is read-only, is a view, or *name* is the last column.
        KeyError
            If *name* does not exist.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot drop a column from a view.")
        if name not in self._cols:
            raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
        if len(self._stored_col_names) == 1:
            raise ValueError("Cannot drop the last column.")
        # Guard: refuse if any computed column depends on this column
        dependents = [cc_name for cc_name, cc in self._computed_cols.items() if name in cc["col_deps"]]
        if dependents:
            raise ValueError(
                f"Cannot drop column {name!r}: it is used by computed column(s) "
                + ", ".join(repr(d) for d in dependents)
                + ". Drop those computed columns first."
            )

        catalog = self._storage.load_index_catalog()
        if name in catalog:
            descriptor = catalog.pop(name)
            self._validate_index_descriptor(name, descriptor)
            self._drop_index_descriptor(name, descriptor)
            self._storage.save_index_catalog(catalog)

        if isinstance(self._storage, FileTableStorage):
            self._storage.delete_column(name)

        self._materialized_cols.pop(name, None)
        del self._cols[name]
        del self._col_widths[name]
        self.col_names.remove(name)

        new_columns = [c for c in self._schema.columns if c.name != name]
        self._schema = CompiledSchema(
            row_cls=self._schema.row_cls,
            columns=new_columns,
            columns_by_name={c.name: c for c in new_columns},
        )
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def rename_column(self, old: str, new: str) -> None:
        """Rename a column.

        On disk tables the corresponding persisted column leaf is renamed.

        Raises
        ------
        ValueError
            If the table is read-only, is a view, or *new* already exists.
        KeyError
            If *old* does not exist.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot rename a column in a view.")
        if old not in self._cols and old not in self._computed_cols:
            raise KeyError(f"No column named {old!r}. Available: {self.col_names}")
        if new in self._cols or new in self._computed_cols:
            raise ValueError(f"Column {new!r} already exists.")
        _validate_column_name(new)

        # Computed columns have no physical storage or schema entry.  Renaming
        # them only updates the computed-column registry and visible names.
        if old in self._computed_cols:
            self._computed_cols[new] = self._computed_cols.pop(old)
            idx = self.col_names.index(old)
            self.col_names[idx] = new
            self._col_widths[new] = max(len(new), self._col_widths.pop(old))
            if isinstance(self._storage, FileTableStorage):
                self._storage.save_schema(self._schema_dict_with_computed())
            return

        # Guard: refuse rename if any computed column depends on this stored column
        dependents = [cc_name for cc_name, cc in self._computed_cols.items() if old in cc["col_deps"]]
        if dependents:
            raise ValueError(
                f"Cannot rename column {old!r}: it is used by computed column(s) "
                + ", ".join(repr(d) for d in dependents)
                + ". Drop those computed columns first."
            )

        catalog = self._storage.load_index_catalog()
        rebuild_kwargs = None
        if old in catalog:
            descriptor = catalog.pop(old)
            self._validate_index_descriptor(old, descriptor)
            rebuild_kwargs = self._index_create_kwargs_from_descriptor(descriptor)
            self._drop_index_descriptor(old, descriptor)
            self._storage.save_index_catalog(catalog)

        if isinstance(self._storage, FileTableStorage):
            self._cols[new] = self._storage.rename_column(old, new)
        else:
            self._cols[new] = self._cols[old]
        del self._cols[old]

        idx = self.col_names.index(old)
        self.col_names[idx] = new
        self._col_widths[new] = max(len(new), self._col_widths.pop(old))

        old_compiled = self._schema.columns_by_name[old]
        renamed = CompiledColumn(
            name=new,
            py_type=old_compiled.py_type,
            spec=old_compiled.spec,
            dtype=old_compiled.dtype,
            default=old_compiled.default,
            config=old_compiled.config,
            display_width=old_compiled.display_width,
        )
        new_columns = [renamed if c.name == old else c for c in self._schema.columns]
        self._schema = CompiledSchema(
            row_cls=self._schema.row_cls,
            columns=new_columns,
            columns_by_name={c.name: c for c in new_columns},
        )
        if old in self._materialized_cols:
            self._materialized_cols[new] = self._materialized_cols.pop(old)
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())
        if rebuild_kwargs is not None:
            self.create_index(new, **rebuild_kwargs)

    # ------------------------------------------------------------------
    # Computed / virtual columns
    # ------------------------------------------------------------------

    @property
    def _stored_col_names(self) -> list[str]:
        """Column names backed by physical NDArrays (excludes computed columns)."""
        return [n for n in self.col_names if n not in self._computed_cols]

    @property
    def _append_input_col_names(self) -> list[str]:
        """Stored columns that callers must normally provide on insert."""
        return [n for n in self._stored_col_names if n not in self._materialized_cols]

    @property
    def computed_columns(self) -> dict[str, dict]:
        """Read-only view of the computed-column definitions.

        Each value is a dict with keys ``expression``, ``col_deps``,
        ``lazy`` (:class:`blosc2.LazyExpr`), and ``dtype``.
        """
        return dict(self._computed_cols)  # shallow copy so callers can't mutate

    def _col_dtype(self, name: str) -> np.dtype | None:
        """Return the dtype for *name*, routing through computed cols."""
        cc = self._computed_cols.get(name)
        if cc is not None:
            return cc["dtype"]
        return getattr(self._cols[name], "dtype", None)

    @staticmethod
    def _readable_computed_expr(cc: dict) -> str:
        """Return the expression string with ``o0``, ``o1``, … replaced by
        their actual column names, for human-readable display.

        Example: ``"(o0 * o1)"`` with ``col_deps=["price", "qty"]``
        becomes ``"(price * qty)"``.
        """
        col_deps = cc["col_deps"]

        def _sub(m: re.Match) -> str:
            idx = int(m.group(1))
            return col_deps[idx] if idx < len(col_deps) else m.group(0)

        return re.sub(r"\bo(\d+)\b", _sub, cc["expression"])

    def _fetch_col_at_positions(self, name: str, positions: np.ndarray):
        """Fetch values at *positions* (physical indices) — used for display."""
        cc = self._computed_cols.get(name)
        if cc is not None:
            if len(positions) == 0:
                return np.array([], dtype=cc["dtype"])
            return np.array(
                [np.asarray(cc["lazy"][int(p)]).ravel()[0] for p in positions],
                dtype=cc["dtype"],
            )
        col = self._cols[name]
        spec = self._schema.columns_by_name[name].spec
        if self._is_list_spec(spec) or isinstance(spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec)):
            return col[positions]
        return col[positions]

    def _schema_dict_with_computed(self) -> dict:
        """Return the schema dict extended with computed/materialized metadata."""
        d = schema_to_dict(self._schema)
        if self._computed_cols:
            d["computed_columns"] = [
                {
                    "name": name,
                    "expression": cc["expression"],
                    "col_deps": cc["col_deps"],
                    "dtype": str(cc["dtype"]),
                }
                for name, cc in self._computed_cols.items()
            ]
        if self._materialized_cols:
            d["materialized_columns"] = [
                {
                    "name": name,
                    "computed_column": meta["computed_column"],
                    "expression": meta["expression"],
                    "col_deps": meta["col_deps"],
                    "dtype": str(meta["dtype"]),
                }
                for name, meta in self._materialized_cols.items()
            ]
        return d

    def _load_computed_cols_from_schema(self, schema_dict: dict) -> None:
        """Reconstruct ``_computed_cols`` from persisted metadata.

        Called from ``__init__``, ``open``, and ``load`` after all stored
        columns have been opened into ``self._cols``.
        """
        for cc_meta in schema_dict.get("computed_columns", []):
            name = cc_meta["name"]
            expression = cc_meta["expression"]
            col_deps = cc_meta["col_deps"]
            dtype = np.dtype(cc_meta["dtype"])
            operands = {f"o{i}": self._cols[dep] for i, dep in enumerate(col_deps)}
            lazy = blosc2.lazyexpr(expression, operands)
            self._computed_cols[name] = {
                "expression": expression,
                "col_deps": col_deps,
                "lazy": lazy,
                "dtype": dtype,
            }
            self.col_names.append(name)
            self._col_widths[name] = max(len(name), 15)

    def _load_materialized_cols_from_schema(self, schema_dict: dict) -> None:
        """Reconstruct ``_materialized_cols`` from persisted metadata."""
        for meta in schema_dict.get("materialized_columns", []):
            self._materialized_cols[meta["name"]] = {
                "computed_column": meta["computed_column"],
                "expression": meta["expression"],
                "col_deps": list(meta["col_deps"]),
                "dtype": np.dtype(meta["dtype"]),
            }

    def _require_computed_column(self, name: str) -> dict:
        """Return metadata for computed column *name* or raise ``KeyError``."""
        try:
            return self._computed_cols[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a computed column. Computed columns: {list(self._computed_cols)}"
            ) from None

    def _autofill_materialized_row_values(self, row: dict[str, Any]) -> dict[str, Any]:
        """Fill omitted materialized-column values for a single inserted row."""
        row = dict(row)
        for name, meta in self._materialized_cols.items():
            if name in row:
                continue
            missing = [dep for dep in meta["col_deps"] if dep not in row]
            if missing:
                raise ValueError(
                    f"Cannot auto-fill materialized column {name!r}: missing dependency columns {missing!r}."
                )
            operands = {f"o{i}": np.asarray([row[dep]]) for i, dep in enumerate(meta["col_deps"])}
            values = blosc2.lazyexpr(meta["expression"], operands)[:]
            row[name] = np.asarray(values, dtype=meta["dtype"])[0]
        return row

    def _validate_no_default_columns_present(self, row: dict[str, Any]) -> None:
        """Raise a clear error when a row omits a column with no default declared."""
        for col in self._schema.columns:
            if col.name in row:
                continue
            is_nullable = getattr(col.spec, "null_value", None) is not None or bool(
                getattr(col.spec, "nullable", False)
            )
            if col.default is MISSING and not is_nullable:
                raise ValueError(f"Column {col.name!r} has no default declared; a value must be provided.")

    def _fill_default_batch_columns(self, raw_columns: dict[str, Any], row_count: int) -> dict[str, Any]:
        """Fill omitted batch columns from defaults, or raise if no default is declared."""
        raw_columns = dict(raw_columns)
        for col in self._schema.columns:
            if col.name in raw_columns:
                continue
            if col.default is MISSING:
                raise ValueError(f"Column {col.name!r} has no default declared; values must be provided.")
            raw_columns[col.name] = [col.default] * row_count
        return raw_columns

    def _autofill_materialized_batch_columns(
        self, raw_columns: dict[str, Any], row_count: int, *, provided_names: set[str]
    ) -> dict[str, Any]:
        """Fill omitted materialized-column arrays for batch inserts."""
        raw_columns = dict(raw_columns)
        for name, meta in self._materialized_cols.items():
            if name in provided_names or name in raw_columns:
                continue
            missing = [dep for dep in meta["col_deps"] if dep not in raw_columns]
            if missing:
                raise ValueError(
                    f"Cannot auto-fill materialized column {name!r}: missing dependency columns {missing!r}."
                )
            operands = {
                f"o{i}": blosc2.asarray(raw_columns[dep], dtype=self._cols[dep].dtype)
                for i, dep in enumerate(meta["col_deps"])
            }
            values = blosc2.lazyexpr(meta["expression"], operands)[:]
            values = np.asarray(values, dtype=meta["dtype"])
            if len(values) != row_count:
                raise ValueError(
                    f"Materialized column {name!r} produced {len(values)} values, expected {row_count}."
                )
            raw_columns[name] = values
        return raw_columns

    @staticmethod
    def _schema_spec_from_dtype(dtype: np.dtype) -> SchemaSpec:
        """Build a minimal schema spec for a stored column with *dtype*."""
        dtype = np.dtype(dtype)
        spec_factory = _DTYPE_SPEC_FACTORIES.get(dtype)
        if spec_factory is not None:
            return spec_factory()
        if dtype.kind == "U":
            max_length = max(1, dtype.itemsize // np.dtype("U1").itemsize)
            return string(max_length=max_length)
        if dtype.kind == "S":
            return b2_bytes(max_length=max(1, dtype.itemsize))
        raise TypeError(f"Cannot materialize a computed column with unsupported dtype {dtype!r}.")

    def _create_empty_stored_column(
        self,
        name: str,
        dtype: np.dtype,
        *,
        cparams: dict | None = None,
    ) -> None:
        """Create an empty stored column aligned with the table's physical row space."""
        spec = self._schema_spec_from_dtype(dtype)
        default = np.array(0, dtype=dtype).item() if dtype.kind not in {"U", "S"} else dtype.type()

        capacity = len(self._valid_rows)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        new_col = self._storage.create_column(
            name,
            dtype=dtype,
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
            cparams=cparams,
            dparams=None,
        )

        compiled_col = CompiledColumn(
            name=name,
            py_type=spec.python_type,
            spec=spec,
            dtype=dtype,
            default=default,
            config=ColumnConfig(cparams=cparams, dparams=None, chunks=None, blocks=None),
            display_width=compute_display_width(spec),
        )
        self._cols[name] = new_col
        self.col_names.append(name)
        self._col_widths[name] = max(len(name), compiled_col.display_width)

        new_columns = self._schema.columns + [compiled_col]
        self._schema = CompiledSchema(
            row_cls=self._schema.row_cls,
            columns=new_columns,
            columns_by_name={**self._schema.columns_by_name, name: compiled_col},
        )
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def _fill_stored_column_from_computed(
        self,
        target_name: str,
        computed_name: str,
        *,
        dtype: np.dtype,
    ) -> None:
        """Evaluate computed column *computed_name* into stored column *target_name*."""
        cc = self._require_computed_column(computed_name)
        operands = {f"o{i}": self._cols[dep] for i, dep in enumerate(cc["col_deps"])}
        lazy = blosc2.lazyexpr(cc["expression"], operands)
        capacity = len(self._valid_rows)
        step = int(self._valid_rows.chunks[0]) if self._valid_rows.chunks else 65536

        for start in range(0, capacity, step):
            stop = min(start + step, capacity)
            values = lazy[start:stop]
            if isinstance(values, blosc2.NDArray):
                values = values[:]
            try:
                values = np.asarray(values, dtype=dtype)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Cannot coerce computed values to dtype {dtype!r}: {exc}") from exc
            if values.ndim != 1:
                raise TypeError(
                    f"Computed column {computed_name!r} produced {values.ndim}-D values; expected 1-D slices."
                )
            if len(values) != stop - start:
                raise ValueError(
                    f"Computed column {computed_name!r} produced {len(values)} values for slice "
                    f"[{start}:{stop}], expected {stop - start}."
                )
            self._cols[target_name][start:stop] = values

    def materialize_computed_column(
        self,
        name: str,
        *,
        new_name: str | None = None,
        dtype: np.dtype | None = None,
        cparams: dict | blosc2.CParams | None = None,
    ) -> None:
        """Materialize a computed column into a new stored snapshot column.

        Parameters
        ----------
        name:
            Existing computed column to materialize.
        new_name:
            Name of the new stored column. Defaults to ``f"{name}_stored"``.
        dtype:
            Optional target dtype for the stored column. Defaults to the
            computed column dtype.
        cparams:
            Optional compression parameters for the new stored column.

        Raises
        ------
        ValueError
            If called on a view, on a read-only table, or if the target name
            collides with an existing stored or computed column.
        KeyError
            If *name* is not a computed column.
        TypeError
            If *dtype* is incompatible with the computed values.
        """
        if self.base is not None:
            raise ValueError("Cannot materialize a computed column from a view.")
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")

        cc = self._require_computed_column(name)
        target_name = new_name or f"{name}_stored"
        _validate_column_name(target_name)
        if target_name in self._cols:
            raise ValueError(f"A stored column named {target_name!r} already exists.")
        if target_name in self._computed_cols:
            raise ValueError(f"A computed column named {target_name!r} already exists.")
        target_dtype = np.dtype(dtype) if dtype is not None else np.dtype(cc["dtype"])

        self._create_empty_stored_column(target_name, target_dtype, cparams=cparams)
        self._materialized_cols[target_name] = {
            "computed_column": name,
            "expression": cc["expression"],
            "col_deps": list(cc["col_deps"]),
            "dtype": target_dtype,
        }
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())
        try:
            self._fill_stored_column_from_computed(target_name, name, dtype=target_dtype)
        except Exception:
            with contextlib.suppress(Exception):
                self.drop_column(target_name)
            raise

    def add_computed_column(
        self,
        name: str,
        expr,
        *,
        dtype: np.dtype | None = None,
    ) -> None:
        """Add a read-only virtual column whose values are computed from other columns.

        The column stores no data — it is evaluated on-the-fly when read.
        It participates in display, filtering, sorting, export (to_arrow / to_csv),
        and aggregates, but cannot be written to, indexed, or included in
        ``append`` / ``extend`` inputs.

        Parameters
        ----------
        name:
            Column name.  Must not collide with any existing stored or computed
            column and must satisfy the usual naming rules.
        expr:
            Either a **callable** ``(cols: dict[str, NDArray]) -> LazyExpr``
            or an **expression string** (e.g. ``"price * qty"``) where column
            names are referenced directly and resolved from stored columns.
        dtype:
            Override the inferred result dtype.  When omitted the dtype is
            taken from the :class:`blosc2.LazyExpr`.

        Raises
        ------
        ValueError
            If called on a view, the table is read-only, *name* already
            exists, or an operand is not a stored column of this table.
        TypeError
            If *expr* is not a callable or string, or does not return a
            :class:`blosc2.LazyExpr`.
        """
        if self.base is not None:
            raise ValueError("Cannot add a computed column to a view.")
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        _validate_column_name(name)
        if name in self._cols:
            raise ValueError(f"A stored column named {name!r} already exists.")
        if name in self._computed_cols:
            raise ValueError(f"A computed column named {name!r} already exists.")

        # Build the LazyExpr
        if callable(expr):
            lazy = expr(self._cols)
        elif isinstance(expr, str):
            lazy = blosc2.lazyexpr(expr, self._cols)
        else:
            raise TypeError(f"expr must be a callable or an expression string, got {type(expr).__name__!r}.")
        if not isinstance(lazy, blosc2.LazyExpr):
            raise TypeError(f"expr must return a blosc2.LazyExpr, got {type(lazy).__name__!r}.")

        # Verify all operands are stored columns of *this* table and record their names
        owned_ids = {id(arr): cname for cname, arr in self._cols.items()}
        sorted_keys = sorted(lazy.operands.keys())  # ["o0", "o1", ...]
        col_deps = []
        for key in sorted_keys:
            arr = lazy.operands[key]
            cname = owned_ids.get(id(arr))
            if cname is None:
                raise ValueError(
                    f"Operand {key!r} in the expression does not reference a stored "
                    f"column of this table.  Only stored columns may be used as "
                    f"dependencies (for v1 computed columns cannot depend on each other)."
                )
            col_deps.append(cname)

        result_dtype = np.dtype(dtype) if dtype is not None else lazy.dtype

        self._computed_cols[name] = {
            "expression": lazy.expression,
            "col_deps": col_deps,
            "lazy": lazy,
            "dtype": result_dtype,
        }
        self.col_names.append(name)
        self._col_widths[name] = max(len(name), 15)

        # Persist metadata if backed by a file store
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def drop_computed_column(self, name: str) -> None:
        """Remove a computed column from the table.

        Parameters
        ----------
        name:
            Name of the computed column to remove.

        Raises
        ------
        KeyError
            If *name* is not a computed column.
        ValueError
            If called on a view.
        """
        if self.base is not None:
            raise ValueError("Cannot drop a computed column from a view.")
        if name not in self._computed_cols:
            raise KeyError(
                f"{name!r} is not a computed column. Computed columns: {list(self._computed_cols)}"
            )
        del self._computed_cols[name]
        self.col_names.remove(name)
        self._col_widths.pop(name, None)

        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    # ------------------------------------------------------------------
    # Column / row access
    # ------------------------------------------------------------------

    @staticmethod
    def _all_strings(seq) -> bool:
        return all(isinstance(v, str) for v in seq)

    @staticmethod
    def _all_ints(seq) -> bool:
        return all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in seq)

    def _getitem_arraylike(self, key):
        if len(key) == 0:
            return self._run_row_logic(key)
        if getattr(key, "dtype", None) is not None:
            if key.dtype == np.bool_:
                return self._run_row_logic(key)
            if np.issubdtype(key.dtype, np.integer):
                return self._run_row_logic(key)
            if key.dtype.kind in {"U", "S"}:
                return self.select(key.tolist())
        values = key.tolist() if hasattr(key, "tolist") else list(key)
        if self._all_strings(values):
            return self.select(values)
        return self._run_row_logic(key)

    def _getitem_row_selector(self, key):
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            return self._materialize_row(int(key))
        if isinstance(key, slice):
            return self._run_row_logic(key)
        if isinstance(key, np.ndarray):
            return self._getitem_arraylike(key)
        if isinstance(key, list):
            if key and self._all_strings(key):
                return self.select(key)
            return self._run_row_logic(key)
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes, tuple)):
            key = list(key)
            if key and self._all_strings(key):
                return self.select(key)
            return self._run_row_logic(key)
        raise TypeError(
            "Row selectors must be an int, slice, integer array/list, or boolean mask; "
            f"got {type(key).__name__}"
        )

    def _structured_array_dtype(self) -> np.dtype:
        fields = []
        for name in self.col_names:
            col_info = self._schema.columns_by_name.get(name)
            if col_info is None:
                dtype = np.asarray(self[name][:0]).dtype
            elif self._is_list_column(col_info) or self._is_varlen_scalar_column(col_info):
                dtype = np.dtype(object)
            else:
                dtype = col_info.dtype if col_info.dtype is not None else np.dtype(object)
            fields.append((name, dtype))
        return np.dtype(fields)

    def __array__(self, dtype=None, copy=None):
        arr = np.empty(self.nrows, dtype=self._structured_array_dtype())
        for name in self.col_names:
            values = self[name][:]
            target_dtype = arr.dtype.fields[name][0]
            if target_dtype == np.dtype(object) and isinstance(values, np.ndarray):
                values = values.tolist()
            arr[name] = values
        if dtype is not None:
            arr = arr.astype(dtype, copy=True if copy is None else copy)
        return arr.copy() if copy else arr

    def __getitem__(self, key):
        """Type-driven indexing for columns, rows, projections, and filters.

        Supported keys are:

        - ``str``: return a :class:`Column` when it matches a stored or computed
          column name; otherwise evaluate it as a boolean expression via
          :meth:`where`.
        - boolean :class:`blosc2.LazyExpr` or :class:`blosc2.NDArray`: return the
          same filtered view as :meth:`where`, e.g. ``t[t.temperature_f > 70]``.
        - ``int``: return one live row as a namedtuple-like object.
        - ``slice``: return a row-range view.
        - integer array/list: return a gathered-row view.
        - boolean NumPy array/list: return a boolean-mask filtered view.
        - string list: return a column-projection view, equivalent to
          :meth:`select`.

        Examples
        --------
        Access columns and rows::

            temps = t["temperature"]
            first = t[0]
            view = t[10:20]

        Filter rows with a string expression, a stored-column expression, or a
        computed-column expression::

            warm = t["temperature > 20"]
            warm_active = t[(t.temperature > 20) & t.active]
            hot_fahrenheit = t[t.temperature_f > 70]

        Project columns::

            slim = t[["sensor_id", "temperature_f"]]
        """
        if isinstance(key, str):
            if key in self._cols or key in self._computed_cols:
                return Column(self, key)
            return self.where(key)
        if isinstance(key, (blosc2.NDArray, blosc2.LazyExpr)) and getattr(key, "dtype", None) == np.bool_:
            return self.where(key)
        if isinstance(key, tuple):
            raise TypeError("Tuple indexing is not supported for CTable in V1")
        return self._getitem_row_selector(key)

    def __getattr__(self, s: str):
        if s in self._cols or s in self._computed_cols:
            return Column(self, s)
        return super().__getattribute__(s)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self):
        """Physically rewrite every column array keeping only live rows.

        Closes the gaps left by prior :meth:`delete` calls.  All existing
        indexes are dropped and must be recreated afterwards.  Raises
        ``ValueError`` if the table is read-only or a view.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot compact a view.")
        self._flush_varlen_columns()
        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        for col in self._schema.columns:
            name = col.name
            v = self._cols[name]
            if self._is_list_column(col):
                compacted = [v[int(pos)] for pos in real_poss[: self._n_rows]]
                replacement = ListArray(spec=col.spec)
                replacement.extend(compacted)
                replacement.flush()
                self._cols[name] = replacement
                continue
            if self._is_varlen_scalar_column(col):
                compacted = [v[int(pos)] for pos in real_poss[: self._n_rows]]
                replacement = _ScalarVarLenArray(col.spec)
                replacement.extend(compacted)
                replacement.flush()
                self._cols[name] = replacement
                continue
            start = 0
            block_size = self._valid_rows.blocks[0]
            end = min(block_size, self._n_rows)
            while start < end:
                v[start:end] = v[real_poss[start:end]]
                start += block_size
                end = min(end + block_size, self._n_rows)

        self._valid_rows[: self._n_rows] = True
        self._valid_rows[self._n_rows :] = False
        self._last_pos = self._n_rows
        self._mark_all_indexes_stale()

    def _normalise_sort_keys(
        self,
        cols: str | list[str],
        ascending: bool | list[bool],
    ) -> tuple[list[str], list[bool]]:
        """Validate and normalise sort key arguments; return (cols, ascending)."""
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(ascending, bool):
            ascending = [ascending] * len(cols)
        if len(cols) != len(ascending):
            raise ValueError(
                f"'ascending' must have the same length as 'cols' ({len(cols)}), got {len(ascending)}."
            )
        for name in cols:
            if name not in self._cols and name not in self._computed_cols:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
            dtype = self._col_dtype(name)
            if dtype is None:
                cc = self._schema.columns_by_name.get(name)
                if cc is not None and isinstance(
                    cc.spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec)
                ):
                    raise TypeError(
                        f"Column {name!r} is a varlen scalar column and does not support sort ordering."
                    )
                raise TypeError(
                    f"Column {name!r} is a list column and does not support sort ordering in V1."
                )
            if np.issubdtype(dtype, np.complexfloating):
                raise TypeError(
                    f"Column {name!r} has complex dtype {dtype} which does not support ordering."
                )
        return cols, ascending

    def _sorted_positions_from_full_index(self, name: str, ascending: bool) -> np.ndarray | None:
        """Return live physical positions from a matching FULL index, if available.

        Reads the pre-sorted positions sidecar directly rather than going through
        the ordered_indices query machinery, which is optimised for selective range
        queries and is much slower for full-table streaming.
        """
        root = self._root_table
        catalog = root._storage.load_index_catalog()
        descriptor = None

        if name in root._cols:
            col_info = root._schema.columns_by_name.get(name)
            if col_info is not None and getattr(col_info.spec, "null_value", None) is not None:
                return None
            descriptor = catalog.get(name)
            if descriptor is None or descriptor.get("kind") != "full" or descriptor.get("stale", False):
                descriptor = None
        elif name in root._computed_cols:
            cc = root._computed_cols[name]
            for _lookup_key, candidate in catalog.items():
                target = candidate.get("target") or {}
                if (
                    target.get("source") == "expression"
                    and candidate.get("kind") == "full"
                    and not candidate.get("stale", False)
                    and target.get("expression_key") == cc["expression"]
                    and list(target.get("dependencies", [])) == list(cc["col_deps"])
                ):
                    descriptor = candidate
                    break
        if descriptor is None:
            return None

        positions_path = descriptor.get("full", {}).get("positions_path")

        # Read pre-sorted positions directly — bypasses the ordered_indices query
        # machinery which is built for selective range queries and is ~70x slower
        # for full-table streaming.
        if positions_path is not None:
            # Persistent table: positions live in a sidecar .b2nd file.
            positions_nd = blosc2.open(positions_path, mode="r")
        else:
            # In-memory table: positions live in the sidecar handle cache.
            from blosc2.indexing import _SIDECAR_HANDLE_CACHE, _sidecar_handle_cache_key

            target_arr = root._cols.get(name)
            if target_arr is None:
                return None
            token = descriptor["token"]
            cache_key = _sidecar_handle_cache_key(target_arr, token, "full", "positions")
            positions_nd = _SIDECAR_HANDLE_CACHE.get(cache_key)
            if positions_nd is None:
                return None

        positions = np.asarray(positions_nd[:], dtype=np.int64)
        valid = root._valid_rows[:]
        positions = np.asarray(positions, dtype=np.int64)
        positions = positions[(positions >= 0) & (positions < len(valid))]
        positions = positions[valid[positions]]
        if self is not root:
            current_valid = self._valid_rows[:]
            positions = positions[current_valid[positions]]
        if not ascending:
            positions = positions[::-1]
        return positions

    def _build_lex_keys(
        self,
        cols: list[str],
        ascending: list[bool],
        live_pos: np.ndarray,
        n: int,
    ) -> list[np.ndarray]:
        """Build the key list for np.lexsort (innermost = last = primary key).

        For nullable columns a null-indicator key (0=non-null, 1=null) is
        inserted immediately after the value key, making it more significant.
        This ensures nulls sort last regardless of ascending/descending order.
        """
        lex_keys = []
        for name, asc in zip(reversed(cols), reversed(ascending), strict=True):
            cc = self._computed_cols.get(name)
            if cc is not None:
                # Materialise computed column values at live positions
                raw = np.asarray(cc["lazy"][:])[live_pos]
            else:
                raw = self._cols[name][live_pos]
            col_info = self._schema.columns_by_name.get(name)
            nv = getattr(col_info.spec, "null_value", None) if col_info else None

            # Value key
            if not asc:
                if raw.dtype.kind in "US":
                    # strings can't be negated — invert via rank
                    rank = np.argsort(np.argsort(raw, kind="stable"), kind="stable")
                    lex_keys.append((n - 1 - rank).astype(np.intp))
                elif np.issubdtype(raw.dtype, np.unsignedinteger):
                    lex_keys.append(-raw.astype(np.int64))
                else:
                    lex_keys.append(-raw)
            else:
                lex_keys.append(raw)

            # Null indicator key — more significant than the value key above,
            # so nulls always sort last (0 before 1 → non-null before null).
            if nv is not None:
                if isinstance(nv, float) and np.isnan(nv):
                    null_ind = np.isnan(raw).astype(np.intp)
                else:
                    null_ind = (raw == nv).astype(np.intp)
                lex_keys.append(null_ind)

        return lex_keys

    def sort_by(
        self,
        cols: str | list[str],
        ascending: bool | list[bool] = True,
        *,
        inplace: bool = False,
    ) -> CTable:
        """Return a copy of the table sorted by one or more columns.

        Parameters
        ----------
        cols:
            Column name or list of column names to sort by.  When multiple
            columns are given, the first is the primary key, the second is
            the tiebreaker, and so on.
        ascending:
            Sort direction.  A single bool applies to all keys; a list must
            have the same length as *cols*.
        inplace:
            If ``True``, rewrite the physical data in place and return
            ``self`` (like :meth:`compact` but sorted).  If ``False``
            (default), return a new in-memory CTable leaving this one
            untouched.

        Raises
        ------
        ValueError
            If called on a view or a read-only table when ``inplace=True``.
        KeyError
            If any column name is not found.
        TypeError
            If a column used as a sort key does not support ordering
            (e.g. complex numbers).
        """
        if self.base is not None and inplace:
            raise ValueError(
                "Cannot sort a view inplace (would modify shared column data). Use sort_by(inplace=False) to get a sorted copy."
            )
        if inplace and self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")

        cols, ascending = self._normalise_sort_keys(cols, ascending)

        # Live physical positions
        valid_np = self._valid_rows[:]
        live_pos = np.where(valid_np)[0]
        n = len(live_pos)

        if n == 0:
            if inplace:
                return self
            return self._empty_copy()

        sorted_pos = None
        if len(cols) == 1:
            sorted_pos = self._sorted_positions_from_full_index(cols[0], ascending[0])
            if sorted_pos is not None and len(sorted_pos) != n:
                sorted_pos = None

        if sorted_pos is None:
            order = np.lexsort(self._build_lex_keys(cols, ascending, live_pos, n))
            sorted_pos = live_pos[order]

        if inplace:
            for col in self._schema.columns:
                arr = self._cols[col.name]
                if self._is_list_column(col):
                    new_arr = ListArray(spec=col.spec)
                    new_arr.extend((arr[int(pos)] for pos in sorted_pos), validate=False)
                    new_arr.flush()
                    self._cols[col.name] = new_arr
                else:
                    arr[:n] = arr[sorted_pos]
            self._valid_rows[:n] = True
            self._valid_rows[n:] = False
            self._n_rows = n
            self._last_pos = n
            self._mark_all_indexes_stale()
            return self
        else:
            # Build a new in-memory table with the sorted rows
            result = self._empty_copy()
            for col in self._schema.columns:
                col_name = col.name
                arr = self._cols[col_name]
                if self._is_list_column(col):
                    result._cols[col_name].extend((arr[int(pos)] for pos in sorted_pos), validate=False)
                    result._cols[col_name].flush()
                else:
                    result._cols[col_name][:n] = arr[sorted_pos]
            result._valid_rows[:n] = True
            result._valid_rows[n:] = False
            result._n_rows = n
            result._last_pos = n
            return result

    def copy(
        self,
        compact: bool = True,
        *,
        urlpath: str | os.PathLike[str] | None = None,
        overwrite: bool = False,
    ) -> CTable:
        """Return a new standalone copy of this table.

        Parameters
        ----------
        compact:
            If ``True`` (default), only live (non-deleted) rows are copied.
            The result is a dense table with no tombstones and no parent
            dependency — ideal for materialising a filtered view.
            If ``False``, all physical slots are copied including deleted gaps,
            preserving the tombstone state exactly for in-memory copies.
        urlpath:
            Destination path for a persistent copy.  The ``.b2z`` extension
            selects a compact zip-backed store; any other path uses a
            directory-backed store.  A ``.b2d`` suffix is recommended for
            directory-backed stores.  If ``None`` (default), return an
            in-memory copy.
        overwrite:
            If ``True``, replace an existing persistent destination.
        """
        if urlpath is not None:
            urlpath = os.fspath(urlpath)
            if urlpath.endswith(".b2z"):
                self.to_b2z(urlpath, overwrite=overwrite, compact=compact)
            else:
                self.to_b2d(urlpath, overwrite=overwrite, compact=compact)
            return CTable.open(urlpath, mode="r")

        valid_np = self._valid_rows[:]
        live_pos = np.where(valid_np)[0]
        n_live = len(live_pos)

        if compact:
            n = n_live
        else:
            # High watermark: number of slots ever written.
            # List columns are written sequentially with no gaps — their length
            # is the exact high watermark.  For scalar-only tables fall back to
            # the last live position + 1 (writes are always sequential so no
            # deleted slot can exist beyond the last live one).
            n = 0
            for col in self._schema.columns:
                if self._is_list_column(col):
                    n = len(self._cols[col.name])
                    break
            if n == 0:
                n = int(live_pos[-1]) + 1 if n_live > 0 else 0

        result = self._empty_copy(capacity=n)

        for col in self._schema.columns:
            col_name = col.name
            arr = self._cols[col_name]
            if self._is_list_column(col):
                src = (arr[int(pos)] for pos in live_pos) if compact else (arr[i] for i in range(n))
                result._cols[col_name].extend(src, validate=False)
                result._cols[col_name].flush()
            else:
                result._cols[col_name][:n] = arr[live_pos] if compact else arr[:n]

        if compact:
            result._valid_rows[:n] = True
            result._n_rows = n
            result._last_pos = n - 1 if n > 0 else None
        else:
            result._valid_rows[:n] = valid_np[:n]
            result._n_rows = n_live
            result._last_pos = None  # recomputed lazily on next append

        return result

    def _empty_copy(self, capacity: int | None = None) -> CTable:
        """Return a new empty in-memory CTable with the same schema and capacity."""
        from blosc2 import compute_chunks_blocks

        capacity = max(capacity if capacity is not None else self._n_rows, 1)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        mem_storage = InMemoryTableStorage()

        new_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        new_cols = {}
        for col in self._schema.columns:
            col_storage = self._resolve_column_storage(col, default_chunks, default_blocks)
            if self._is_list_column(col):
                new_cols[col.name] = mem_storage.create_list_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
            else:
                new_cols[col.name] = mem_storage.create_column(
                    col.name,
                    dtype=col.dtype,
                    shape=(capacity,),
                    chunks=col_storage["chunks"],
                    blocks=col_storage["blocks"],
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )

        obj = CTable.__new__(CTable)
        obj._schema = self._schema
        obj._row_type = self._row_type
        obj._table_cparams = self._table_cparams
        obj._table_dparams = self._table_dparams
        obj._storage = mem_storage
        obj._valid_rows = new_valid
        obj._cols = new_cols
        obj._col_widths = self._col_widths.copy()
        obj.col_names = [col.name for col in self._schema.columns]
        obj._materialized_cols = {name: dict(meta) for name, meta in self._materialized_cols.items()}
        obj._expr_index_arrays = dict(self._expr_index_arrays)
        # Rebuild computed columns with the new NDArray objects as operands
        obj._computed_cols = {}
        for cc_name, cc in self._computed_cols.items():
            operands = {f"o{i}": new_cols[dep] for i, dep in enumerate(cc["col_deps"])}
            new_lazy = blosc2.lazyexpr(cc["expression"], operands)
            obj._computed_cols[cc_name] = {
                "expression": cc["expression"],
                "col_deps": cc["col_deps"],
                "lazy": new_lazy,
                "dtype": cc["dtype"],
            }
            obj.col_names.append(cc_name)
            obj._col_widths.setdefault(cc_name, max(len(cc_name), 15))
        obj._n_rows = 0
        obj._last_pos = None
        obj._read_only = False
        obj.base = None
        obj.auto_compact = self.auto_compact
        obj._validate = self._validate
        return obj

    # ------------------------------------------------------------------
    # Properties / info
    # ------------------------------------------------------------------

    @property
    def nrows(self) -> int:
        return self._n_rows

    @property
    def ncols(self) -> int:
        """Total number of columns, including computed (virtual) columns."""
        return len(self.col_names)

    @property
    def cbytes(self) -> int:
        """Total compressed size in bytes (all columns + valid_rows mask)."""
        return sum(col.cbytes for col in self._cols.values()) + self._valid_rows.cbytes

    @property
    def nbytes(self) -> int:
        """Total uncompressed size in bytes (all columns + valid_rows mask)."""
        return sum(col.nbytes for col in self._cols.values()) + self._valid_rows.nbytes

    @property
    def cratio(self) -> float:
        """Compression ratio for the whole table payload."""
        if self.cbytes == 0:
            return float("inf")
        return self.nbytes / self.cbytes

    @property
    def schema(self) -> CompiledSchema:
        """The compiled schema that drives this table's columns and validation."""
        return self._schema

    def column_schema(self, name: str) -> CompiledColumn:
        """Return the :class:`CompiledColumn` descriptor for *name*.

        Raises
        ------
        KeyError
            If *name* is not a column in this table.
        """
        try:
            return self._schema.columns_by_name[name]
        except KeyError:
            raise KeyError(f"No column named {name!r}. Available: {self.col_names}") from None

    def schema_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible dict describing this table's schema."""
        return schema_to_dict(self._schema)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    @property
    def _root_table(self) -> CTable:
        """Return the root (non-view) table; *self* if not a view."""
        t = self
        while t.base is not None:
            t = t.base
        return t

    def _mark_all_indexes_stale(self) -> None:
        """Bump value_epoch and mark every catalog entry stale on the root table."""
        root = self._root_table
        root._storage.bump_value_epoch()
        catalog = root._storage.load_index_catalog()
        if not catalog:
            return
        changed = False
        for desc in catalog.values():
            if not desc.get("stale", False):
                desc["stale"] = True
                changed = True
        if changed:
            root._storage.save_index_catalog(catalog)

    @staticmethod
    def _validate_index_descriptor(col_name: str, descriptor: dict) -> None:
        """Raise ValueError when an index catalog entry is malformed."""
        if not isinstance(descriptor, dict):
            raise ValueError(f"Malformed index metadata for column {col_name!r}: descriptor must be a dict.")
        token = descriptor.get("token")
        if not isinstance(token, str) or not token:
            raise ValueError(f"Malformed index metadata for column {col_name!r}: missing token.")
        kind = descriptor.get("kind")
        if kind not in {"summary", "bucket", "partial", "full"}:
            raise ValueError(f"Malformed index metadata for column {col_name!r}: invalid kind {kind!r}.")
        if kind == "bucket" and not isinstance(descriptor.get("bucket"), dict):
            raise ValueError(f"Malformed index metadata for column {col_name!r}: missing bucket payload.")
        if kind == "partial" and not isinstance(descriptor.get("partial"), dict):
            raise ValueError(f"Malformed index metadata for column {col_name!r}: missing partial payload.")
        if kind == "full" and not isinstance(descriptor.get("full"), dict):
            raise ValueError(f"Malformed index metadata for column {col_name!r}: missing full payload.")

    def _drop_index_descriptor(self, col_name: str, descriptor: dict) -> None:
        """Delete sidecars/cache for a catalog descriptor without touching the column mapping."""
        from pathlib import Path

        from blosc2.indexing import (
            _IN_MEMORY_INDEXES,
            _PERSISTENT_INDEXES,
            _array_key,
            _clear_cached_data,
            _drop_descriptor_sidecars,
            _is_persistent_array,
        )

        token = descriptor["token"]
        col_arr = None
        with contextlib.suppress(Exception):
            col_arr = self._index_target_array(col_name, descriptor)

        if col_arr is not None:
            _clear_cached_data(col_arr, token)

        if col_arr is not None and _is_persistent_array(col_arr):
            arr_key = _array_key(col_arr)
            store = _PERSISTENT_INDEXES.get(arr_key)
            if store is not None:
                store["indexes"].pop(token, None)
        elif col_arr is not None:
            store = _IN_MEMORY_INDEXES.get(id(col_arr))
            if store is not None:
                store["indexes"].pop(token, None)

        _drop_descriptor_sidecars(descriptor)
        self._root_table._expr_index_arrays.pop(token, None)

        expr_values_path = descriptor.get("expr_values_path")
        if expr_values_path is not None:
            with contextlib.suppress(OSError):
                os.remove(expr_values_path)

        anchor = self._storage.index_anchor_path(col_name)
        if anchor is not None:
            proxy_key = ("persistent", str(Path(anchor).resolve()))
            _PERSISTENT_INDEXES.pop(proxy_key, None)
            with contextlib.suppress(OSError):
                os.rmdir(os.path.dirname(anchor))

    def _index_create_kwargs_from_descriptor(self, descriptor: dict) -> dict[str, Any]:
        """Return create_index kwargs that rebuild an existing descriptor."""
        build = "ooc" if bool(descriptor.get("ooc", False)) else "memory"
        kwargs = {
            "kind": descriptor["kind"],
            "optlevel": int(descriptor.get("optlevel", 5)),
            "name": descriptor.get("name") or None,
            "build": build,
            "cparams": descriptor.get("cparams"),
        }
        if descriptor.get("kind") == "full":
            kwargs["method"] = descriptor.get("full", {}).get("build_method", "global-sort")
        if descriptor.get("kind") == "opsi":
            kwargs["opsi_max_cycles"] = descriptor.get("opsi", {}).get("max_cycles")
        target = descriptor.get("target") or {}
        if target.get("source") == "expression":
            kwargs["expression"] = target.get("expression")
        return kwargs

    def _normalize_table_expression_target(
        self, expression: str, operands: dict | None = None
    ) -> tuple[dict, np.dtype]:
        """Normalize a same-table expression target and infer its dtype."""
        if operands is None:
            operands = self._cols
        try:
            ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("expression is not valid Python syntax") from exc

        owned_ids = {id(arr): name for name, arr in self._root_table._cols.items()}
        dependencies: list[str] = []
        valid = True

        class _Canonicalizer(ast.NodeTransformer):
            def visit_Name(self_inner, node: ast.Name) -> ast.AST:
                nonlocal valid
                operand = operands.get(node.id)
                if operand is None or not isinstance(operand, blosc2.NDArray):
                    return node
                cname = owned_ids.get(id(operand))
                if cname is None:
                    valid = False
                    return node
                dependencies.append(cname)
                return ast.copy_location(ast.Name(id=cname, ctx=node.ctx), node)

        normalized = _Canonicalizer().visit(
            ast.fix_missing_locations(ast.parse(expression, mode="eval")).body
        )
        if not valid or not dependencies:
            raise ValueError("expression indexes require operands from stored columns of the same table")
        dependencies = list(dict.fromkeys(dependencies))
        expression_key = ast.unparse(normalized)
        lazy = blosc2.lazyexpr(expression_key, {dep: self._root_table._cols[dep] for dep in dependencies})
        sample_stop = min(
            len(self._root_table._valid_rows), max(1, int(self._root_table._valid_rows.blocks[0]))
        )
        sample = lazy[:sample_stop]
        if isinstance(sample, blosc2.NDArray):
            sample = sample[:]
        sample = np.asarray(sample)
        dtype = np.dtype(sample.dtype)
        if sample.ndim != 1:
            raise ValueError("expression indexes require expressions returning a 1-D scalar stream")
        target = {
            "source": "expression",
            "expression": expression,
            "expression_key": expression_key,
            "dependencies": dependencies,
        }
        return target, dtype

    def _expression_index_values_path(self, token: str) -> str | None:
        anchor = self._storage.index_anchor_path(token)
        if anchor is None:
            return None
        return os.path.join(os.path.dirname(anchor), "values.b2nd")

    def _build_expression_values_array(self, target: dict, dtype: np.dtype, cparams=None) -> blosc2.NDArray:
        """Build a physical 1-D values array for a table expression target."""
        from blosc2.indexing import _target_token

        root = self._root_table
        capacity = len(root._valid_rows)
        chunks, blocks = compute_chunks_blocks((capacity,), dtype=dtype)
        urlpath = root._expression_index_values_path(_target_token(target))
        if urlpath is not None:
            os.makedirs(os.path.dirname(urlpath), exist_ok=True)
            arr = blosc2.zeros(
                (capacity,), dtype=dtype, urlpath=urlpath, mode="w", chunks=chunks, blocks=blocks
            )
        else:
            arr = blosc2.zeros((capacity,), dtype=dtype, chunks=chunks, blocks=blocks)
        lazy = blosc2.lazyexpr(
            target["expression_key"], {dep: root._cols[dep] for dep in target["dependencies"]}
        )
        step = int(root._valid_rows.chunks[0]) if root._valid_rows.chunks else 65536
        for start in range(0, capacity, step):
            stop = min(start + step, capacity)
            values = lazy[start:stop]
            if isinstance(values, blosc2.NDArray):
                values = values[:]
            arr[start:stop] = np.asarray(values, dtype=dtype)
        root._expr_index_arrays[_target_token(target)] = arr
        return arr

    def _index_target_array(self, lookup_key: str, descriptor: dict) -> blosc2.NDArray:
        """Return the physical array backing a column or expression index."""
        target = descriptor.get("target") or {}
        if target.get("source") != "expression":
            return self._root_table._cols[lookup_key]
        token = descriptor["token"]
        root = self._root_table
        arr = root._expr_index_arrays.get(token)
        if arr is not None:
            return arr
        path = descriptor.get("expr_values_path")
        if path is None:
            raise KeyError(f"No backing array found for expression index {token!r}.")
        arr = blosc2.open(path, mode="r" if root._read_only else "a")
        root._expr_index_arrays[token] = arr
        return arr

    def _resolve_index_catalog_entry(
        self, col_name: str | None = None, *, expression: str | None = None, name: str | None = None
    ) -> tuple[str, dict]:
        """Resolve an index catalog entry by column, expression, or label."""
        catalog = self._root_table._storage.load_index_catalog()
        if col_name is not None and expression is not None:
            raise ValueError("col_name and expression are mutually exclusive")
        if col_name is not None:
            if col_name not in catalog:
                raise KeyError(f"No index found for column {col_name!r}.")
            return col_name, catalog[col_name]
        if expression is not None:
            from blosc2.indexing import _target_token

            target, _ = self._normalize_table_expression_target(expression)
            token = _target_token(target)
            if token not in catalog:
                raise KeyError(f"No index found for expression {expression!r}.")
            return token, catalog[token]
        if name is not None:
            matches = [(key, desc) for key, desc in catalog.items() if desc.get("name") == name]
            if not matches:
                raise KeyError(f"No index found with name {name!r}.")
            if len(matches) > 1:
                raise ValueError(f"Multiple indexes found with name {name!r}; specify a target explicitly.")
            return matches[0]
        raise TypeError("must specify col_name, expression, or name")

    def _build_index_persistent(
        self,
        col_name: str,
        col_arr: blosc2.NDArray,
        *,
        kind: str,
        optlevel: int,
        name_hint: str | None,
        build: str,
        tmpdir: str | None,
        cparams_obj,
        method: str | None = None,
        opsi_max_cycles: int | None = None,
    ) -> dict:
        """Build index sidecar files for a persistent-table column; return the descriptor."""
        import tempfile
        from pathlib import Path

        from blosc2.indexing import (
            _PERSISTENT_INDEXES,
            _array_key,
            _build_bucket_descriptor,
            _build_bucket_descriptor_ooc,
            _build_descriptor,
            _build_full_descriptor,
            _build_full_descriptor_ooc,
            _build_levels_descriptor,
            _build_levels_descriptor_ooc,
            _build_opsi_descriptor,
            _build_partial_descriptor,
            _build_partial_descriptor_ooc,
            _copy_descriptor,
            _field_target_descriptor,
            _resolve_full_index_tmpdir,
            _resolve_ooc_mode,
            _target_token,
            _values_for_target,
        )

        anchor = self._storage.index_anchor_path(col_name)
        os.makedirs(os.path.dirname(anchor), exist_ok=True)
        proxy = _CTableBuildProxy(col_arr, anchor)
        proxy_key = _array_key(proxy)
        _PERSISTENT_INDEXES.pop(proxy_key, None)  # clear any stale cache entry

        target = _field_target_descriptor(None)
        token = _target_token(target)
        persistent = True
        dtype = col_arr.dtype
        use_ooc = _resolve_ooc_mode(kind, build)
        if opsi_max_cycles is None:
            opsi_max_cycles = max(1, optlevel if optlevel < 8 else optlevel * 2)

        if use_ooc:
            resolved_tmpdir = _resolve_full_index_tmpdir(proxy, tmpdir)
            levels = _build_levels_descriptor_ooc(proxy, target, token, kind, dtype, persistent, cparams_obj)
            bucket = (
                _build_bucket_descriptor_ooc(
                    proxy, target, token, kind, dtype, optlevel, persistent, cparams_obj
                )
                if kind == "bucket"
                else None
            )
            partial = (
                _build_partial_descriptor_ooc(
                    proxy, target, token, kind, dtype, optlevel, persistent, cparams_obj
                )
                if kind == "partial"
                else None
            )
            full = None
            opsi = None
            if kind == "full":
                with tempfile.TemporaryDirectory(prefix="blosc2-index-ooc-", dir=resolved_tmpdir) as td:
                    full = _build_full_descriptor_ooc(
                        proxy, target, token, kind, dtype, persistent, Path(td), cparams_obj, optlevel
                    )
                    full["build_method"] = "global-sort"
            if kind == "opsi":
                opsi = _build_opsi_descriptor(
                    proxy, target, token, kind, dtype, persistent, cparams_obj, opsi_max_cycles, optlevel
                )
            descriptor = _build_descriptor(
                proxy,
                target,
                token,
                kind,
                optlevel,
                persistent,
                True,
                name_hint,
                dtype,
                levels,
                bucket,
                partial,
                full,
                cparams_obj,
                opsi,
            )
        else:
            values = _values_for_target(proxy, target)
            levels = _build_levels_descriptor(
                proxy, target, token, kind, dtype, values, persistent, cparams_obj
            )
            bucket = (
                _build_bucket_descriptor(proxy, token, kind, values, optlevel, persistent, cparams_obj)
                if kind == "bucket"
                else None
            )
            partial = (
                _build_partial_descriptor(proxy, token, kind, values, optlevel, persistent, cparams_obj)
                if kind == "partial"
                else None
            )
            full = None
            opsi = None
            if kind == "full":
                full = _build_full_descriptor(proxy, token, kind, values, persistent, cparams_obj, optlevel)
                full["build_method"] = "global-sort"
            if kind == "opsi":
                opsi = _build_opsi_descriptor(
                    proxy, target, token, kind, dtype, persistent, cparams_obj, opsi_max_cycles, optlevel
                )
            descriptor = _build_descriptor(
                proxy,
                target,
                token,
                kind,
                optlevel,
                persistent,
                False,
                name_hint,
                dtype,
                levels,
                bucket,
                partial,
                full,
                cparams_obj,
                opsi,
            )

        result = _copy_descriptor(descriptor)
        _PERSISTENT_INDEXES.pop(proxy_key, None)  # evict proxy to avoid memory leak
        return result

    def create_index(  # noqa: C901
        self,
        col_name: str | None = None,
        *,
        field: str | None = None,
        expression: str | None = None,
        operands: dict | None = None,
        kind: blosc2.IndexKind = blosc2.IndexKind.BUCKET,
        optlevel: int = 5,
        name: str | None = None,
        build: str = "auto",
        tmpdir: str | None = None,
        **kwargs,
    ) -> blosc2.Index:
        """Build and register an index for a stored column or table expression."""
        if self.base is not None:
            raise ValueError("Cannot create an index on a view.")
        if col_name is not None and field is not None:
            raise ValueError("col_name and field are mutually exclusive")
        if expression is not None and (col_name is not None or field is not None):
            raise ValueError("column targets and expression are mutually exclusive")
        if operands is not None and expression is None:
            raise ValueError("operands can only be provided together with expression")
        col_name = field if field is not None else col_name

        from blosc2.indexing import (
            _IN_MEMORY_INDEXES,
            _copy_descriptor,
            _normalize_build_mode,
            _normalize_full_build_method,
            _normalize_index_cparams,
            _normalize_index_kind,
            _target_token,
        )
        from blosc2.indexing import create_index as _ix_create_index

        cparams_obj = _normalize_index_cparams(kwargs.pop("cparams", None))
        method = kwargs.pop("method", None)
        opsi_max_cycles = kwargs.pop("opsi_max_cycles", None)
        if opsi_max_cycles is not None:
            opsi_max_cycles = max(1, int(opsi_max_cycles))
        if kwargs:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(kwargs))}")

        kind_str = _normalize_index_kind(kind)
        build_str = _normalize_build_mode(build)
        method_str = _normalize_full_build_method(method) if kind_str == "full" else None
        if method is not None and kind_str != "full":
            raise ValueError("method is only supported for kind=IndexKind.FULL")
        catalog = self._storage.load_index_catalog()

        if expression is not None:
            target, dtype = self._normalize_table_expression_target(expression, operands)
            token = _target_token(target)
            if token in catalog:
                raise ValueError(
                    f"Index already exists for expression {expression!r}. "
                    "Call rebuild_index() to replace it or drop_index() first."
                )
            expr_arr = self._build_expression_values_array(target, dtype, cparams=cparams_obj)
            _ix_create_index(
                expr_arr,
                kind=blosc2.IndexKind(kind_str),
                optlevel=optlevel,
                name=name,
                build=build,
                tmpdir=tmpdir,
                cparams=cparams_obj,
                method=method_str,
                opsi_max_cycles=opsi_max_cycles,
            )
            store = _IN_MEMORY_INDEXES.get(id(expr_arr))
            if store is None:
                from blosc2.indexing import _load_store

                store = _load_store(expr_arr)
            descriptor = _copy_descriptor(store["indexes"]["__self__"])
            descriptor["target"] = target
            descriptor["token"] = token
            descriptor["dtype"] = str(np.dtype(dtype))
            descriptor["expr_values_path"] = getattr(expr_arr, "urlpath", None)
            value_epoch, _ = self._storage.get_epoch_counters()
            descriptor["built_value_epoch"] = value_epoch
            catalog[token] = descriptor
            self._storage.save_index_catalog(catalog)
            return blosc2.Index._from_table(self, token, descriptor)

        if col_name is None:
            raise TypeError("must specify col_name/field or expression")
        if col_name in self._computed_cols:
            raise ValueError(
                f"Cannot create an index on computed column {col_name!r}: "
                "computed columns have no physical storage."
            )
        if col_name not in self._cols:
            raise KeyError(f"No column named {col_name!r}. Available: {self.col_names}")
        if col_name in catalog:
            raise ValueError(
                f"Index already exists for column {col_name!r}. "
                "Call rebuild_index() to replace it or drop_index() first."
            )

        col_arr = self._cols[col_name]
        if isinstance(self._schema.columns_by_name[col_name].spec, ListSpec):
            raise ValueError(f"Cannot create an index on list column {col_name!r} in V1.")
        if isinstance(
            self._schema.columns_by_name[col_name].spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec)
        ):
            raise NotImplementedError(
                f"Cannot create an index on variable-length scalar column {col_name!r}: "
                "indexing for vlstring/vlbytes/struct/object columns is not supported yet."
            )
        is_persistent = self._storage.index_anchor_path(col_name) is not None

        if is_persistent:
            descriptor = self._build_index_persistent(
                col_name,
                col_arr,
                kind=kind_str,
                optlevel=optlevel,
                name_hint=name,
                build=build_str,
                tmpdir=tmpdir,
                cparams_obj=cparams_obj,
                method=method_str,
                opsi_max_cycles=opsi_max_cycles,
            )
        else:
            _ix_create_index(
                col_arr,
                field=None,
                kind=blosc2.IndexKind(kind_str),
                optlevel=optlevel,
                name=name,
                build=build,
                tmpdir=tmpdir,
                cparams=cparams_obj,
                method=method_str,
                opsi_max_cycles=opsi_max_cycles,
            )
            store = _IN_MEMORY_INDEXES[id(col_arr)]
            descriptor = _copy_descriptor(store["indexes"]["__self__"])

        value_epoch, _ = self._storage.get_epoch_counters()
        descriptor["built_value_epoch"] = value_epoch

        catalog = self._storage.load_index_catalog()
        catalog[col_name] = descriptor
        self._storage.save_index_catalog(catalog)
        return blosc2.Index._from_table(self, col_name, descriptor)

    def drop_index(
        self, col_name: str | None = None, *, expression: str | None = None, name: str | None = None
    ) -> None:
        """Remove an index and delete any sidecar files."""
        if self.base is not None:
            raise ValueError("Cannot drop an index from a view.")

        lookup_key, descriptor = self._resolve_index_catalog_entry(
            col_name, expression=expression, name=name
        )
        catalog = self._storage.load_index_catalog()
        catalog.pop(lookup_key, None)
        self._validate_index_descriptor(lookup_key, descriptor)
        self._drop_index_descriptor(lookup_key, descriptor)
        self._storage.save_index_catalog(catalog)

    def rebuild_index(
        self, col_name: str | None = None, *, expression: str | None = None, name: str | None = None
    ) -> blosc2.Index:
        """Drop and recreate an index with the same parameters."""
        if self.base is not None:
            raise ValueError("Cannot rebuild an index on a view.")

        lookup_key, old_desc = self._resolve_index_catalog_entry(col_name, expression=expression, name=name)
        self._validate_index_descriptor(lookup_key, old_desc)
        create_kwargs = self._index_create_kwargs_from_descriptor(old_desc)

        self.drop_index(col_name, expression=expression, name=name)
        if "expression" in create_kwargs:
            return self.create_index(expression=create_kwargs.pop("expression"), **create_kwargs)
        return self.create_index(lookup_key, **create_kwargs)

    def compact_index(
        self, col_name: str | None = None, *, expression: str | None = None, name: str | None = None
    ) -> blosc2.Index:
        """Compact an index, merging any incremental append runs."""
        if self.base is not None:
            raise ValueError("Cannot compact an index on a view.")

        from blosc2.indexing import (
            _IN_MEMORY_INDEXES,
            _PERSISTENT_INDEXES,
            _array_key,
            _copy_descriptor,
            _default_index_store,
            _is_persistent_array,
        )
        from blosc2.indexing import compact_index as _ix_compact_index

        lookup_key, descriptor = self._resolve_index_catalog_entry(
            col_name, expression=expression, name=name
        )
        col_arr = self._index_target_array(lookup_key, descriptor)
        catalog = self._storage.load_index_catalog()

        if _is_persistent_array(col_arr):
            anchor = self._storage.index_anchor_path(lookup_key)
            proxy = _CTableBuildProxy(col_arr, anchor)
            proxy_key = _array_key(proxy)
            store = _default_index_store()
            store["indexes"][descriptor["token"]] = descriptor
            _PERSISTENT_INDEXES[proxy_key] = store
            try:
                _ix_compact_index(proxy)
                updated_store = _PERSISTENT_INDEXES.get(proxy_key) or store
                updated_desc = _copy_descriptor(updated_store["indexes"][descriptor["token"]])
            finally:
                _PERSISTENT_INDEXES.pop(proxy_key, None)
            updated_desc["built_value_epoch"] = descriptor.get("built_value_epoch", 0)
            catalog[lookup_key] = updated_desc
            self._storage.save_index_catalog(catalog)
            return blosc2.Index._from_table(self, lookup_key, updated_desc)
        else:
            _ix_compact_index(col_arr)
            store = _IN_MEMORY_INDEXES.get(id(col_arr))
            if store:
                token = descriptor["token"]
                updated_desc = _copy_descriptor(store["indexes"].get(token, descriptor))
                updated_desc["built_value_epoch"] = descriptor.get("built_value_epoch", 0)
                catalog[lookup_key] = updated_desc
                self._storage.save_index_catalog(catalog)
                return blosc2.Index._from_table(self, lookup_key, updated_desc)
            return blosc2.Index._from_table(self, lookup_key, descriptor)

    def index(
        self, col_name: str | None = None, *, expression: str | None = None, name: str | None = None
    ) -> blosc2.Index:
        """Return the index handle for a stored-column or expression target."""
        lookup_key, descriptor = self._resolve_index_catalog_entry(
            col_name, expression=expression, name=name
        )
        return blosc2.Index._from_table(self, lookup_key, descriptor)

    @property
    def indexes(self) -> list[blosc2.Index]:
        """Return a list of :class:`blosc2.Index` handles for all active indexes."""
        catalog = self._root_table._storage.load_index_catalog()
        return [blosc2.Index._from_table(self, col_name, desc) for col_name, desc in catalog.items()]

    def _rewrite_expression_query_for_index(
        self, expression: str, operands: dict, target: dict
    ) -> str | None:
        """Rewrite matching table-expression subtrees to ``_where_x`` for planning."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return None

        class _Rewriter(ast.NodeTransformer):
            def __init__(self, outer):
                self.outer = outer
                self.changed = False

            def generic_visit(self, node):
                normalized = None
                with contextlib.suppress(Exception):
                    normalized, _ = self.outer._normalize_table_expression_target(
                        ast.unparse(node), operands
                    )
                if normalized is not None and normalized.get("expression_key") == target.get(
                    "expression_key"
                ):
                    self.changed = True
                    return ast.copy_location(ast.Name(id="_where_x", ctx=ast.Load()), node)
                return super().generic_visit(node)

        rewriter = _Rewriter(self)
        new_body = rewriter.visit(tree.body)
        if not rewriter.changed:
            return None
        return ast.unparse(new_body)

    def _try_expression_index_where(self, expr_result: blosc2.LazyExpr, catalog: dict) -> np.ndarray | None:
        """Attempt to resolve *expr_result* via a direct table expression index."""
        from blosc2.indexing import evaluate_bucket_query, evaluate_segment_query, plan_query

        expression = expr_result.expression
        operands = dict(expr_result.operands)
        for lookup_key, descriptor in catalog.items():
            target = descriptor.get("target") or {}
            if target.get("source") != "expression" or descriptor.get("stale", False):
                continue
            rewritten = self._rewrite_expression_query_for_index(expression, operands, target)
            if rewritten is None:
                continue
            expr_arr = self._index_target_array(lookup_key, descriptor)
            where_dict = {"_where_x": expr_arr}
            merged_operands = {"_where_x": expr_arr}
            plan = plan_query(rewritten, merged_operands, where_dict)
            if not plan.usable:
                continue
            if plan.exact_positions is not None:
                return np.asarray(plan.exact_positions, dtype=np.int64)
            if plan.bucket_masks is not None:
                _, positions = evaluate_bucket_query(
                    rewritten, merged_operands, {}, where_dict, plan, return_positions=True
                )
                return np.asarray(positions, dtype=np.int64)
            if plan.candidate_units is not None and plan.segment_len is not None:
                _, positions = evaluate_segment_query(
                    rewritten, merged_operands, {}, where_dict, plan, return_positions=True
                )
                return np.asarray(positions, dtype=np.int64)
        return None

    @staticmethod
    def _find_indexed_columns(root_cols, catalog, operands):
        """Return live indexed columns referenced by *operands* in expression order."""
        indexed = []
        seen = set()
        for operand in operands.values():
            if not isinstance(operand, blosc2.NDArray):
                continue
            for col_name, col_arr in root_cols.items():
                if col_arr is not operand or col_name in seen or col_name not in catalog:
                    continue
                descriptor = catalog[col_name]
                CTable._validate_index_descriptor(col_name, descriptor)
                if descriptor.get("stale", False):
                    continue
                indexed.append((col_name, col_arr, descriptor))
                seen.add(col_name)
        return indexed

    def _try_index_where(self, expr_result: blosc2.LazyExpr) -> np.ndarray | None:  # noqa: C901
        """Attempt to resolve *expr_result* via a column index.

        Returns a 1-D int64 array of physical row positions that satisfy the
        predicate, or ``None`` if no usable index was found (caller falls back
        to a full scan).
        """
        from blosc2.indexing import (
            _IN_MEMORY_INDEXES,
            _PERSISTENT_INDEXES,
            _array_key,
            _default_index_store,
            _is_persistent_array,
            evaluate_bucket_query,
            evaluate_segment_query,
            plan_query,
        )

        root = self._root_table
        catalog = root._storage.load_index_catalog()
        if not catalog:
            return None

        positions = self._try_expression_index_where(expr_result, catalog)
        if positions is not None:
            return positions

        expression = expr_result.expression
        operands = dict(expr_result.operands)

        indexed_columns = self._find_indexed_columns(root._cols, catalog, operands)
        if not indexed_columns:
            return None

        primary_col_name, primary_col_arr, _ = indexed_columns[0]
        nullable_indexed = [
            name
            for name, _arr, _descriptor in indexed_columns
            if getattr(root._schema.columns_by_name[name].spec, "null_value", None) is not None
        ]

        # Global null post-filtering is not correct for OR expressions.
        if nullable_indexed and ("|" in expr_result.expression or " or " in expr_result.expression):
            return None

        # Inject every usable table-owned descriptor so plan_query can combine them.
        # In .b2z read mode all columns share the same urlpath, so _array_key()
        # returns the same key for every column — causing _SIDECAR_HANDLE_CACHE
        # collisions across queries.  Clear stale handles before each injection so
        # the upcoming query always loads the correct sidecar for this column.
        from blosc2.indexing import _clear_cached_data

        for _col_name, col_arr, descriptor in indexed_columns:
            arr_key = _array_key(col_arr)
            if _is_persistent_array(col_arr):
                store = _PERSISTENT_INDEXES.get(arr_key) or _default_index_store()
                if store["indexes"].get(descriptor["token"]) is not descriptor:
                    _clear_cached_data(col_arr, descriptor["token"])
                store["indexes"][descriptor["token"]] = descriptor
                _PERSISTENT_INDEXES[arr_key] = store
            else:
                store = _IN_MEMORY_INDEXES.get(id(col_arr)) or _default_index_store()
                store["indexes"][descriptor["token"]] = descriptor
                _IN_MEMORY_INDEXES[id(col_arr)] = store

        where_dict = {"_where_x": primary_col_arr}
        merged_operands = {**operands, "_where_x": primary_col_arr}

        plan = plan_query(expression, merged_operands, where_dict)
        if not plan.usable:
            return None

        def _exclude_null_positions(positions):
            positions = np.asarray(positions, dtype=np.int64)
            for name in nullable_indexed:
                col = root._schema.columns_by_name[name]
                raw = root._cols[name][positions]
                nv = getattr(col.spec, "null_value", None)
                if isinstance(nv, float) and np.isnan(nv):
                    keep = ~np.isnan(raw)
                else:
                    keep = raw != nv
                positions = positions[keep]
            return positions

        if plan.exact_positions is not None:
            return _exclude_null_positions(plan.exact_positions)

        if plan.bucket_masks is not None:
            _, positions = evaluate_bucket_query(
                expression, merged_operands, {}, where_dict, plan, return_positions=True
            )
            return _exclude_null_positions(positions)

        if plan.candidate_units is not None and plan.segment_len is not None:
            _, positions = evaluate_segment_query(
                expression, merged_operands, {}, where_dict, plan, return_positions=True
            )
            return _exclude_null_positions(positions)

        return None

    @property
    def info_items(self) -> list[tuple[str, object]]:
        """Structured summary items used by :meth:`info`."""
        storage_type = "persistent" if isinstance(self._storage, FileTableStorage) else "in-memory"
        urlpath = self._storage._root if isinstance(self._storage, FileTableStorage) else None
        schema_summary = {}
        for name in self.col_names:
            if name in self._computed_cols:
                cc = self._computed_cols[name]
                schema_summary[name] = _InfoLiteral(
                    f"{cc['dtype']} (computed: {self._readable_computed_expr(cc)})"
                )
            else:
                col_meta = self._schema.columns_by_name.get(name)
                schema_summary[name] = _InfoLiteral(
                    self._dtype_info_label(
                        getattr(self._cols[name], "dtype", None), col_meta.spec if col_meta else None
                    )
                )

        index_summary = {}
        for idx in self.indexes:
            stale = " stale" if idx.stale else ""
            label = f" name={idx.name!r}" if idx.name and idx.name != "__self__" else ""
            stats = idx.storage_stats()
            if stats is None:
                suffix = "size=n/a (sidecars not directly addressable)"
            else:
                _, cbytes, _ = stats
                suffix = f"cbytes={format_nbytes_info(cbytes)}"
            index_summary[idx.col_name] = f"[{idx.kind}{stale}{label}] {suffix}"

        items = [
            ("type", self.__class__.__name__),
            ("storage", storage_type),
            ("rows", self.nrows),
            ("columns", self.ncols),
            ("view", self.base is not None),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.1f}x"),
            ("schema", schema_summary),
            (
                "valid_rows_mask",
                f"cbytes={format_nbytes_info(self._valid_rows.cbytes)}",
            ),
            ("indexes", index_summary if index_summary else "none"),
        ]
        if urlpath is not None:
            items.insert(2, ("urlpath", urlpath))
            open_mode = self._storage.open_mode()
            if open_mode is not None:
                items.insert(3, ("open_mode", open_mode))
        return items

    @staticmethod
    def _dtype_info_label(dtype: np.dtype | None, spec: SchemaSpec | None = None) -> str:
        """Return a compact dtype label for info reports."""
        if isinstance(spec, VLStringSpec):
            return "vlstring"
        if isinstance(spec, VLBytesSpec):
            return "vlbytes"
        if isinstance(spec, StructSpec):
            return spec.display_label()
        if isinstance(spec, ObjectSpec):
            return spec.display_label()
        if isinstance(spec, ListSpec):
            return spec.display_label()
        if dtype is None:
            return "None"
        if dtype.kind == "U":
            nchars = dtype.itemsize // 4
            return f"U{nchars} (Unicode, max {nchars} chars)"
        if dtype.kind == "S":
            return f"S{dtype.itemsize}"
        return str(dtype)

    @property
    def info(self) -> _CTableInfoReporter:
        """Get information about this table.

        Examples
        --------
        >>> print(t.info)
        >>> t.info()
        """
        return _CTableInfoReporter(self)

    # ------------------------------------------------------------------
    # Mutation: append / extend / delete
    # ------------------------------------------------------------------

    def _load_initial_data(self, new_data) -> None:
        """Dispatch new_data to append() or extend() as appropriate."""
        is_append = False

        if isinstance(new_data, (np.void, np.record)):
            is_append = True
        elif isinstance(new_data, np.ndarray):
            if new_data.dtype.names is not None and new_data.ndim == 0:
                is_append = True
        elif isinstance(new_data, list) and len(new_data) > 0:
            first_elem = new_data[0]
            if isinstance(first_elem, (str, bytes, int, float, bool, complex)):
                is_append = True

        if is_append:
            self.append(new_data)
        else:
            self.extend(new_data)

    def append(self, data: list | np.void | np.ndarray) -> None:
        """Append a single row to the table.

        *data* may be a list, tuple, ``numpy.void``, or structured
        ``numpy.ndarray`` whose fields match the schema column order.
        Materialized columns whose values are omitted are auto-filled from
        their recorded expression.  Raises ``ValueError`` if the table is
        read-only or a view.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise TypeError("Cannot extend view.")

        # Normalize → validate → coerce
        row = self._normalize_row_input(data)
        row = self._autofill_materialized_row_values(row)
        self._validate_no_default_columns_present(row)
        if self._validate:
            from blosc2.schema_validation import validate_row

            row = validate_row(self._schema, row)
        row = self._coerce_row_to_storage(row)

        pos = self._resolve_last_pos()
        if pos >= len(self._valid_rows):
            self._grow()

        for col in self._schema.columns:
            name = col.name
            col_array = self._cols[name]
            if self._is_list_column(col) or self._is_varlen_scalar_column(col):
                col_array.append(row[name])
            else:
                col_array[pos] = row[name]

        self._valid_rows[pos] = True
        self._last_pos = pos + 1
        self._n_rows += 1
        self._mark_all_indexes_stale()

    def delete(self, ind: int | slice | str | Iterable) -> None:
        """Mark one or more rows as deleted (tombstone deletion).

        *ind* may be a logical row index (``int``), a slice, or an iterable of
        logical indices.  Deleted rows are excluded from all subsequent queries
        and aggregates.  Physical storage is not reclaimed until
        :meth:`compact` is called.  Raises ``ValueError`` if the table is
        read-only or a view.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot delete rows from a view.")
        valid_rows_np = self._valid_rows[:]
        true_pos = np.where(valid_rows_np)[0]

        if isinstance(ind, Iterable) and not isinstance(ind, (str, bytes)):
            ind = list(ind)
        elif not isinstance(ind, int) and not isinstance(ind, slice):
            raise TypeError(f"Invalid type '{type(ind)}'")

        false_pos = true_pos[ind]
        n_deleted = len(np.unique(false_pos))

        valid_rows_np[false_pos] = False
        self._valid_rows[:] = valid_rows_np  # write back in-place; no new array created
        self._n_rows -= n_deleted
        self._last_pos = None  # recalculate on next write
        self._storage.bump_visibility_epoch()

    def extend(self, data: list | CTable | Any, *, validate: bool | None = None) -> None:  # noqa: C901
        """Append multiple rows at once.

        *data* may be:

        * a **dict of arrays** ``{"col": array, ...}`` — all arrays must have
          the same length; omitted columns are filled from their declared default;
          columns with no default declared must be provided;
        * a **list of rows**, each compatible with :meth:`append`;
        * another **CTable** — columns are matched by name.

        Pass ``validate=False`` to skip per-row Pydantic validation on trusted
        bulk imports.  Raises ``ValueError`` if the table is read-only or a view.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise TypeError("Cannot extend view.")
        if len(data) <= 0:
            if isinstance(data, dict):
                raise ValueError("No columns provided for extend().")
            return

        # Resolve effective validate flag: per-call override takes precedence
        do_validate = self._validate if validate is None else validate

        start_pos = self._resolve_last_pos()

        current_col_names = self._stored_col_names  # skip computed columns
        input_col_names = self._append_input_col_names
        new_nrows = 0
        provided_names: set[str] = set()

        if hasattr(data, "_cols") and hasattr(data, "_n_rows"):
            new_nrows = data._n_rows
            raw_columns = {}
            for name in current_col_names:
                if name in data._cols:
                    raw_columns[name] = data._cols[name][: data._n_rows]
                    provided_names.add(name)
        else:
            if isinstance(data, dict):
                known_names = [name for name in current_col_names if name in data]
                if not known_names:
                    raise ValueError("No known stored columns provided for extend().")
                column_lengths = {}
                for name in known_names:
                    try:
                        column_lengths[name] = len(data[name])
                    except TypeError as exc:
                        raise TypeError(f"Column {name!r} does not have a length.") from exc
                new_nrows = column_lengths[known_names[0]]
                mismatched = {name: n for name, n in column_lengths.items() if n != new_nrows}
                if mismatched:
                    details = ", ".join(f"{name}={n}" for name, n in mismatched.items())
                    raise ValueError(
                        f"All provided columns must have the same length; "
                        f"expected {new_nrows}, got {details}."
                    )
                provided_names = set(known_names)
                raw_columns = {name: data[name] for name in known_names}
            elif isinstance(data, np.ndarray) and data.dtype.names is not None:
                new_nrows = len(data)
                raw_columns = {name: data[name] for name in data.dtype.names if name in current_col_names}
                provided_names = set(raw_columns)
            else:
                new_nrows = len(data)
                batch_columns = list(zip(*data, strict=False))
                raw_columns = {
                    input_col_names[i]: batch_columns[i]
                    for i in range(min(len(input_col_names), len(batch_columns)))
                }
                provided_names = set(raw_columns)

        raw_columns = self._autofill_materialized_batch_columns(
            raw_columns, new_nrows, provided_names=provided_names
        )
        raw_columns = self._fill_default_batch_columns(raw_columns, new_nrows)

        # Validate constraints column-by-column before writing
        if do_validate:
            from blosc2.schema_vectorized import validate_column_batch

            validate_column_batch(self._schema, raw_columns)

        scalar_processed_cols: dict[str, blosc2.NDArray] = {}
        list_processed_cols: dict[str, list] = {}
        varlen_scalar_processed_cols: dict[str, list] = {}
        for name in current_col_names:
            col_meta = self._schema.columns_by_name[name]
            if self._is_list_column(col_meta):
                list_processed_cols[name] = list(raw_columns[name])
            elif self._is_varlen_scalar_column(col_meta):
                varlen_scalar_processed_cols[name] = list(raw_columns[name])
            else:
                target_dtype = self._cols[name].dtype
                scalar_processed_cols[name] = np.ascontiguousarray(raw_columns[name], dtype=target_dtype)

        end_pos = start_pos + new_nrows

        if self.auto_compact and end_pos >= len(self._valid_rows):
            self.compact()  # sets _last_pos = _n_rows
            start_pos = self._last_pos
            end_pos = start_pos + new_nrows

        while end_pos > len(self._valid_rows):
            self._grow()

        for name in current_col_names:
            col_meta = self._schema.columns_by_name[name]
            if self._is_list_column(col_meta):
                self._cols[name].extend(list_processed_cols[name], validate=do_validate)
            elif self._is_varlen_scalar_column(col_meta):
                self._cols[name].extend(varlen_scalar_processed_cols[name])
            else:
                self._cols[name][start_pos:end_pos] = scalar_processed_cols[name][:]

        self._valid_rows[start_pos:end_pos] = True
        self._last_pos = end_pos
        self._n_rows += new_nrows
        self._mark_all_indexes_stale()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _where_expression_operands(self) -> dict[str, blosc2.NDArray | blosc2.LazyExpr]:
        operands = {}
        for name, arr in self._cols.items():
            col = self._schema.columns_by_name.get(name)
            if col is not None and not (self._is_list_column(col) or self._is_varlen_scalar_column(col)):
                operands[name] = arr
        operands.update({name: cc["lazy"] for name, cc in self._computed_cols.items()})
        return operands

    def _guard_varlen_scalar_expression(self, expr: str) -> None:
        for col in self._schema.columns:
            if self._is_varlen_scalar_column(col) and re.search(
                rf"(?<!\w){re.escape(col.name)}(?!\w)", expr
            ):
                raise NotImplementedError(
                    f"Column {col.name!r} is a variable-length scalar column (vlstring/vlbytes/struct/object); "
                    "lazy expressions are not supported yet."
                )

    def where(
        self,
        expr_result: str | np.ndarray | blosc2.NDArray | blosc2.LazyExpr | Column,
        *,
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> CTable:
        """Return a row-filtered view matching a boolean predicate.

        Signature::

            where(expr_result) -> CTable

        The predicate can be supplied as a boolean :class:`blosc2.LazyExpr`,
        a boolean :class:`blosc2.NDArray`, a boolean NumPy array, a boolean
        ``Column``, or a string expression evaluated against this table's
        columns.  String expressions can reference stored and computed columns
        directly by name.

        The returned object is a :class:`CTable` view sharing the original
        column data.  The row-selection mask is evaluated immediately and
        intersected with the table's current live rows; selected column data is
        not copied.

        Parameters
        ----------
        expr_result:
            Boolean predicate selecting rows.  Strings are converted to a
            lazy expression with table columns as operands, e.g.
            ``"value * category >= 150"``.  Column objects can also be used in
            Python expressions, e.g. ``(t.value * t.category) >= 150``.

        Returns
        -------
        CTable
            A view over the same columns containing only rows where the
            predicate is true and the source row is live.  When ``columns`` is
            provided, the returned view is additionally projected to that
            ordered subset of columns.

        Raises
        ------
        TypeError
            If *expr_result* does not evaluate to a boolean Blosc2/NumPy
            array or lazy expression.

        Examples
        --------
        Filter using a string expression::

            view = t.where("value * category >= 150")
            slim = t.where("value * category >= 150", columns=["value", "category"])

        Filter using column arithmetic::

            view = t.where((t.value * t.category) >= 150)

        Blosc2 lazy functions can be used in column expressions::

            view = t.where(((t.value + 2) * blosc2.sin(t.category)) >= 10)

        For column names that are not valid Python identifiers, use item
        access::

            view = t.where((t["unit price"] * t["quantity"]) > 100)

        Notes
        -----
        Use bitwise operators (``&``, ``|``, ``~``) or string expressions for
        element-wise boolean logic.  Python's logical operators ``and``, ``or``
        and ``not`` cannot be overloaded and therefore do not build lazy column
        expressions.

        Use::

            t.where((t.x > 0) & (t.y < 10))
            t.where(~t.returned)
            t.where("not returned")

        not::

            t.where((t.x > 0) and (t.y < 10))
            t.where(not t.returned)
        """
        if isinstance(expr_result, str):
            self._guard_varlen_scalar_expression(expr_result)
            expr_result = blosc2.lazyexpr(expr_result, self._where_expression_operands())
        if isinstance(expr_result, np.ndarray) and expr_result.dtype == np.bool_:
            expr_result = blosc2.asarray(expr_result)
        if isinstance(expr_result, Column):
            expr_result = (
                expr_result._raw_col == 1 if expr_result._is_nullable_bool else expr_result._raw_col
            )

        if not (
            isinstance(expr_result, (blosc2.NDArray, blosc2.LazyExpr))
            and (getattr(expr_result, "dtype", None) == np.bool_)
        ):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(expr_result).__name__}")

        # Attempt index-accelerated filtering before falling back to a full scan.
        if isinstance(expr_result, blosc2.LazyExpr):
            positions = self._try_index_where(expr_result)
            if positions is not None:
                total = len(self._valid_rows)
                mask = np.zeros(total, dtype=bool)
                valid_pos = positions[(positions >= 0) & (positions < total)]
                mask[valid_pos] = True
                mask &= self._valid_rows[:]
                result = self.view(blosc2.asarray(mask))
                return result if columns is None else result.select(list(columns))

        filter = expr_result.compute() if isinstance(expr_result, blosc2.LazyExpr) else expr_result

        target_len = len(self._valid_rows)

        if len(filter) > target_len:
            filter = filter[:target_len]
        elif len(filter) < target_len:
            padding = blosc2.zeros(target_len, dtype=np.bool_)
            padding[: len(filter)] = filter[:]
            filter = padding

        filter = (filter & self._valid_rows).compute()

        result = self.view(filter)
        return result if columns is None else result.select(list(columns))

    def _run_row_logic(self, ind: int | slice | str | Iterable) -> CTable:
        valid_rows_np = self._valid_rows[:]
        true_pos = np.where(valid_rows_np)[0]

        if isinstance(ind, Iterable) and not isinstance(ind, (str, bytes)):
            ind = list(ind)

        mant_pos = true_pos[ind]

        new_mask_np = np.zeros_like(valid_rows_np, dtype=bool)
        new_mask_np[mant_pos] = True

        new_mask = blosc2.asarray(new_mask_np)
        return self.view(new_mask)
