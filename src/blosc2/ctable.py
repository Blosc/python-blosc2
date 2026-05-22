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
import json
import os
import pprint
import re
import shutil
from collections import deque, namedtuple
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING, dataclass
from dataclasses import field as dataclass_field
from textwrap import TextWrapper
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

import numpy as np

import blosc2
from blosc2 import compute_chunks_blocks
from blosc2.ctable_storage import (
    FileTableStorage,
    InMemoryTableStorage,
    TableStorage,
    TreeStoreTableStorage,
    _column_name_to_relpath,
    join_field_path,
    split_field_path,
)
from blosc2.info import InfoReporter, format_nbytes_info
from blosc2.list_array import ListArray, coerce_list_cell
from blosc2.scalar_array import _ScalarVarLenArray

if TYPE_CHECKING:
    from blosc2.dictionary_column import DictionaryColumn
from blosc2.schema import (
    DictionarySpec,
    ListSpec,
    NDArraySpec,
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
    timestamp,
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
    timestamp_value: int = int(np.iinfo(np.int64).min)
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
        if pa.types.is_timestamp(pa_type):
            return self.timestamp_value
        return None


DEFAULT_NULL_POLICY = NullPolicy()
_NULL_POLICY = contextvars.ContextVar("blosc2_null_policy", default=DEFAULT_NULL_POLICY)
_CTABLE_PRINT_OPTIONS: dict[str, Any] = {
    "display_index": True,
    "display_rows": 60,
    "display_precision": 6,
    "fancy": False,
}
_SMALL_NROWS_LIMIT = 50_000_000
_SMALL_SORT_MATERIALIZE_LIMIT = _SMALL_NROWS_LIMIT
_WHERE_NUMPY_MASK_LIMIT = _SMALL_NROWS_LIMIT


def get_null_policy() -> NullPolicy:
    """Return the current default null policy."""
    return _NULL_POLICY.get()


def set_printoptions(
    *,
    display_index: bool | None = None,
    display_rows: int | None = None,
    display_precision: int | None = None,
    fancy: bool | None = None,
) -> None:
    """Set global display options for :class:`CTable` string representations.

    Parameters
    ----------
    display_index:
        Whether ``str(ctable)`` should include a pandas-like logical row index
        column.  ``None`` leaves the current setting unchanged.
    display_rows:
        Maximum number of rows allowed before truncating to a compact head/tail
        view (five first and five last rows, when possible).  ``None`` leaves
        the current setting unchanged.
    display_precision:
        Number of digits after the decimal point for floating-point values in
        table displays.  Trailing zeros are trimmed.  ``None`` leaves the
        current setting unchanged.
    fancy:
        Whether to use the more decorated table display, including separator
        rules and a detailed footer.  ``False`` (default) uses a simpler
        pandas-like footer such as ``[726017 rows x 5 columns]`` and omits
        separator rules.  ``None`` leaves the current setting unchanged.
    """
    if display_index is not None:
        if not isinstance(display_index, bool):
            raise TypeError("display_index must be a bool or None")
        _CTABLE_PRINT_OPTIONS["display_index"] = display_index
    if display_rows is not None:
        if not isinstance(display_rows, int) or isinstance(display_rows, bool) or display_rows < 0:
            raise TypeError("display_rows must be a non-negative int or None")
        _CTABLE_PRINT_OPTIONS["display_rows"] = display_rows
    if display_precision is not None:
        if (
            not isinstance(display_precision, int)
            or isinstance(display_precision, bool)
            or display_precision < 0
        ):
            raise TypeError("display_precision must be a non-negative int or None")
        _CTABLE_PRINT_OPTIONS["display_precision"] = display_precision
    if fancy is not None:
        if not isinstance(fancy, bool):
            raise TypeError("fancy must be a bool or None")
        _CTABLE_PRINT_OPTIONS["fancy"] = fancy


def get_printoptions() -> dict[str, Any]:
    """Return a copy of the global :class:`CTable` display options."""
    return dict(_CTABLE_PRINT_OPTIONS)


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

    def __len__(self) -> int:
        return len(self.obj.info_items)

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


class RowTransformer:
    """Row-wise transformer for fixed-shape ndarray columns.

    A row transformer sees one table row at a time.  For a source column with
    physical shape ``(nrows, *item_shape)``, axes passed to reductions are axes
    within ``item_shape`` (so they are shifted by one for batch evaluation).
    """

    def __init__(
        self,
        source: str,
        *,
        selection=(),
        op: str | None = None,
        axis=None,
        ord=None,
    ) -> None:
        self.source = source
        self.selection = tuple(selection)
        self.op = op
        self.axis = axis
        self.ord = ord
        self.kind = "row_transformer"
        self.source_columns = [source]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        return RowTransformer(
            self.source,
            selection=(*self.selection, *key),
            op=self.op,
            axis=self.axis,
            ord=self.ord,
        )

    def _with_op(self, op: str, *, axis=None, ord=None):
        return RowTransformer(self.source, selection=self.selection, op=op, axis=axis, ord=ord)

    def sum(self, *, axis=None):
        return self._with_op("sum", axis=axis)

    def mean(self, *, axis=None):
        return self._with_op("mean", axis=axis)

    def min(self, *, axis=None):
        return self._with_op("min", axis=axis)

    def max(self, *, axis=None):
        return self._with_op("max", axis=axis)

    def argmin(self, *, axis=None):
        return self._with_op("argmin", axis=axis)

    def argmax(self, *, axis=None):
        return self._with_op("argmax", axis=axis)

    def norm(self, *, axis=None, ord=None):
        return self._with_op("norm", axis=axis, ord=ord)

    @staticmethod
    def _serialize_selector(selector):
        if isinstance(selector, slice):
            return {"kind": "slice", "start": selector.start, "stop": selector.stop, "step": selector.step}
        if selector is Ellipsis:
            return {"kind": "ellipsis"}
        if selector is None:
            return {"kind": "newaxis"}
        if isinstance(selector, (int, np.integer)):
            return {"kind": "int", "value": int(selector)}
        raise TypeError(f"Unsupported row-transformer selector {selector!r}")

    @staticmethod
    def _deserialize_selector(data):
        kind = data["kind"]
        if kind == "slice":
            return slice(data.get("start"), data.get("stop"), data.get("step"))
        if kind == "ellipsis":
            return Ellipsis
        if kind == "newaxis":
            return None
        if kind == "int":
            return int(data["value"])
        raise ValueError(f"Unsupported row-transformer selector kind {kind!r}")

    @staticmethod
    def _serialize_axis(axis):
        if isinstance(axis, tuple):
            return list(axis)
        return axis

    @staticmethod
    def _deserialize_axis(axis):
        if isinstance(axis, list):
            return tuple(axis)
        return axis

    def to_metadata(self) -> dict:
        meta = {
            "kind": "row_transformer",
            "source": self.source,
            "selection": [self._serialize_selector(s) for s in self.selection],
        }
        if self.op is not None:
            meta["op"] = self.op
            meta["axis"] = self._serialize_axis(self.axis)
            if self.ord is not None:
                meta["ord"] = self.ord
        return meta

    @classmethod
    def from_metadata(cls, meta: dict):
        return cls(
            meta["source"],
            selection=tuple(cls._deserialize_selector(s) for s in meta.get("selection", ())),
            op=meta.get("op"),
            axis=cls._deserialize_axis(meta.get("axis")),
            ord=meta.get("ord"),
        )

    def _row_axis_to_batch_axis(self, ndim: int, *, none_means_all_item: bool = False):
        axis = self.axis
        item_ndim = max(0, ndim - 1)
        if axis is None:
            return tuple(range(1, ndim)) if none_means_all_item and item_ndim else None

        def one(ax):
            ax = int(ax)
            if ax < 0:
                ax += item_ndim
            if not 0 <= ax < item_ndim:
                raise ValueError(f"axis {ax} is out of bounds for row item with {item_ndim} dimensions")
            return ax + 1

        if isinstance(axis, tuple):
            return tuple(one(ax) for ax in axis)
        return one(axis)

    def _apply_selection(self, arr: np.ndarray) -> np.ndarray:
        if not self.selection:
            return arr
        return arr[(slice(None), *self.selection)]

    def evaluate_batch(self, raw_columns: Mapping[str, Any]) -> np.ndarray:
        arr = np.asarray(raw_columns[self.source])
        if arr.ndim == 0:
            arr = arr.reshape((1,))
        arr = self._apply_selection(arr)
        if self.op is None:
            return np.asarray(arr)
        axis = self._row_axis_to_batch_axis(arr.ndim, none_means_all_item=True)
        if self.op == "sum":
            return np.asarray(np.sum(arr, axis=axis))
        if self.op == "mean":
            return np.asarray(np.mean(arr, axis=axis))
        if self.op == "min":
            return np.asarray(np.min(arr, axis=axis))
        if self.op == "max":
            return np.asarray(np.max(arr, axis=axis))
        if self.op == "argmin":
            if self.axis is None:
                return np.asarray(np.argmin(arr.reshape((arr.shape[0], -1)), axis=1), dtype=np.int64)
            return np.asarray(np.argmin(arr, axis=axis), dtype=np.int64)
        if self.op == "argmax":
            if self.axis is None:
                return np.asarray(np.argmax(arr.reshape((arr.shape[0], -1)), axis=1), dtype=np.int64)
            return np.asarray(np.argmax(arr, axis=axis), dtype=np.int64)
        if self.op == "norm":
            if self.axis is None:
                return np.asarray(np.linalg.norm(arr.reshape((arr.shape[0], -1)), ord=self.ord, axis=1))
            return np.asarray(np.linalg.norm(arr, ord=self.ord, axis=axis))
        raise ValueError(f"Unsupported row-transformer op {self.op!r}")

    def evaluate_row(self, row: Mapping[str, Any]):
        arr = np.asarray(row[self.source])
        if self.selection:
            arr = arr[self.selection]
        if self.op is None:
            return arr
        if self.op == "sum":
            return np.sum(arr, axis=self.axis)
        if self.op == "mean":
            return np.mean(arr, axis=self.axis)
        if self.op == "min":
            return np.min(arr, axis=self.axis)
        if self.op == "max":
            return np.max(arr, axis=self.axis)
        if self.op == "argmin":
            return np.asarray(np.argmin(arr, axis=self.axis), dtype=np.int64)
        if self.op == "argmax":
            return np.asarray(np.argmax(arr, axis=self.axis), dtype=np.int64)
        if self.op == "norm":
            return np.linalg.norm(arr, ord=self.ord, axis=self.axis)
        raise ValueError(f"Unsupported row-transformer op {self.op!r}")

    def evaluate_existing(self, table: CTable) -> np.ndarray:
        return self.evaluate_batch({self.source: table[self.source][:]})


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
    def is_generated(self) -> bool:
        """True if this column is a stored generated/materialized column."""
        return self._col_name in self._table._root_table._materialized_cols

    @property
    def is_stale(self) -> bool:
        """True if this generated column needs to be refreshed before use."""
        meta = self._table._root_table._materialized_cols.get(self._col_name)
        return bool(meta and meta.get("stale", False))

    def _ensure_not_stale(self) -> None:
        if self.is_stale:
            raise ValueError(
                f"Generated column {self._col_name!r} is stale because one or more source columns were "
                f"modified. Call refresh_generated_column({self._col_name!r}) before reading it, or use "
                f"t[{self._col_name!r}].read_stale() to explicitly read the last stored stale values."
            )

    def read_stale(self, key=slice(None)):
        """Read stored values even when this generated column is marked stale.

        This is an explicit escape hatch for inspecting the last materialized
        values.  Normal reads raise for stale generated columns so outdated
        values are not used accidentally.
        """
        return self._values_from_key(key, check_stale=False)

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
    def is_dictionary(self) -> bool:
        """True if this column is a dictionary-encoded string column."""
        col = self._table._schema.columns_by_name.get(self._col_name)
        return col is not None and isinstance(col.spec, DictionarySpec)

    @property
    def is_ndarray(self) -> bool:
        """True if this column stores fixed-shape N-D array values per row."""
        col = self._table._schema.columns_by_name.get(self._col_name)
        return col is not None and isinstance(col.spec, NDArraySpec)

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

    def _values_from_key(self, key, *, check_stale: bool = True):  # noqa: C901
        """Materialise values for a logical index key."""
        if check_stale:
            self._ensure_not_stale()
        if isinstance(key, tuple) and self.is_ndarray:
            if len(key) == 0:
                raise IndexError("empty tuple index is not valid for Column")
            row_key, inner_key = key[0], key[1:]
            values = self._values_from_key(row_key, check_stale=False)
            if not inner_key:
                return values
            if isinstance(row_key, (int, np.integer)) and not isinstance(row_key, (bool, np.bool_)):
                return values[inner_key]
            return values[(slice(None), *inner_key)]

        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            if self.is_dictionary:
                return self._raw_col[int(pos_true)]
            return self._maybe_decode_timestamp_values(self._raw_col[int(pos_true)])

        elif isinstance(key, slice):
            real_pos = np.where(self._valid_rows[:])[0]
            start, stop, step = key.indices(len(real_pos))
            if start >= stop:
                if self.is_list or self.is_varlen_scalar or self.is_dictionary:
                    return []
                if self.is_ndarray:
                    spec = self._table._schema.columns_by_name[self._col_name].spec
                    return np.empty((0, *spec.item_shape), dtype=self.dtype)
                return np.array([], dtype=self.dtype)
            selected_pos = real_pos[start:stop:step]  # physical row positions
            if self.is_computed:
                lo, hi = int(selected_pos.min()), int(selected_pos.max())
                chunk = np.asarray(self._raw_col[lo : hi + 1])
                return chunk[selected_pos - lo]
            if self.is_list or self.is_varlen_scalar or self.is_dictionary:
                return self._raw_col[selected_pos]
            return self._maybe_decode_timestamp_values(np.asarray(self._raw_col[selected_pos]))

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
            if self.is_list or self.is_varlen_scalar or self.is_dictionary:
                return self._raw_col[phys_indices]
            return self._maybe_decode_timestamp_values(self._raw_col[phys_indices])

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = np.where(self._valid_rows[:])[0]
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            if self.is_computed:
                raw_np = np.asarray(self._raw_col[:])
                return raw_np[phys_indices]
            if self.is_list or self.is_varlen_scalar or self.is_dictionary:
                return self._raw_col[phys_indices]
            return self._maybe_decode_timestamp_values(self._raw_col[phys_indices])

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
            if self.is_ndarray:
                spec = self._table._schema.columns_by_name[self._col_name].spec
                value = CTable._coerce_ndarray_value(self._col_name, spec, value)
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
                if self.is_ndarray:
                    spec = self._table._schema.columns_by_name[self._col_name].spec
                    value = CTable._coerce_ndarray_batch(self._col_name, spec, value, len(phys_indices))
                elif isinstance(value, (list, tuple)):
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
                if self.is_ndarray:
                    spec = self._table._schema.columns_by_name[self._col_name].spec
                    value = CTable._coerce_ndarray_batch(self._col_name, spec, value, len(phys_indices))
                elif isinstance(value, (list, tuple)):
                    value = np.array(value, dtype=self._raw_col.dtype)
                self._raw_col[phys_indices] = value

        else:
            raise TypeError(f"Invalid index type: {type(key)}")
        root = self._table._root_table
        root._mark_generated_columns_stale(self._col_name)
        root._mark_all_indexes_stale()

    def __iter__(self):
        """Iterate over live column values in insertion order, skipping deleted rows."""
        self._ensure_not_stale()
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

    @staticmethod
    def _format_array_value(value) -> str:
        arr = np.asarray(value)
        if arr.ndim == 1 and arr.size <= 6:
            return np.array2string(arr, separator=", ", max_line_width=10_000)
        return f"ndarray(shape={arr.shape}, dtype={arr.dtype})"

    def __repr__(self) -> str:
        preview_len = self._REPR_PREVIEW_ITEMS + 1
        if self.is_list:
            label = self._table._dtype_info_label(
                self.dtype, self._table._schema.columns_by_name[self._col_name].spec
            )
            preview_values = [f"<{label}>"] * min(len(self), preview_len)
        else:
            preview_pos = np.where(self._valid_rows[:])[0][:preview_len]
            if self.is_dictionary or self.is_varlen_scalar:
                preview_values = self._raw_col[preview_pos]
            elif len(preview_pos) == 0:
                preview_values = []
            else:
                preview_values = self._maybe_decode_timestamp_values(self._raw_col[preview_pos]).tolist()
        truncated = len(preview_values) > self._REPR_PREVIEW_ITEMS
        if truncated:
            preview_values = preview_values[: self._REPR_PREVIEW_ITEMS]

        if self.is_ndarray and preview_values:
            preview_items = [self._format_array_value(value) for value in preview_values]
            if truncated:
                preview_items.append("...")
            preview = ", ".join(preview_items)
        elif self.dtype is not None and self.dtype.kind in "biufc" and preview_values:
            arr = np.asarray(preview_values, dtype=self.dtype)
            preview = np.array2string(arr, separator=", ", max_line_width=10_000)[1:-1]
            if truncated:
                preview = f"{preview}, ..." if preview else "..."
        else:
            preview_items = []
            for value in preview_values:
                if isinstance(value, np.generic):
                    value = value.item()
                preview_items.append(repr(value))
            if truncated:
                preview_items.append("...")
            preview = ", ".join(preview_items)

        return f"Column({self._col_name!r}, dtype={self.dtype}, len={len(self)}, values=[{preview}])"

    def __len__(self):
        """Return the number of live (non-deleted) values in this column."""
        return blosc2.count_nonzero(self._valid_rows)

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical shape of the live column values."""
        if self.is_ndarray:
            spec = self._table._schema.columns_by_name[self._col_name].spec
            return (len(self), *spec.item_shape)
        return (len(self),)

    def summary(self) -> str:
        """Return and print a compact summary for this column.

        For fixed-shape ndarray columns this includes logical shape, storage, and
        row-norm statistics when numeric.  Scalar columns fall back to ``info``.
        """
        if not self.is_ndarray:
            text = str(self.info)
            print(text)
            return text
        raw = self._raw_col
        rows = len(self)
        capacity = raw.shape[0] if hasattr(raw, "shape") else len(self._table._valid_rows)
        lines = [
            f"ndarray column {self._col_name!r}",
            f"  rows       : {rows:,} live / {capacity:,} capacity",
            f"  item_shape : {self.item_shape}",
            f"  dtype      : {self.dtype}",
            f"  storage    : NDArray shape={getattr(raw, 'shape', None)}, chunks={getattr(raw, 'chunks', None)}, blocks={getattr(raw, 'blocks', None)}",
        ]
        cbytes = getattr(raw, "cbytes", None)
        if cbytes is not None:
            lines.append(f"  cbytes     : {format_nbytes_info(cbytes)}")
        if rows and self.dtype is not None and self.dtype.kind in "biufc":
            flat = np.asarray(self[:]).reshape(rows, -1)
            norms = np.linalg.norm(flat, axis=1)
            lines.append(
                "  row stats  : "
                f"min(norm(axis=1))={norms.min():.6g}, "
                f"mean(norm(axis=1))={norms.mean():.6g}, "
                f"max(norm(axis=1))={norms.max():.6g}"
            )
        text = "\n".join(lines)
        print(text)
        return text

    @property
    def info(self) -> _CTableInfoReporter:
        """Get information about this column.

        The report includes both logical/live-row details and, when available,
        the physical storage details used internally by lazy predicates.

        Examples
        --------
        >>> print(t["score"].info)
        >>> t["score"].info()
        """
        return _CTableInfoReporter(self)

    @property
    def info_items(self) -> list[tuple[str, object]]:
        """Structured summary items used by :attr:`info`."""
        raw = self._raw_col
        table = self._table
        col_meta = table._schema.columns_by_name.get(self._col_name)
        spec = col_meta.spec if col_meta is not None else None
        chunks = getattr(raw, "chunks", None)
        blocks = getattr(raw, "blocks", None)
        items: list[tuple[str, object]] = [
            ("type", self.__class__.__name__),
            ("name", self._col_name),
            ("nrows", len(self)),
            ("shape", self.shape),
        ]
        if chunks is not None:
            items.append(("chunks", chunks))
        if blocks is not None:
            items.append(("blocks", blocks))
        items.extend(
            [
                ("dtype", table._dtype_info_label(self.dtype, spec)),
                ("computed", self.is_computed),
                ("nullable", self.null_value is not None or getattr(spec, "nullable", False)),
            ]
        )

        if self.is_list:
            items.append(("storage", "list"))
        elif self.is_varlen_scalar:
            items.append(("storage", "variable-length scalar"))
        elif self.is_dictionary:
            items.append(("storage", "dictionary"))
            items.append(("dictionary_size", len(raw.dictionary)))
        else:
            items.append(("storage", "ndarray" if isinstance(raw, blosc2.NDArray) else type(raw).__name__))

        nbytes = getattr(raw, "nbytes", None)
        cbytes = getattr(raw, "cbytes", None)
        cratio = getattr(raw, "cratio", None)
        if nbytes is not None:
            items.append(("nbytes", format_nbytes_info(nbytes)))
        if cbytes is not None:
            items.append(("cbytes", format_nbytes_info(cbytes)))
        if cratio is not None:
            items.append(("cratio", f"{cratio:.2f}"))

        urlpath = getattr(raw, "urlpath", None)
        if urlpath is not None:
            items.append(("urlpath", urlpath))
        cparams = getattr(raw, "cparams", None)
        dparams = getattr(raw, "dparams", None)
        if cparams is not None:
            items.append(("cparams", cparams))
        if dparams is not None:
            items.append(("dparams", dparams))
        return items

    @property
    def item_shape(self) -> tuple[int, ...]:
        """Per-row item shape; ``()`` for scalar columns."""
        if self.is_ndarray:
            return tuple(self._table._schema.columns_by_name[self._col_name].spec.item_shape)
        return ()

    @property
    def item_ndim(self) -> int:
        """Number of per-row item dimensions."""
        return len(self.item_shape)

    @property
    def item_size(self) -> int:
        """Number of scalar values stored in each row item."""
        return int(np.prod(self.item_shape, dtype=np.int64)) if self.item_shape else 1

    @property
    def ndim(self) -> int:
        """Number of logical dimensions."""
        return 1 + self.item_ndim

    @property
    def size(self) -> int:
        """Number of live scalar values in the logical column array."""
        return len(self) * self.item_size

    @property
    def row_transformer(self) -> RowTransformer:
        """Build row-wise projections/reductions for generated columns."""
        if not self.is_ndarray:
            raise TypeError(f"Column {self._col_name!r} is not a fixed-shape ndarray column.")
        return RowTransformer(self._col_name)

    def _ensure_queryable(self) -> None:
        self._ensure_not_stale()
        if self.is_varlen_scalar:
            raise NotImplementedError(
                f"Column {self._col_name!r} is a vlstring/vlbytes column; "
                "lazy expressions and vectorized comparisons are not supported yet."
            )
        if self.is_dictionary:
            raise NotImplementedError(
                f"Column {self._col_name!r} is a dictionary column; "
                "use == and isin() for dictionary column comparisons."
            )

    def _raise_ndarray_compare(self) -> None:
        raise TypeError(
            f"Cannot compare ndarray column {self._col_name!r} directly; the result would not be a "
            "1-D row mask. Use an element projection like t.embedding[:, 0] > 0.5 or an "
            "axis-aware reduction like t.embedding.max(axis=1) > 0.5."
        )

    def _ensure_comparable(self) -> None:
        self._ensure_queryable()
        if self.is_ndarray:
            self._raise_ndarray_compare()

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

    @property
    def _timestamp_spec(self):
        col = self._table._schema.columns_by_name.get(self._col_name)
        return col.spec if col is not None and isinstance(col.spec, timestamp) else None

    def _maybe_decode_timestamp_values(self, values):
        spec = self._timestamp_spec
        if spec is None:
            return values
        if np.isscalar(values):
            return np.datetime64(int(values), spec.unit)
        return np.asarray(values).astype(f"datetime64[{spec.unit}]")

    def _coerce_timestamp_operand(self, other):
        spec = self._timestamp_spec
        if isinstance(other, Column) and other.is_ndarray:
            other._raise_ndarray_compare()
        other = self._unwrap_operand(other)
        if spec is None:
            return other
        if isinstance(other, np.datetime64):
            return other.astype(f"datetime64[{spec.unit}]").astype(np.int64)
        if isinstance(other, str):
            return np.datetime64(other).astype(f"datetime64[{spec.unit}]").astype(np.int64)
        if hasattr(other, "isoformat"):
            return np.datetime64(other).astype(f"datetime64[{spec.unit}]").astype(np.int64)
        if isinstance(other, np.ndarray) and np.issubdtype(other.dtype, np.datetime64):
            return other.astype(f"datetime64[{spec.unit}]").astype(np.int64)
        return other

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
        self._ensure_comparable()
        return self._raw_col < self._coerce_timestamp_operand(other)

    def __le__(self, other):
        self._ensure_comparable()
        return self._raw_col <= self._coerce_timestamp_operand(other)

    def __eq__(self, other):
        if self.is_dictionary:
            return self._dictionary_eq(other)
        self._ensure_comparable()
        if self._is_nullable_bool and isinstance(other, (bool, np.bool_)):
            return self._raw_col == int(other)
        return self._raw_col == self._coerce_timestamp_operand(other)

    def __ne__(self, other):
        if self.is_dictionary:
            result = self._dictionary_eq(other)
            if isinstance(result, np.ndarray):
                return ~result
            return ~np.asarray(result, dtype=bool)
        self._ensure_comparable()
        if self._is_nullable_bool and isinstance(other, (bool, np.bool_)):
            return self._raw_col == int(not other)
        return self._raw_col != self._coerce_timestamp_operand(other)

    def _dictionary_eq(self, other):
        """Return a physical-slot boolean predicate for dictionary equality.

        Regular fixed-width columns build predicates against their raw physical
        arrays, whose length is the table slot capacity.  Dictionary predicates
        need to use the same coordinate system so they can be combined with
        regular predicates before aggregate/view code intersects them with
        ``_valid_rows``.
        """
        dc = self._raw_col  # DictionaryColumn
        spec = self._table._schema.columns_by_name[self._col_name].spec
        if other is None:
            target_code = spec.null_code
        elif isinstance(other, str):
            try:
                target_code = dc.value_to_code(other)
            except KeyError:
                return blosc2.zeros(len(self._table._valid_rows), dtype=np.bool_)
        else:
            raise TypeError(
                f"Dictionary column {self._col_name!r} can only be compared with str or None, "
                f"got {type(other).__name__!r}."
            )
        pred = dc.codes == np.int32(target_code)
        valid = self._lazy_valid_rows()
        if len(dc.codes) != len(self._table._valid_rows):
            physical = blosc2.zeros(len(self._table._valid_rows), dtype=np.bool_)
            physical[: len(dc.codes)] = pred
            pred = physical
        return pred & valid

    def isin(self, values) -> np.ndarray:
        """Return a boolean array True where the live value is in *values*.

        For dictionary columns this performs efficient integer-code membership
        testing (no decoding of all values).  Values absent from the
        dictionary are treated as not-present.

        For non-dictionary columns this decodes all live values and tests
        membership in a set.
        """
        if self.is_dictionary:
            return self._dictionary_isin(values)
        live_values = self[:]
        test_set = set(values)
        if isinstance(live_values, np.ndarray):
            return np.array([v in test_set for v in live_values.tolist()], dtype=bool)
        return np.array([v in test_set for v in live_values], dtype=bool)

    def _dictionary_isin(self, values) -> np.ndarray:
        """Return a boolean array for in-membership tests against a dictionary column."""
        dc = self._raw_col  # DictionaryColumn
        spec = self._table._schema.columns_by_name[self._col_name].spec
        valid = self._valid_rows
        live_pos = np.where(valid[:])[0]
        if len(live_pos) == 0:
            return np.zeros(0, dtype=bool)
        # Map requested values to codes, ignoring absent values.
        target_codes: set[int] = set()
        for v in values:
            if v is None:
                target_codes.add(spec.null_code)
            elif isinstance(v, str):
                with contextlib.suppress(KeyError):
                    target_codes.add(dc.value_to_code(v))
        if not target_codes:
            return np.zeros(len(live_pos), dtype=bool)
        live_codes = dc.codes[live_pos]
        mask = np.zeros(len(live_codes), dtype=bool)
        for code in target_codes:
            mask |= live_codes == np.int32(code)
        return mask

    def __gt__(self, other):
        self._ensure_comparable()
        return self._raw_col > self._coerce_timestamp_operand(other)

    def __ge__(self, other):
        self._ensure_comparable()
        return self._raw_col >= self._coerce_timestamp_operand(other)

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
        self._ensure_not_stale()
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
            root = self._table._root_table
            root._mark_generated_columns_stale(self._col_name)
            root._mark_all_indexes_stale()
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
        root = self._table._root_table
        root._mark_generated_columns_stale(self._col_name)
        root._mark_all_indexes_stale()

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
        arr = np.asarray(arr)
        if self.is_ndarray:
            if arr.ndim <= self.item_ndim:
                arr = arr.reshape((1, *arr.shape))
            if isinstance(nv, (float, np.floating)) and np.isnan(nv):
                elem_mask = np.isnan(arr)
            else:
                elem_mask = arr == nv
            inner_axes = tuple(range(1, elem_mask.ndim))
            return elem_mask.all(axis=inner_axes) if inner_axes else elem_mask.astype(np.bool_)
        if isinstance(nv, (float, np.floating)) and np.isnan(nv):
            return np.isnan(arr)
        return arr == nv

    def is_null(self) -> np.ndarray:
        """Return a boolean array True where the live value is the null sentinel.

        For varlen scalar columns (vlstring/vlbytes) nullability is represented
        as native ``None`` values, so this returns True wherever the value is
        ``None``.  For dictionary columns, returns True where the code equals
        the null_code (``-1`` by default).
        """
        if self.is_dictionary:
            return self._dictionary_eq(None)
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
        if self.is_dictionary:
            return int(self.is_null().sum())
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
        self._ensure_not_stale()
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
            operands = self._table._where_expression_operands()
            where, operands = self._table._rewrite_nested_expression(where, operands)
            where = blosc2.lazyexpr(where, operands)
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
        """Build a lazy visible-row mask, optionally intersected with non-null values.

        When all physical rows are visible, avoid injecting ``_valid_rows`` into
        the expression.  This keeps aggregate predicates aligned with the data
        columns, which lets the miniexpr reduction fast path run for common
        no-deletes/no-filtered-view cases.
        """
        raw = self._raw_col
        if not isinstance(raw, (blosc2.NDArray, blosc2.LazyExpr)):
            return NotImplemented

        table_n_rows = self._table._known_n_rows()
        all_rows_visible = (
            self._mask is None and table_n_rows is not None and table_n_rows == len(self._table._valid_rows)
        )
        mask = None if all_rows_visible else self._lazy_valid_rows()
        if where is not None:
            mask = where if mask is None else mask & where
        nv = self.null_value
        if nv is not None:
            if isinstance(nv, (float, np.floating)) and np.isnan(nv):
                nonnull = ~blosc2.isnan(raw)
            else:
                nonnull = raw != nv
            mask = nonnull if mask is None else mask & nonnull
        return mask

    def _sum_lazy_fastpath(self, acc_dtype, where=None, *, jit=None, jit_backend=None):
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
            and self._table.nrows / total_rows < 0.25
        ):
            return NotImplemented

        mask = self._lazy_nonnull_mask(where=where)
        if mask is NotImplemented:
            return NotImplemented

        try:
            if mask is None:
                return raw.sum(dtype=acc_dtype, jit=jit, jit_backend=jit_backend)
            force_miniexpr = jit is True or jit_backend is not None
            if force_miniexpr and isinstance(raw, blosc2.NDArray):
                zero = blosc2.zeros(
                    raw.shape, dtype=np.dtype(acc_dtype), chunks=raw.chunks, blocks=raw.blocks
                )
            else:
                zero = acc_dtype(0)
            return blosc2.where(mask, raw, zero).sum(dtype=acc_dtype, jit=jit, jit_backend=jit_backend)
        except Exception:
            return NotImplemented

    def _ndarray_values_for_reduction(self, where=None) -> np.ndarray:
        arr = np.asarray(self[:])
        null_mask = self._null_mask_for(arr) if self.null_value is not None else None
        if null_mask is not None and null_mask.any():
            arr = arr[~null_mask]
        if where is None:
            return arr
        where = self._normalize_sum_where(where)
        mask = where.compute() if isinstance(where, blosc2.LazyExpr) else where[:]
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1:
            raise ValueError("Column reduction where= must be a 1-D row mask.")
        if len(mask) != len(self._table._valid_rows):
            if len(mask) != len(self):
                raise ValueError(
                    f"Column reduction where= mask length {len(mask)} does not match live rows {len(self)}."
                )
            if null_mask is not None and len(null_mask) == len(mask):
                mask = mask[~null_mask]
            return arr[mask]
        live_pos = np.where(self._valid_rows[:])[0]
        row_mask = mask[live_pos]
        if null_mask is not None and len(null_mask) == len(row_mask):
            row_mask = row_mask[~null_mask]
        return arr[row_mask]

    def _ndarray_reduce(self, op: str, *, axis=None, dtype=None, where=None, ddof: int = 0):
        arr = self._ndarray_values_for_reduction(where=where)
        if op == "sum":
            return np.sum(arr, axis=axis, dtype=dtype)
        if op == "mean":
            return np.mean(arr, axis=axis, dtype=dtype)
        if op == "min":
            return np.min(arr, axis=axis)
        if op == "max":
            return np.max(arr, axis=axis)
        if op == "argmin":
            return np.argmin(arr, axis=axis)
        if op == "argmax":
            return np.argmax(arr, axis=axis)
        if op == "std":
            return np.std(arr, axis=axis, ddof=ddof, dtype=dtype)
        raise ValueError(f"Unsupported ndarray reduction {op!r}")

    def sum(self, dtype=None, axis=None, *, where=None, jit=None, jit_backend=None):
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
        jit:
            Optional miniexpr JIT policy passed to the lazy reduction engine.
        jit_backend:
            Optional miniexpr JIT backend. Use ``"tcc"`` or ``"cc"``.

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
        if self.is_ndarray:
            self._require_kind("biufc", "sum")
            return self._ndarray_reduce("sum", axis=axis, dtype=dtype, where=where)
        if axis not in (None, 0):
            return np.sum(self[:], axis=axis, dtype=dtype)
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

        result = self._sum_lazy_fastpath(acc_dtype, where=where, jit=jit, jit_backend=jit_backend)
        if result is NotImplemented:
            if where is not None:
                return self._table.where(where)[self._col_name].sum(
                    dtype=dtype, jit=jit, jit_backend=jit_backend
                )
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
            if op in {"min", "max"} and mask is not None:
                count = int(mask.where(blosc2.ones(raw.shape, dtype=np.int64), 0).sum(dtype=np.int64))
                if count == 0:
                    raise ValueError(f"{op}() called on a column where all values are null.")
            if op == "mean":
                return float(
                    raw.mean(dtype=dtype or np.float64)
                    if mask is None
                    else raw.mean(where=mask, dtype=dtype or np.float64)
                )
            if op == "std":
                return float(
                    raw.std(dtype=dtype or np.float64, ddof=ddof)
                    if mask is None
                    else raw.std(where=mask, dtype=dtype or np.float64, ddof=ddof)
                )
            if op == "min":
                return raw.min() if mask is None else raw.min(where=mask)
            if op == "max":
                return raw.max() if mask is None else raw.max(where=mask)
        except ValueError:
            if op in {"mean", "std"}:
                return float("nan")
            raise
        except Exception:
            return NotImplemented
        return NotImplemented

    def min(self, axis=None, *, where=None):
        """Minimum live, non-null value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        """
        if self.is_ndarray:
            self._require_kind("biuf", "min")
            return self._ndarray_reduce("min", axis=axis, where=where)
        if axis not in (None, 0):
            return np.min(self[:], axis=axis)
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

    def max(self, axis=None, *, where=None):
        """Maximum live, non-null value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        """
        if self.is_ndarray:
            self._require_kind("biuf", "max")
            return self._ndarray_reduce("max", axis=axis, where=where)
        if axis not in (None, 0):
            return np.max(self[:], axis=axis)
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

    def argmin(self, axis=None, *, where=None):
        """Index of the minimum live, non-null value.

        For fixed-shape ndarray columns, this follows NumPy axis semantics on
        the logical array of shape ``(nrows, *item_shape)``.  For scalar
        columns, the result is the logical row position within this column (or
        filtered view).
        """
        if self.is_ndarray:
            self._require_kind("biuf", "argmin")
            return self._ndarray_reduce("argmin", axis=axis, where=where)
        if axis not in (None, 0):
            return np.argmin(self[:], axis=axis)
        self._require_kind("biuf", "argmin")
        if where is not None:
            return self._table.where(self._normalize_sum_where(where))[self._col_name].argmin()
        arr = np.asarray(self[:])
        if arr.size == 0:
            raise ValueError("argmin() called on an empty column.")
        mask = (
            self._null_mask_for(arr) if self.null_value is not None else np.zeros(len(arr), dtype=np.bool_)
        )
        if mask.all():
            raise ValueError("argmin() called on a column where all values are null.")
        positions = np.where(~mask)[0]
        return int(positions[np.argmin(arr[positions])])

    def argmax(self, axis=None, *, where=None):
        """Index of the maximum live, non-null value.

        For fixed-shape ndarray columns, this follows NumPy axis semantics on
        the logical array of shape ``(nrows, *item_shape)``.  For scalar
        columns, the result is the logical row position within this column (or
        filtered view).
        """
        if self.is_ndarray:
            self._require_kind("biuf", "argmax")
            return self._ndarray_reduce("argmax", axis=axis, where=where)
        if axis not in (None, 0):
            return np.argmax(self[:], axis=axis)
        self._require_kind("biuf", "argmax")
        if where is not None:
            return self._table.where(self._normalize_sum_where(where))[self._col_name].argmax()
        arr = np.asarray(self[:])
        if arr.size == 0:
            raise ValueError("argmax() called on an empty column.")
        mask = (
            self._null_mask_for(arr) if self.null_value is not None else np.zeros(len(arr), dtype=np.bool_)
        )
        if mask.all():
            raise ValueError("argmax() called on a column where all values are null.")
        positions = np.where(~mask)[0]
        return int(positions[np.argmax(arr[positions])])

    def mean(self, axis=None, *, where=None):
        """Arithmetic mean of all live, non-null values.

        Supported dtypes: bool, int, uint, float.
        Null sentinel values are skipped. When *where* is provided, only rows
        matching the boolean predicate are included.
        Always returns a Python float.
        """
        if self.is_ndarray:
            self._require_kind("biuf", "mean")
            return self._ndarray_reduce("mean", axis=axis, where=where)
        if axis not in (None, 0):
            return np.mean(self[:], axis=axis)
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

    def std(self, ddof: int = 0, axis=None, *, where=None):
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
        if self.is_ndarray:
            self._require_kind("biuf", "std")
            return self._ndarray_reduce("std", axis=axis, where=where, ddof=ddof)
        if axis not in (None, 0):
            return np.std(self[:], axis=axis, ddof=ddof)
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

    def norm(self, ord=None, axis=None, *, where=None):
        """Vector/matrix norm of a fixed-shape ndarray column.

        The column is treated as a logical array of shape ``(nrows, *item_shape)``.
        For example, ``axis=1`` computes one norm per row for a 1-D item shape.
        """
        if not self.is_ndarray:
            raise TypeError(f"Column.norm() is only supported for ndarray columns, got {self._col_name!r}.")
        self._require_kind("biuf", "norm")
        arr = self._ndarray_values_for_reduction(where=where)
        return np.linalg.norm(arr, ord=ord, axis=axis)

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


class _StructPathColumn:
    """Virtual read-only column representing a struct prefix path.

    Values are reconstructed per row from descendant dotted leaf columns.
    """

    def __init__(self, table: CTable, prefix: str, leaves: list[str]):
        self._table = table
        self._prefix = prefix
        self._leaves = list(leaves)

    def _leaf_is_null_at_logical(self, leaf: str, idx: int) -> bool:
        col = self._table[leaf]
        v = col[idx]
        nv = col.null_value
        if nv is None:
            return v is None
        try:
            return bool(col._null_mask_for(np.asarray([v]))[0])
        except Exception:
            return v is None

    def _row_value_at_logical(self, idx: int):
        # If every descendant leaf is null at this row, represent the struct as None.
        if self._leaves and all(self._leaf_is_null_at_logical(leaf, idx) for leaf in self._leaves):
            return None
        prefix_parts = split_field_path(self._prefix)
        result: dict[str, Any] = {}
        for leaf in self._leaves:
            parts = split_field_path(leaf)
            rel_parts = parts[len(prefix_parts) :]
            if not rel_parts:
                continue
            node = result
            for part in rel_parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[rel_parts[-1]] = self._table._normalize_scalar_value(self._table[leaf][idx])
        return result

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._row_value_at_logical(key)
        if isinstance(key, slice):
            start, stop, step = key.indices(self._table.nrows)
            return [self._row_value_at_logical(i) for i in range(start, stop, step)]
        if isinstance(key, (list, np.ndarray)):
            if len(key) == 0:
                return []
            if isinstance(key, np.ndarray) and key.dtype == np.bool_:
                idxs = np.where(key)[0]
            elif isinstance(key[0], (bool, np.bool_)):
                idxs = [i for i, v in enumerate(key) if v]
            else:
                idxs = [int(i) for i in key]
            return [self._row_value_at_logical(i) for i in idxs]
        raise TypeError(f"Invalid index type: {type(key)}")

    def __iter__(self):
        for i in range(self._table.nrows):
            yield self._row_value_at_logical(i)


class _NestedColumnNamespace:
    """Attribute proxy for dotted nested column paths.

    Allows `t.trip.begin.lon` when the physical leaf column is named
    `"trip.begin.lon"`.
    """

    def __init__(self, table: CTable, prefix: str):
        self._table = table
        self._prefix = prefix

    def _descendant_col_names(self) -> list[str]:
        prefix_parts = split_field_path(self._prefix)
        return [
            name
            for name in self._table.col_names
            if (parts := split_field_path(name))[: len(prefix_parts)] == prefix_parts
            and len(parts) > len(prefix_parts)
        ]

    def _relative_col_name(self, name: str) -> str:
        prefix_parts = split_field_path(self._prefix)
        return join_field_path(split_field_path(name)[len(prefix_parts) :])

    @property
    def col_names(self) -> list[str]:
        """Descendant leaf column names relative to this nested prefix."""
        return [self._relative_col_name(name) for name in self._descendant_col_names()]

    @property
    def nrows(self) -> int:
        """Number of logical rows in this nested namespace."""
        return self._table.nrows

    @property
    def ncols(self) -> int:
        """Number of descendant leaf columns in this nested namespace."""
        return len(self._descendant_col_names())

    @property
    def nbytes(self) -> int:
        """Uncompressed size in bytes for stored descendant columns."""
        return sum(
            getattr(self._table._cols[name], "nbytes", 0)
            for name in self._descendant_col_names()
            if name in self._table._cols
        )

    @property
    def cbytes(self) -> int:
        """Compressed size in bytes for stored descendant columns."""
        return sum(
            getattr(self._table._cols[name], "cbytes", 0)
            for name in self._descendant_col_names()
            if name in self._table._cols
        )

    @property
    def cratio(self) -> float:
        """Compression ratio for stored descendant columns."""
        if self.cbytes == 0:
            return float("inf")
        return self.nbytes / self.cbytes

    @property
    def info_items(self) -> list[tuple[str, object]]:
        """Structured summary items used by :attr:`info`."""
        table = self._table
        storage_type = "persistent" if isinstance(table._storage, FileTableStorage) else "in-memory"
        schema_summary = {}
        for name in self._descendant_col_names():
            rel_name = self._relative_col_name(name)
            if name in table._computed_cols:
                cc = table._computed_cols[name]
                schema_summary[rel_name] = _InfoLiteral(
                    f"{cc['dtype']} (computed: {table._readable_computed_expr(cc)})"
                )
            else:
                col_meta = table._schema.columns_by_name.get(name)
                schema_summary[rel_name] = _InfoLiteral(
                    table._dtype_info_label(
                        getattr(table._cols[name], "dtype", None), col_meta.spec if col_meta else None
                    )
                )

        return [
            ("type", self.__class__.__name__),
            ("storage", storage_type),
            ("nrows", self.nrows),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.1f}x"),
            ("schema", schema_summary),
        ]

    @property
    def info(self) -> _CTableInfoReporter:
        """Get information about this nested column namespace.

        Examples
        --------
        >>> print(t.trip.info)
        >>> t.trip.info()
        """
        return _CTableInfoReporter(self)

    def __getattr__(self, name: str):
        path = join_field_path((*split_field_path(self._prefix), name))
        if path in self._table._cols or path in self._table._computed_cols:
            return Column(self._table, path)
        path_parts = split_field_path(path)
        for col_name in self._table.col_names:
            parts = split_field_path(col_name)
            if parts[: len(path_parts)] == path_parts and len(parts) > len(path_parts):
                return _NestedColumnNamespace(self._table, path)
        raise AttributeError(path)

    def __repr__(self) -> str:
        return f"<NestedColumnNamespace {self._prefix!r}>"


class _LazyColumnDict(dict):
    """Dict-like column cache that opens persistent columns on first use.

    Persistent CTables can be wide, and opening every stored column eagerly is
    expensive for workloads that touch only a small subset of columns, e.g.
    ``blosc2.open(path).trip.km.sum()`` on a nested table.  Keep the public and
    internal ``_cols`` access pattern mostly unchanged while deferring each
    ``storage.open_*_column()`` call until that column is actually requested.

    Methods that logically need all materialized columns, such as ``items()``
    and ``values()``, force-load the cache for compatibility with normal
    ``dict`` usage.  Name-oriented operations, such as ``keys()``, iteration,
    ``len()``, and ``in``, operate from the schema column list without opening
    the column payloads.
    """

    def __init__(self, table: CTable, storage: TableStorage, col_names: list[str]):
        super().__init__()
        self._table = table
        self._storage = storage
        self._col_names = list(col_names)
        self._available = set(col_names)

    def _load(self, name: str):
        if name not in self._available:
            raise KeyError(name)
        if not dict.__contains__(self, name):
            dict.__setitem__(self, name, self._table._open_column_from_storage(self._storage, name))
        return dict.__getitem__(self, name)

    def _load_all(self) -> None:
        for name in self._col_names:
            self._load(name)

    def __getitem__(self, name: str):
        return self._load(name)

    def get(self, name: str, default=None):
        return self._load(name) if name in self._available else default

    def __contains__(self, name: object) -> bool:
        return name in self._available

    def __iter__(self):
        return iter(self._col_names)

    def __len__(self) -> int:
        return len(self._col_names)

    def keys(self):
        return dict.fromkeys(self._col_names).keys()

    def items(self):
        self._load_all()
        return dict.items(self)

    def values(self):
        self._load_all()
        return dict.values(self)

    def __setitem__(self, name: str, value) -> None:
        if name not in self._available:
            self._available.add(name)
            self._col_names.append(name)
        dict.__setitem__(self, name, value)

    def __delitem__(self, name: str) -> None:
        self._available.remove(name)
        self._col_names.remove(name)
        if dict.__contains__(self, name):
            dict.__delitem__(self, name)


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

    @property
    def _n_rows(self) -> int:
        """Number of live rows, computed lazily for reopened tables."""
        n_rows = getattr(self, "_n_rows_cached", None)
        if n_rows is None:
            n_rows = int(blosc2.count_nonzero(self._valid_rows))
            self._n_rows_cached = n_rows
        return n_rows

    @_n_rows.setter
    def _n_rows(self, value: int | None) -> None:
        self._n_rows_cached = value

    def _known_n_rows(self) -> int | None:
        """Return cached live-row count without triggering a scan."""
        return getattr(self, "_n_rows_cached", None)

    def _iter_live_positions_chunks(self):
        """Yield chunks of physical positions for live rows without materialising the full mask."""
        valid_rows = self._valid_rows
        n = len(valid_rows)
        chunks = getattr(valid_rows, "chunks", None)
        chunk_len = chunks[0] if chunks else n

        for start in range(0, n, chunk_len):
            stop = min(start + chunk_len, n)
            local_pos = np.flatnonzero(valid_rows[start:stop])
            if len(local_pos):
                yield (local_pos + start).astype(np.intp, copy=False)

    def _live_positions_from_valid_rows_chunks(self) -> np.ndarray:
        """Return live physical row positions by scanning the validity NDArray chunk-wise."""
        positions = list(self._iter_live_positions_chunks())
        if not positions:
            return np.empty(0, dtype=np.intp)
        if len(positions) == 1:
            return positions[0]
        return np.concatenate(positions).astype(np.intp, copy=False)

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
            self._cols = _LazyColumnDict(self, storage, self.col_names)
            for name in self.col_names:
                cc = self._schema.columns_by_name[name]
                self._col_widths[name] = max(len(name), cc.display_width)
            self._n_rows = None
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
    def _is_dictionary_column(col: CompiledColumn) -> bool:
        return isinstance(col.spec, DictionarySpec)

    @staticmethod
    def _is_ndarray_column(col: CompiledColumn) -> bool:
        return isinstance(col.spec, NDArraySpec)

    @staticmethod
    def _column_physical_shape(col: CompiledColumn, capacity: int) -> tuple[int, ...]:
        if CTable._is_ndarray_column(col):
            return (capacity, *col.spec.item_shape)
        return (capacity,)

    @staticmethod
    def _ndarray_null_item(spec: NDArraySpec) -> np.ndarray:
        null_value = getattr(spec, "null_value", None)
        if null_value is None:
            raise TypeError("NDArraySpec is not nullable")
        return np.full(spec.item_shape, null_value, dtype=spec.dtype)

    @staticmethod
    def _coerce_ndarray_value(name: str, spec: NDArraySpec, val) -> np.ndarray:
        if val is None:
            if getattr(spec, "null_value", None) is None:
                raise TypeError(f"Column {name!r} is not nullable; received None.")
            return CTable._ndarray_null_item(spec)
        arr = np.asarray(val, dtype=spec.dtype)
        if arr.shape != spec.item_shape:
            raise ValueError(f"Column {name!r}: expected item shape {spec.item_shape}, got {arr.shape}")
        return np.ascontiguousarray(arr)

    @staticmethod
    def _coerce_ndarray_batch(name: str, spec: NDArraySpec, values, nrows: int) -> np.ndarray:
        if values is None:
            null_item = CTable._coerce_ndarray_value(name, spec, None)
            return np.broadcast_to(null_item, (nrows, *spec.item_shape)).copy()
        if isinstance(values, np.ndarray) and values.dtype != object:
            arr = np.ascontiguousarray(values, dtype=spec.dtype)
            if arr.ndim == len(spec.item_shape):
                arr = arr.reshape((1, *arr.shape))
            if arr.shape != (nrows, *spec.item_shape):
                raise ValueError(
                    f"Column {name!r}: expected batch shape {(nrows, *spec.item_shape)}, got {arr.shape}"
                )
            return arr
        rows = [CTable._coerce_ndarray_value(name, spec, value) for value in values]
        arr = np.ascontiguousarray(rows, dtype=spec.dtype)
        if arr.shape != (nrows, *spec.item_shape):
            raise ValueError(
                f"Column {name!r}: expected batch shape {(nrows, *spec.item_shape)}, got {arr.shape}"
            )
        return arr

    @staticmethod
    def _column_chunks_blocks(col: CompiledColumn, shape: tuple[int, ...]):
        return compute_chunks_blocks(shape, dtype=col.dtype)

    @staticmethod
    def _is_list_spec(spec: SchemaSpec) -> bool:
        return isinstance(spec, ListSpec)

    @staticmethod
    def _policy_null_value_for_spec(spec: SchemaSpec, policy: NullPolicy):
        if isinstance(spec, NDArraySpec):
            dtype = spec.dtype
            if dtype == np.dtype(np.bool_):
                return policy.bool_value
            if dtype.kind == "i":
                info = np.iinfo(dtype)
                return info.min if policy.signed_int_strategy == "min" else info.max
            if dtype.kind == "u":
                info = np.iinfo(dtype)
                return info.min if policy.unsigned_int_strategy == "min" else info.max
            if dtype.kind == "f":
                return policy.float_value
            return None
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
        if isinstance(spec, timestamp):
            return policy.timestamp_value
        return None

    @staticmethod
    def _validate_null_value_for_spec(name: str, spec: SchemaSpec, null_value) -> None:  # noqa: C901
        if isinstance(spec, NDArraySpec):
            dtype = spec.dtype
            if dtype == np.dtype(np.bool_):
                if null_value != 255:
                    raise ValueError(f"Null sentinel for nullable bool ndarray column {name!r} must be 255")
                return
            if dtype.kind in "iu":
                if isinstance(null_value, (bool, np.bool_)) or not isinstance(null_value, (int, np.integer)):
                    raise TypeError(f"Null sentinel for ndarray column {name!r} must be an integer")
                info = np.iinfo(dtype)
                if not info.min <= int(null_value) <= info.max:
                    raise ValueError(
                        f"Null sentinel for ndarray column {name!r}={null_value!r} is outside {dtype} range"
                    )
                return
            if dtype.kind == "f":
                if not isinstance(null_value, (int, float, np.integer, np.floating)):
                    raise TypeError(f"Null sentinel for ndarray column {name!r} must be numeric")
                return
            raise TypeError(
                f"Nullable ndarray column {name!r} has unsupported dtype {dtype!r}; "
                "use bool, integer, unsigned integer, or floating dtype."
            )
        if isinstance(spec, (int8, int16, int32, int64, uint8, uint16, uint32, uint64, timestamp)):
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
            if isinstance(spec, NDArraySpec) and getattr(spec, "null_value", None) is not None:
                cls._validate_null_value_for_spec(col.name, spec, spec.null_value)
                if spec.dtype == np.dtype(np.bool_):
                    spec.dtype = np.dtype(np.uint8)
                    spec.itemsize = spec.dtype.itemsize
                    spec.kind = spec.dtype.kind
                    spec.type = spec.dtype.type
                    spec.str = spec.dtype.str
                    spec.name = spec.dtype.name
                col.dtype = getattr(spec, "dtype", None)
                col.display_width = compute_display_width(spec)
                continue
            if (
                isinstance(
                    spec,
                    (ListSpec, VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec, DictionarySpec),
                )
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
            elif isinstance(spec, NDArraySpec) and spec.dtype == np.dtype(np.bool_):
                spec.dtype = np.dtype(np.uint8)
                spec.itemsize = spec.dtype.itemsize
                spec.kind = spec.dtype.kind
                spec.type = spec.dtype.type
                spec.str = spec.dtype.str
                spec.name = spec.dtype.name
            col.dtype = getattr(spec, "dtype", None)
            col.display_width = compute_display_width(spec)

    def _flush_varlen_columns(self) -> None:
        for col in self._schema.columns:
            if (
                self._is_list_column(col)
                or self._is_varlen_scalar_column(col)
                or self._is_dictionary_column(col)
            ):
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
            if self._is_dictionary_column(col):
                dict_col = storage.create_dictionary_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
                if len(dict_col.codes) < expected_size:
                    dict_col.resize((expected_size,))
                self._cols[col.name] = dict_col
                continue
            # Recompute chunks/blocks using the actual dtype so that wide
            # string columns (e.g. U183642) don't produce multi-GB chunks.
            chunks = col_storage["chunks"]
            blocks = col_storage["blocks"]
            shape = self._column_physical_shape(col, expected_size)
            if col.config.chunks is None and col.config.blocks is None:
                chunks, blocks = self._column_chunks_blocks(col, shape)
            self._cols[col.name] = storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=shape,
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

    @staticmethod
    def _flatten_nested_dict(d: dict, prefix: str = "") -> dict:
        """Recursively flatten a nested dict into a dotted-key flat dict.

        Works for both single-row dicts ``{field: value}`` and column-batch
        dicts ``{field: array}``.  Leaves non-dict values unchanged.

        Example::

            {"trip": {"begin": {"lon": 1.0}}} -> {"trip.begin.lon": 1.0}
        """
        result = {}
        for k, v in d.items():
            full_key = join_field_path((*split_field_path(prefix), k)) if prefix else join_field_path((k,))
            if isinstance(v, dict):
                result.update(CTable._flatten_nested_dict(v, full_key))
            else:
                result[full_key] = v
        return result

    def _normalize_row_input(self, data: Any) -> dict[str, Any]:
        """Normalize a row input to a ``{col_name: value}`` dict.

        Accepted shapes:
        - list / tuple  → positional, zipped with stored column names (computed columns skipped)
        - dict          → used as-is (nested dicts are flattened to dotted keys)
        - dataclass     → ``dataclasses.asdict`` (nested fields flattened)
        - np.void / structured scalar → field-name access
        """
        stored = self._append_input_col_names
        if isinstance(data, dict):
            if any(isinstance(v, dict) for v in data.values()):
                return self._flatten_nested_dict(data)
            return data
        if isinstance(data, (list, tuple)):
            return dict(zip(stored, data, strict=False))
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            d = dataclasses.asdict(data)
            if any(isinstance(v, dict) for v in d.values()):
                return self._flatten_nested_dict(d)
            return d
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
            elif self._is_dictionary_column(col):
                # Pass str/None through; DictionaryColumn.__setitem__ encodes.
                result[col.name] = val
            elif self._is_ndarray_column(col):
                result[col.name] = self._coerce_ndarray_value(col.name, col.spec, val)
            elif isinstance(col.spec, timestamp):
                if val is None:
                    result[col.name] = col.spec.null_value
                elif isinstance(val, (np.datetime64, str)) or hasattr(val, "isoformat"):
                    result[col.name] = (
                        np.datetime64(val).astype(f"datetime64[{col.spec.unit}]").astype(np.int64).item()
                    )
                else:
                    result[col.name] = np.array(val, dtype=col.dtype).item()
            else:
                result[col.name] = np.array(val, dtype=col.dtype).item()
        return result

    def _open_column_from_storage(self, storage: TableStorage, name: str):
        """Open one stored column from *storage*."""
        cc = self._schema.columns_by_name[name]
        if self._is_list_column(cc):
            return storage.open_list_column(name)
        if self._is_varlen_scalar_column(cc):
            return storage.open_varlen_scalar_column(name, cc.spec)
        if self._is_dictionary_column(cc):
            return storage.open_dictionary_column(name, cc.spec)
        return storage.open_column(name)

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
            if self._is_dictionary_column(cc):
                col_arr.resize((c * 2,))
                continue
            col_arr.resize(self._column_physical_shape(cc, c * 2))
        self._valid_rows.resize((c * 2,))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_positions(self, display_rows: int | None = None):
        nrows = self._n_rows
        display_rows = _CTABLE_PRINT_OPTIONS["display_rows"] if display_rows is None else display_rows
        if display_rows == 0:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), nrows
        valid_np = self._valid_rows[:]
        all_pos = np.where(valid_np)[0]
        if nrows <= display_rows:
            return all_pos, np.array([], dtype=all_pos.dtype), 0

        preview_rows = min(10, display_rows)
        head_rows = (preview_rows + 1) // 2
        tail_rows = preview_rows // 2
        hidden = max(0, nrows - head_rows - tail_rows)
        tail_pos = all_pos[-tail_rows:] if tail_rows else np.array([], dtype=all_pos.dtype)
        return all_pos[:head_rows], tail_pos, hidden

    def _display_widths(self, col_names: list[str] | None = None) -> dict[str, int]:
        widths: dict[str, int] = {}
        col_names = self.col_names if col_names is None else col_names
        single_col = len(col_names) == 1
        for name in col_names:
            if name == "...":
                widths[name] = 3
                continue
            spec = self._schema.columns_by_name.get(name)
            dtype_label = self._dtype_info_label(self._col_dtype(name), spec.spec if spec else None)
            widths[name] = max(self._col_widths[name], len(dtype_label))
            if single_col:
                widths[name] = max(widths[name], 80)
        return widths

    def _display_columns(
        self, *, display_index: bool = False, index_width: int = 0
    ) -> tuple[list[str], int]:
        """Return terminal-width-friendly display columns and hidden count."""
        col_names = list(self.col_names)
        widths = self._display_widths(col_names)
        widths["..."] = 3
        total_width = sum(widths[n] + 2 for n in col_names) + 2 * max(0, len(col_names) - 1)
        if display_index:
            total_width += index_width + 2 + 2
        term_width = shutil.get_terminal_size((120, 20)).columns
        if total_width <= term_width or len(col_names) <= 2:
            return col_names, 0

        selected: list[str] = []
        left = 0
        right = len(col_names) - 1
        used = index_width + 2 + 2 if display_index else 0

        def extra_width(name: str, n_existing: int) -> int:
            return widths[name] + 2 + (2 if n_existing else 0)

        # Account for an ellipsis column between left and right blocks.
        used += widths["..."] + 2
        while left <= right:
            left_name = col_names[left]
            need = extra_width(left_name, len(selected) + 1)
            if used + need > term_width:
                break
            selected.append(left_name)
            used += need
            left += 1
            if left > right:
                break

            right_name = col_names[right]
            need = extra_width(right_name, len(selected) + 1)
            if used + need > term_width:
                break
            selected.append(right_name)
            used += need
            right -= 1

        left_cols = [n for n in col_names if n in selected and col_names.index(n) < left]
        right_cols = [n for n in col_names if n in selected and col_names.index(n) > right]
        display_cols = left_cols + ["..."] + right_cols
        hidden = len(col_names) - len(left_cols) - len(right_cols)
        return display_cols, hidden

    @staticmethod
    def _cell_text(value, float_precision: int | None = None) -> str:
        if isinstance(value, np.datetime64):
            s = str(value).replace("T", " ")
            if s.endswith(".000"):
                s = s[:-4]
            return s
        if isinstance(value, np.ndarray):
            if value.ndim == 1 and value.size <= 6:
                return np.array2string(value, separator=", ", max_line_width=10_000)
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, (float, np.floating)):
            precision = (
                _CTABLE_PRINT_OPTIONS["display_precision"] if float_precision is None else float_precision
            )
            if _CTABLE_PRINT_OPTIONS["fancy"]:
                return np.format_float_positional(float(value), precision=precision, trim="-")
            return f"{float(value):.{precision}f}"
        return str(value)

    @staticmethod
    def _format_cell(value, width: int, float_precision: int | None = None) -> str:
        s = CTable._cell_text(value, float_precision)
        if len(s) > width:
            s = s[: width - 1] + "…"
        if _CTABLE_PRINT_OPTIONS["fancy"]:
            return f" {s:>{width}} "
        return f"{s:>{width}}"

    @staticmethod
    def _format_index_cell(value, width: int) -> str:
        s = "" if value is None else str(value)
        if len(s) > width:
            s = s[: width - 1] + "…"
        if _CTABLE_PRINT_OPTIONS["fancy"]:
            return f" {s:<{width}} "
        return f"{s:<{width}}"

    @staticmethod
    def _display_index_width(nrows: int, hidden: int, index_name: str) -> int:
        width = max(len(index_name), len(str(max(nrows - 1, 0))))
        if hidden > 0:
            width = max(width, 3)
        return width

    def _format_display_row(
        self,
        values: dict,
        widths: dict[str, int],
        col_names: list[str],
        float_precisions: dict[str, int] | None = None,
    ) -> str:
        float_precisions = {} if float_precisions is None else float_precisions
        return "  ".join(self._format_cell(values[n], widths[n], float_precisions.get(n)) for n in col_names)

    def _format_display_row_with_index(
        self,
        values: dict,
        widths: dict[str, int],
        col_names: list[str],
        index_value,
        index_width: int,
        float_precisions: dict[str, int] | None = None,
    ) -> str:
        return (
            self._format_index_cell(index_value, index_width)
            + "  "
            + self._format_display_row(values, widths, col_names, float_precisions)
        )

    def _rows_to_dicts(self, positions, col_names: list[str] | None = None) -> list[dict]:
        if len(positions) == 0:
            return []
        col_names = self.col_names if col_names is None else col_names
        real_cols = [n for n in col_names if n != "..."]
        col_data = {n: self._fetch_col_at_positions(n, positions) for n in real_cols}
        rows = []
        for i in range(len(positions)):
            row = {}
            for n in col_names:
                # Keep NumPy scalar types for display so their compact string
                # formatting is preserved (notably float32, e.g. 224.97
                # instead of Python float's 224.97000122070312).
                row[n] = "..." if n == "..." else col_data[n][i]
            rows.append(row)
        return rows

    def _display_separator(
        self,
        widths: dict[str, int],
        display_cols: list[str],
        display_index: bool,
        index_width: int,
        fancy: bool,
    ) -> str | None:
        if not fancy:
            return None
        sep_parts = ["─" * (widths[n] + 2) for n in display_cols]
        if display_index:
            sep_parts.insert(0, "─" * (index_width + 2))
        return "  ".join(sep_parts)

    def _display_dtype_row(self, display_cols: list[str]) -> dict:
        dtype_row = {}
        for n in display_cols:
            if n == "...":
                dtype_row[n] = "..."
            else:
                dtype_row[n] = self._dtype_info_label(
                    self._col_dtype(n),
                    self._schema.columns_by_name[n].spec if n in self._schema.columns_by_name else None,
                )
        return dtype_row

    def _compact_float_precisions(self, display_cols: list[str], head_pos, tail_pos) -> dict[str, int]:
        default_precision = _CTABLE_PRINT_OPTIONS["display_precision"]
        precisions: dict[str, int] = {}
        for n in display_cols:
            finite_float_seen = False
            integer_valued = True
            for positions in (head_pos, tail_pos):
                for row in self._rows_to_dicts(positions, [n]):
                    value = row[n]
                    if not isinstance(value, (float, np.floating)):
                        continue
                    value = float(value)
                    if not np.isfinite(value):
                        continue
                    finite_float_seen = True
                    if not value.is_integer():
                        integer_valued = False
                        break
                if not integer_valued:
                    break
            if finite_float_seen and integer_valued:
                precisions[n] = 1
            else:
                precisions[n] = default_precision
        return precisions

    def _compact_display_widths(
        self,
        display_cols: list[str],
        head_pos,
        tail_pos,
        hidden: int,
        float_precisions: dict[str, int],
    ) -> dict[str, int]:
        widths = {n: len(n) for n in display_cols}
        if hidden > 0:
            for n in display_cols:
                widths[n] = max(widths[n], 3)
        for positions in (head_pos, tail_pos):
            for row in self._rows_to_dicts(positions, display_cols):
                for n, value in row.items():
                    widths[n] = max(widths[n], len(self._cell_text(value, float_precisions.get(n))))
        return widths

    @staticmethod
    def _display_footer(nrows: int, ncols: int, hidden: int, hidden_cols: int, fancy: bool) -> list[str]:
        if not fancy:
            return ["", f"[{nrows} rows x {ncols} columns]"]
        footer = f"{nrows:,} rows × {ncols} columns"
        notes = []
        if hidden > 0:
            notes.append(f"{hidden:,} rows hidden")
        if hidden_cols > 0:
            notes.append(f"{hidden_cols:,} columns hidden")
        if notes:
            footer += f"  ({', '.join(notes)})"
        return [footer]

    def _display_lines_with_index(
        self,
        *,
        display_cols: list[str],
        widths: dict[str, int],
        index_name: str,
        index_width: int,
        head_pos,
        tail_pos,
        hidden: int,
        sep: str | None,
        fancy: bool,
        float_precisions: dict[str, int] | None = None,
    ) -> list[str]:
        header_row = {n: n for n in display_cols}
        lines = [
            self._format_display_row_with_index(
                header_row, widths, display_cols, index_name, index_width, float_precisions
            )
        ]
        if fancy:
            dtype_row = self._display_dtype_row(display_cols)
            lines.append(
                self._format_display_row_with_index(
                    dtype_row, widths, display_cols, None, index_width, float_precisions
                )
            )
        if sep is not None:
            lines.append(sep)
        lines.extend(
            self._format_display_row_with_index(row, widths, display_cols, i, index_width, float_precisions)
            for i, row in enumerate(self._rows_to_dicts(head_pos, display_cols))
        )
        if hidden > 0:
            lines.append(
                self._format_display_row_with_index(
                    dict.fromkeys(display_cols, "..."),
                    widths,
                    display_cols,
                    "...",
                    index_width,
                    float_precisions,
                )
            )
        tail_start = self._n_rows - len(tail_pos)
        lines.extend(
            self._format_display_row_with_index(
                row, widths, display_cols, tail_start + i, index_width, float_precisions
            )
            for i, row in enumerate(self._rows_to_dicts(tail_pos, display_cols))
        )
        return lines

    def _display_lines_without_index(
        self,
        *,
        display_cols: list[str],
        widths: dict[str, int],
        head_pos,
        tail_pos,
        hidden: int,
        sep: str | None,
        fancy: bool,
        float_precisions: dict[str, int] | None = None,
    ) -> list[str]:
        header_row = {n: n for n in display_cols}
        lines = [self._format_display_row(header_row, widths, display_cols, float_precisions)]
        if fancy:
            lines.append(
                self._format_display_row(
                    self._display_dtype_row(display_cols), widths, display_cols, float_precisions
                )
            )
        if sep is not None:
            lines.append(sep)
        lines.extend(
            self._format_display_row(row, widths, display_cols, float_precisions)
            for row in self._rows_to_dicts(head_pos, display_cols)
        )
        if hidden > 0:
            lines.append(
                self._format_display_row(
                    dict.fromkeys(display_cols, "..."), widths, display_cols, float_precisions
                )
            )
        lines.extend(
            self._format_display_row(row, widths, display_cols, float_precisions)
            for row in self._rows_to_dicts(tail_pos, display_cols)
        )
        return lines

    def to_string(self, *, display_index: bool | None = None, index_name: str = "") -> str:
        """Return a tabular string representation of the table.

        Parameters
        ----------
        display_index:
            Whether to include a pandas-like logical row index column.  If
            ``None`` (default), use the global value configured with
            :func:`blosc2.set_printoptions`.
        index_name:
            Optional label for the displayed index column.
        """
        if display_index is None:
            display_index = _CTABLE_PRINT_OPTIONS["display_index"]
        if not isinstance(display_index, bool):
            raise TypeError("display_index must be a bool or None")
        if not isinstance(index_name, str):
            raise TypeError("index_name must be a str")

        nrows = self._n_rows
        ncols = len(self.col_names)
        head_pos, tail_pos, hidden = self._display_positions()
        index_width = self._display_index_width(nrows, hidden, index_name) if display_index else 0
        display_cols, hidden_cols = self._display_columns(
            display_index=display_index, index_width=index_width
        )
        fancy = _CTABLE_PRINT_OPTIONS["fancy"]
        float_precisions = {} if fancy else self._compact_float_precisions(display_cols, head_pos, tail_pos)
        widths = (
            self._display_widths(display_cols)
            if fancy
            else self._compact_display_widths(display_cols, head_pos, tail_pos, hidden, float_precisions)
        )
        sep = self._display_separator(widths, display_cols, display_index, index_width, fancy)

        if display_index:
            lines = self._display_lines_with_index(
                display_cols=display_cols,
                widths=widths,
                index_name=index_name,
                index_width=index_width,
                head_pos=head_pos,
                tail_pos=tail_pos,
                hidden=hidden,
                sep=sep,
                fancy=fancy,
                float_precisions=float_precisions,
            )
        else:
            lines = self._display_lines_without_index(
                display_cols=display_cols,
                widths=widths,
                head_pos=head_pos,
                tail_pos=tail_pos,
                hidden=hidden,
                sep=sep,
                fancy=fancy,
                float_precisions=float_precisions,
            )
        if sep is not None:
            lines.append(sep)
        lines.extend(self._display_footer(nrows, ncols, hidden, hidden_cols, fancy))
        return "\n".join(lines)

    def __str__(self) -> str:
        """Pandas-style tabular display with column names, dtypes, and a row count footer."""
        return self.to_string()

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

    def _row_namedtuple_type_for_fields(self, fields: tuple[str, ...]):
        cache = getattr(self, "_row_namedtuple_type_cache_by_fields", None)
        if cache is None:
            cache = {}
            self._row_namedtuple_type_cache_by_fields = cache
        row_type = cache.get(fields)
        if row_type is None:
            row_type = _make_namedtuple_row_type(fields)
            cache[fields] = row_type
        return row_type

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
        value = self._normalize_scalar_value(self._cols[col_name][pos])
        spec = self._schema.columns_by_name[col_name].spec
        if isinstance(spec, timestamp):
            return np.datetime64(int(value), spec.unit)
        return value

    def _materialize_row(self, index: int):
        n_rows = self.nrows
        if index < 0:
            index += n_rows
        if not (0 <= index < n_rows):
            raise IndexError(f"row index {index} is out of bounds for table with {n_rows} rows")
        pos = _find_physical_index(self._valid_rows, index)

        nested_meta = self._schema.metadata.get("nested") if self._schema.metadata else None
        reconstruct = isinstance(nested_meta, dict) and bool(nested_meta.get("reconstruct_rows", False))
        if not reconstruct:
            row_type = self._row_namedtuple_type()
            return row_type(*(self._physical_row_value(name, int(pos)) for name in self.col_names))

        row_dict: dict[str, Any] = {}
        for name in self.col_names:
            value = self._physical_row_value(name, int(pos))
            parts = split_field_path(name)
            if len(parts) <= 1:
                row_dict[name] = value
                continue
            node = row_dict
            for part in parts[:-1]:
                child = node.get(part)
                if not isinstance(child, dict):
                    child = {}
                    node[part] = child
                node = child
            node[parts[-1]] = value

        fields = tuple(row_dict.keys())
        row_type = self._row_namedtuple_type_for_fields(fields)
        return row_type(*(row_dict[f] for f in fields))

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
    def _open_from_existing_filestore(cls, urlpath: str, *, mode: str, store: blosc2.TreeStore) -> CTable:
        """Open a root CTable reusing an already-opened TreeStore."""
        storage = FileTableStorage(urlpath, mode, store=store)
        return cls._open_from_storage(storage)

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
            if self._is_dictionary_column(col):
                src_dc = self._cols[name]
                disk_dc = storage.create_dictionary_column(
                    name,
                    spec=col.spec,
                    cparams=col.config.cparams if col.config.cparams is not None else self._table_cparams,
                    dparams=col.config.dparams if col.config.dparams is not None else self._table_dparams,
                )
                # Copy dictionary values first
                for v in src_dc.dictionary:
                    disk_dc.encode(v)
                disk_dc.flush()
                # Copy live codes
                if n_live > 0:
                    raw_codes = src_dc.codes[live_pos]
                    disk_dc.codes[:n_live] = raw_codes
                continue
            shape = self._column_physical_shape(col, capacity)
            dtype_chunks, dtype_blocks = self._column_chunks_blocks(col, shape)
            col_storage = self._resolve_column_storage(col, dtype_chunks, dtype_blocks)
            disk_col = storage.create_column(
                name,
                dtype=col.dtype,
                shape=shape,
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
        obj._cols = _LazyColumnDict(obj, storage, col_names)
        for name in col_names:
            cc = schema.columns_by_name[name]
            obj._col_widths[name] = max(len(name), cc.display_width)

        obj._n_rows = None
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
            elif cls._is_dictionary_column(col):
                disk_cols[col.name] = file_storage.open_dictionary_column(col.name, col.spec)
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
            if cls._is_dictionary_column(col):
                mem_col = mem_storage.create_dictionary_column(name, spec=col.spec)
                disk_dc = disk_cols[name]
                # Copy dictionary values
                for v in disk_dc.dictionary:
                    mem_col.encode(v)
                # Copy codes
                if phys_size > 0:
                    mem_col.codes[:phys_size] = disk_dc.codes[:phys_size]
                mem_cols[name] = mem_col
                continue
            shape = cls._column_physical_shape(col, capacity)
            col_chunks, col_blocks = cls._column_chunks_blocks(col, shape)
            mem_col = mem_storage.create_column(
                name,
                dtype=col.dtype,
                shape=shape,
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
        # Keep row counts lazy for views.  Many pipelines (e.g. where(...).sort_by(...))
        # immediately scan the mask for positions, so counting here would duplicate work.
        obj._n_rows = None
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
        all_pos = self._live_positions_from_valid_rows_chunks()
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
            Ordered list of column names to keep.  For tables with **nested
            (dotted) column names**, a struct-prefix name automatically expands
            to all descendant leaves::

                t.select(["trip.begin"])   # expands to trip.begin.lon, trip.begin.lat
                t.select(["trip"])          # expands to all trip.* leaves

        Raises
        ------
        KeyError
            If any name in *cols* is not a column of this table (and does not
            match any struct prefix).
        ValueError
            If *cols* is empty.
        """
        if not cols:
            raise ValueError("select() requires at least one column name.")
        expanded_cols = []
        for name in cols:
            expanded_cols.extend(self._expand_logical_column_selector(name))
        cols = expanded_cols
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
        obj._n_rows = self._known_n_rows()
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

    def group_by(
        self,
        keys: str | Sequence[str],
        *,
        sort: bool = False,
        dropna: bool = True,
        engine: str = "auto",
        chunk_size: int | None = None,
    ):
        """Return a deferred group-by object for this table.

        Parameters
        ----------
        keys:
            Column name or sequence of column names to group by.
        sort:
            If ``True``, sort the result by the group keys.  The default
            ``False`` preserves the hash aggregation order and is usually
            faster.
        dropna:
            If ``True`` (default), rows with null/NaN group keys are skipped.
            If ``False``, null/NaN keys form their own group.
        engine:
            Execution engine.  Phase 1 accepts ``"auto"`` and uses the NumPy
            chunked implementation.
        chunk_size:
            Optional number of physical rows processed per chunk.

        Returns
        -------
        CTableGroupBy
            A lightweight deferred operation builder.  Call methods such as
            ``.size()``, ``.count(column)`` or ``.agg({...})`` to materialize a
            grouped result as a new :class:`CTable`.
        """
        if engine != "auto":
            raise ValueError("Only engine='auto' is supported for group_by() in Phase 1")
        from blosc2.groupby import CTableGroupBy

        return CTableGroupBy(self, keys, sort=sort, dropna=dropna, engine=engine, chunk_size=chunk_size)

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

            if isinstance(spec.spec, NDArraySpec) if spec is not None else False:
                lines.append(f"    count      : {n:,}")
                lines.append(f"    item_shape : {spec.spec.item_shape}")
                lines.append(
                    "    (scalar stats not available for ndarray columns; use column reductions with axis=)"
                )
            elif isinstance(spec.spec, ListSpec) if spec is not None else False:
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
            col_info = self._schema.columns_by_name.get(name)
            if col_info is not None and self._is_ndarray_column(col_info):
                raise TypeError(
                    f"Column {name!r} is a fixed-shape ndarray column and is not supported by cov(). "
                    "Materialize scalar generated columns first."
                )
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

            # If top-level struct aliases are present in schema metadata (virtual
            # entries not physically stored), prefer exporting them instead of
            # their descendant dotted leaves.
            virtual_structs = [
                n
                for n, cc in self._schema.columns_by_name.items()
                if n not in self.col_names and isinstance(cc.spec, StructSpec)
            ]
            for alias in sorted(virtual_structs, key=len, reverse=True):
                alias_parts = split_field_path(alias)
                children = [
                    n
                    for n in names
                    if split_field_path(n)[: len(alias_parts)] == alias_parts
                    and len(split_field_path(n)) > len(alias_parts)
                ]
                if not children:
                    continue
                first = min(names.index(c) for c in children)
                child_set = set(children)
                names = [n for n in names if n not in child_set]
                names.insert(first, alias)
        else:
            names = []
            for name in columns:
                names.extend(self._expand_logical_column_selector(name))
        if len(set(names)) != len(names):
            raise ValueError("columns must be unique")
        for name in names:
            if name not in self.col_names and name not in self._schema.columns_by_name:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
        return names

    @staticmethod
    def _pa_type_from_spec(pa, spec):
        if isinstance(spec, DictionarySpec):
            return pa.dictionary(pa.int32(), pa.string(), ordered=spec.ordered)
        if isinstance(spec, VLStringSpec):
            return pa.string()
        if isinstance(spec, VLBytesSpec):
            return pa.large_binary()
        if isinstance(spec, ListSpec):
            return pa.list_(CTable._pa_type_from_spec(pa, spec.item_spec))
        if isinstance(spec, NDArraySpec):
            return pa.list_(pa.from_numpy_dtype(spec.dtype), list_size=int(np.prod(spec.item_shape)))
        if isinstance(spec, timestamp):
            return pa.timestamp(spec.unit, tz=spec.timezone)
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

    def _export_arrow_names(self, names: list[str]) -> list[str]:
        nested = self._schema.metadata.get("nested") if self._schema.metadata else None
        exported = list(names)
        if isinstance(nested, dict):
            root_meta = nested.get("root")
            if isinstance(root_meta, dict):
                physical = root_meta.get("physical")
                if isinstance(physical, str) and physical:
                    exported = ["" if n == physical else n for n in exported]
        for i, n in enumerate(names):
            cc = self._schema.columns_by_name.get(n)
            if n not in self.col_names and cc is not None and isinstance(cc.spec, StructSpec):
                parts = split_field_path(n)
                if len(parts) == 1:
                    exported[i] = parts[0]
        return exported

    def _arrow_schema_for_columns(self, columns=None, *, include_computed: bool = True):
        pa = self._require_pyarrow("to_arrow()/to_parquet()")
        names = self._resolve_arrow_columns(columns, include_computed=include_computed)
        arrow_names = self._export_arrow_names(names)
        fields = []
        for name, arrow_name in zip(names, arrow_names, strict=True):
            cc = self._schema.columns_by_name.get(name)
            metadata = None
            if cc is not None:
                pa_type = self._pa_type_from_spec(pa, cc.spec)
                if isinstance(cc.spec, NDArraySpec):
                    metadata = {b"blosc2:ndarray_shape": json.dumps(list(cc.spec.item_shape)).encode()}
            else:
                pa_type = pa.from_numpy_dtype(np.asarray(self[name][:0]).dtype)
            fields.append(pa.field(arrow_name, pa_type, metadata=metadata))
        return pa.schema(fields)

    def iter_arrow_batches(  # noqa: C901
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
        arrow_names = self._export_arrow_names(names)

        for start in range(0, self._n_rows, batch_size):
            stop = min(start + batch_size, self._n_rows)
            arrays = []
            for name in names:
                cc = self._schema.columns_by_name.get(name)
                if name not in self.col_names and cc is not None and isinstance(cc.spec, StructSpec):
                    values = self[name][start:stop]
                    arrays.append(pa.array(values, type=self._pa_type_from_spec(pa, cc.spec)))
                    continue
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
                if col.is_dictionary:
                    dc = self._cols[name]  # DictionaryColumn
                    spec = self._schema.columns_by_name[name].spec
                    # Get physical positions for live rows in [start, stop)
                    valid = self._valid_rows
                    real_pos = blosc2.where(valid, _arange(len(valid))).compute()
                    batch_real_pos = real_pos[start:stop]
                    if len(batch_real_pos) == 0:
                        pa_dict = pa.array(dc.dictionary, type=pa.string())
                        pa_indices = pa.array([], type=pa.int32())
                        arrays.append(
                            pa.DictionaryArray.from_arrays(pa_indices, pa_dict, ordered=spec.ordered)
                        )
                    else:
                        raw_codes = dc.codes[batch_real_pos]
                        null_mask = raw_codes == np.int32(spec.null_code)
                        safe_codes = raw_codes.copy()
                        safe_codes[null_mask] = 0
                        pa_dict = pa.array(dc.dictionary, type=pa.string())
                        pa_indices = pa.array(
                            safe_codes,
                            type=pa.int32(),
                            mask=null_mask if null_mask.any() else None,
                        )
                        arrays.append(
                            pa.DictionaryArray.from_arrays(pa_indices, pa_dict, ordered=spec.ordered)
                        )
                    continue
                if col.is_ndarray:
                    spec = self._schema.columns_by_name[name].spec
                    values = np.asarray(col[start:stop])
                    null_mask = col._null_mask_for(values) if col.null_value is not None else None
                    pa_type = self._pa_type_from_spec(pa, spec)
                    flat_values = np.ascontiguousarray(values.reshape(-1))
                    pa_values = pa.array(flat_values, type=pa_type.value_type)
                    arrays.append(
                        pa.FixedSizeListArray.from_arrays(
                            pa_values,
                            type=pa_type,
                            mask=(
                                pa.array(null_mask, type=pa.bool_())
                                if null_mask is not None and null_mask.any()
                                else None
                            ),
                        )
                    )
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
                elif self._schema.columns_by_name.get(name) is not None and isinstance(
                    self._schema.columns_by_name[name].spec, timestamp
                ):
                    spec = self._schema.columns_by_name[name].spec
                    values = arr.astype(f"datetime64[{spec.unit}]")
                    arrays.append(
                        pa.array(
                            values,
                            mask=null_mask if has_nulls else None,
                            type=pa.timestamp(spec.unit, tz=spec.timezone),
                        )
                    )
                else:
                    arrays.append(pa.array(arr, mask=null_mask if has_nulls else None))
            yield pa.RecordBatch.from_arrays(arrays, names=arrow_names)

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
        if pa.types.is_dictionary(pa_type):
            vt = pa_type.value_type
            return vt not in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8())
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
        if pa.types.is_timestamp(pa_type):
            return False
        return not (
            pa.types.is_list(pa_type)
            or pa.types.is_large_list(pa_type)
            or pa.types.is_fixed_size_list(pa_type)
            or pa.types.is_struct(pa_type)
        )

    @staticmethod
    def _arrow_type_to_spec(  # noqa: C901
        pa,
        pa_type,
        arrow_col=None,
        *,
        field_metadata=None,
        string_max_length=None,
        null_value=None,
        nullable=False,
        object_fallback: bool = False,
    ):
        import blosc2.schema as b2s

        # Handle Arrow dictionary types (dict-encoded strings)
        if pa.types.is_fixed_size_list(pa_type):
            shape = None
            if field_metadata:
                encoded = field_metadata.get(b"blosc2:ndarray_shape") or field_metadata.get(
                    "blosc2:ndarray_shape"
                )
                if encoded is not None:
                    if isinstance(encoded, bytes):
                        encoded = encoded.decode()
                    shape = tuple(int(x) for x in json.loads(encoded))
            if shape is None:
                shape = (int(pa_type.list_size),)
            value_type = pa_type.value_type
            value_spec = CTable._arrow_type_to_spec(pa, value_type, object_fallback=object_fallback)
            value_dtype = getattr(value_spec, "dtype", None)
            if value_dtype is None:
                raise TypeError(f"FixedSizeList values must have a fixed NumPy dtype, got {value_type!r}")
            if int(np.prod(shape)) != int(pa_type.list_size):
                raise ValueError(
                    f"Arrow fixed-size-list metadata shape {shape} has size {int(np.prod(shape))}, "
                    f"but the Arrow list size is {pa_type.list_size}."
                )
            return b2s.ndarray(shape, dtype=value_dtype, nullable=nullable, null_value=null_value)

        if pa.types.is_dictionary(pa_type):
            vt = pa_type.value_type
            if vt in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
                index_type = pa_type.index_type
                # Accept signed and unsigned integer index types; validate fit in int32.
                if not (pa.types.is_integer(index_type) or pa.types.is_unsigned_integer(index_type)):
                    raise TypeError(
                        f"Dictionary column has unsupported index type {index_type!r}; "
                        "expected an integer type."
                    )
                if arrow_col is not None:
                    # Validate all indices fit in signed int32.
                    if pa.types.is_unsigned_integer(index_type):
                        max_idx = arrow_col.combine_chunks().indices.to_pandas().max(skipna=True)
                        if max_idx is not None and max_idx > np.iinfo(np.int32).max:
                            raise ValueError(
                                f"Arrow dictionary column has unsigned indices exceeding int32.max "
                                f"(max={max_idx})."
                            )
                    combined = (
                        arrow_col.combine_chunks() if hasattr(arrow_col, "combine_chunks") else arrow_col
                    )
                    n_cats = len(combined.dictionary)
                    if n_cats > np.iinfo(np.int32).max:
                        raise OverflowError(
                            f"Arrow dictionary has {n_cats} categories, exceeding int32 capacity."
                        )
                return b2s.dictionary(
                    index_type=b2s.int32(),
                    value_type=b2s.vlstring(),
                    ordered=bool(pa_type.ordered),
                    nullable=nullable,
                )
            if object_fallback:
                return b2s.object(nullable=nullable)
            raise TypeError(
                f"No blosc2 spec for Arrow dictionary type {pa_type!r} with "
                f"value type {pa_type.value_type!r}. Only string dictionary values are supported in v1."
            )

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
        if pa.types.is_timestamp(pa_type):
            return b2s.timestamp(
                unit=pa_type.unit, timezone=pa_type.tz, nullable=nullable, null_value=null_value
            )

        for arrow_t, spec_cls in mapping:
            if pa_type == arrow_t:
                if null_value is not None and hasattr(spec_cls(), "null_value"):
                    return spec_cls(null_value=null_value)
                if null_value is not None and spec_cls is b2s.bool:
                    return spec_cls(null_value=null_value)
                return spec_cls()

        if pa.types.is_list(pa_type) or pa.types.is_large_list(pa_type):
            if arrow_col is not None:
                combined = arrow_col.combine_chunks() if hasattr(arrow_col, "combine_chunks") else arrow_col
                item_arrow_col = combined.values
                nullable = nullable or combined.null_count > 0
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
            field_is_ndarray = pa.types.is_fixed_size_list(field.type)
            field_is_list = (
                pa.types.is_list(field.type) or pa.types.is_large_list(field.type)
            ) and not field_is_ndarray
            field_is_struct = pa.types.is_struct(field.type)
            field_is_dictionary = pa.types.is_dictionary(field.type)
            column_string_max_length = cls._string_max_length_for_column(string_max_length, name)
            field_is_varlen_scalar = (
                not field_is_list
                and not field_is_struct
                and not field_is_dictionary
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
            if has_null_value_override and (
                field_is_list or field_is_struct or field_is_dictionary or field_is_object_fallback
            ):
                raise TypeError(
                    f"column_null_values only supports scalar columns and ndarray columns; {name!r} is not scalar"
                )
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
                    field_is_list
                    or field_is_struct
                    or field_is_dictionary
                    or field_is_varlen_scalar
                    or field_is_object_fallback
                )
            ):
                arrow_type_for_null = field.type.value_type if field_is_ndarray else field.type
                null_value = cls._auto_null_sentinel(pa, arrow_type_for_null, null_policy=null_policy)
            if (
                arrow_col is not None
                and arrow_col.null_count
                and not (
                    field_is_list
                    or field_is_struct
                    or field_is_dictionary
                    or field_is_varlen_scalar
                    or field_is_object_fallback
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
                field_metadata=field.metadata,
                string_max_length=column_string_max_length,
                null_value=null_value,
                nullable=field.nullable,
                object_fallback=object_fallback,
            )
            if null_value is not None and not (
                field_is_list
                or field_is_struct
                or field_is_dictionary
                or field_is_varlen_scalar
                or field_is_object_fallback
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

    @classmethod
    def _apply_arrow_column_cparams(
        cls, columns: list[CompiledColumn], column_cparams: Mapping[str, dict[str, Any]] | None
    ) -> None:
        if column_cparams is None:
            return
        unknown = set(column_cparams) - {col.name for col in columns}
        if unknown:
            names = ", ".join(sorted(unknown))
            raise KeyError(f"column_cparams contains unknown columns: {names}")
        for col in columns:
            if col.name in column_cparams:
                if cls._is_list_column(col) or cls._is_varlen_scalar_column(col):
                    raise TypeError(
                        f"column_cparams only supports fixed-width columns; {col.name!r} is not fixed-width"
                    )
                col.config.cparams = dict(column_cparams[col.name])

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
        new_cols: dict[str, blosc2.NDArray | ListArray | _ScalarVarLenArray | DictionaryColumn] = {}
        for col in columns:
            if cls._is_list_column(col):
                new_cols[col.name] = storage.create_list_column(
                    col.name, spec=col.spec, cparams=cparams, dparams=dparams
                )
            elif cls._is_varlen_scalar_column(col):
                new_cols[col.name] = storage.create_varlen_scalar_column(
                    col.name, spec=col.spec, cparams=cparams, dparams=dparams
                )
            elif cls._is_dictionary_column(col):
                dict_col = storage.create_dictionary_column(
                    col.name, spec=col.spec, cparams=cparams, dparams=dparams
                )
                if len(dict_col.codes) < capacity:
                    dict_col.resize((capacity,))
                new_cols[col.name] = dict_col
            else:
                shape = cls._column_physical_shape(col, capacity)
                chunks, blocks = default_chunks, default_blocks
                if col.dtype is not None:
                    chunks, blocks = cls._column_chunks_blocks(col, shape)
                new_cols[col.name] = storage.create_column(
                    col.name,
                    dtype=col.dtype,
                    shape=shape,
                    chunks=chunks,
                    blocks=blocks,
                    cparams=col.config.cparams if col.config.cparams is not None else cparams,
                    dparams=col.config.dparams if col.config.dparams is not None else dparams,
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

    @staticmethod
    def _timestamp_normalizer_for_spec(spec: SchemaSpec):  # noqa: C901
        """Build a trusted Arrow-import normalizer for timestamp leaves.

        Arrow already validates list/struct values during import, so list columns
        normally skip Python-level coercion.  The exception is nested timestamps:
        ``to_pylist()`` yields ``datetime``/``numpy.datetime64`` objects, while
        msgpack-backed ListArray storage expects integer epoch offsets.  Return a
        small normalizer that descends only into branches containing timestamps,
        or ``None`` when no normalization is needed.
        """
        if isinstance(spec, timestamp):

            def normalize_timestamp(value, unit=spec.unit):
                if value is None:
                    return None
                if isinstance(value, (int, np.integer)):
                    return int(value)
                return np.datetime64(value).astype(f"datetime64[{unit}]").astype(np.int64).item()

            return normalize_timestamp

        if isinstance(spec, ListSpec):
            item_normalizer = CTable._timestamp_normalizer_for_spec(spec.item_spec)
            if item_normalizer is None:
                return None

            def normalize_list(value, item_normalizer=item_normalizer):
                if value is None:
                    return None
                for i, item in enumerate(value):
                    value[i] = item_normalizer(item)
                return value

            return normalize_list

        if isinstance(spec, StructSpec):
            field_normalizers = {
                name: normalizer
                for name, child in spec.fields.items()
                if (normalizer := CTable._timestamp_normalizer_for_spec(child)) is not None
            }
            if not field_normalizers:
                return None

            def normalize_struct(value, field_normalizers=field_normalizers):
                if value is None:
                    return None
                for name, normalizer in field_normalizers.items():
                    if name in value:
                        value[name] = normalizer(value[name])
                return value

            return normalize_struct

        return None

    @classmethod
    def _trim_arrow_import_capacity(cls, obj, columns, new_cols, new_valid, n_rows: int) -> None:
        """Shrink append-only Arrow-import columns from capacity to actual row count."""
        if n_rows <= 0 or len(new_valid) == n_rows:
            return
        for col in columns:
            if cls._is_list_column(col) or cls._is_varlen_scalar_column(col):
                continue
            if cls._is_dictionary_column(col):
                new_cols[col.name].resize((n_rows,))
            else:
                new_cols[col.name].resize(cls._column_physical_shape(col, n_rows))
        new_valid.resize((n_rows,))
        new_valid[:] = True

    @classmethod
    def _write_arrow_batches(cls, obj, batches, columns, new_cols, new_valid) -> None:
        pos = 0
        list_normalizers = {
            col.name: cls._timestamp_normalizer_for_spec(col.spec)
            for col in columns
            if cls._is_list_column(col)
        }
        for batch in batches:
            end = pos + len(batch)
            while end > len(new_valid):
                obj._grow()
                new_valid = obj._valid_rows
            pos = cls._write_arrow_batch(batch, columns, new_cols, new_valid, pos, list_normalizers)
        for col in columns:
            if (
                cls._is_list_column(col)
                or cls._is_varlen_scalar_column(col)
                or cls._is_dictionary_column(col)
            ):
                new_cols[col.name].flush()
        cls._trim_arrow_import_capacity(obj, columns, new_cols, new_valid, pos)
        obj._n_rows = pos
        obj._last_pos = pos

    @classmethod
    def _write_arrow_batch(cls, batch, columns, new_cols, new_valid, pos: int, list_normalizers) -> int:
        m = len(batch)
        if m == 0:
            return pos
        for col in columns:
            arrow_col = batch.column(batch.schema.get_field_index(col.name))
            if cls._is_list_column(col):
                if getattr(col.spec, "serializer", None) == "arrow":
                    new_cols[col.name].extend_arrow(arrow_col)
                    continue
                # Trusted Arrow-import fast path: schema has already been inferred,
                # so avoid Python-level per-item coercion.  If nested timestamps
                # are present, normalize only those leaves before storing.
                values = arrow_col.to_pylist()
                normalizer = list_normalizers[col.name]
                if normalizer is not None:
                    values = [normalizer(value) for value in values]
                new_cols[col.name].extend(values, validate=False)
            elif cls._is_varlen_scalar_column(col):
                new_cols[col.name].extend(arrow_col.to_pylist())
            elif cls._is_dictionary_column(col):
                import pyarrow as _pa

                if _pa.types.is_dictionary(arrow_col.type):
                    # Arrow dictionary array: use unification algorithm.
                    new_cols[col.name].extend_from_arrow(_pa, arrow_col, pos, m, ordered=col.spec.ordered)
                else:
                    # Plain string array: encode values into the dictionary.
                    new_cols[col.name][pos : pos + m] = arrow_col.to_pylist()
            else:
                new_cols[col.name][pos : pos + m] = cls._arrow_column_to_numpy(arrow_col, col)
        new_valid[pos : pos + m] = True
        return pos + m

    @staticmethod
    def _arrow_column_to_numpy(arrow_col, col: CompiledColumn) -> np.ndarray:
        nv = getattr(col.spec, "null_value", None)
        if col.spec.to_metadata_dict().get("kind") == "bool" and col.dtype == np.dtype(np.uint8):
            return np.array([nv if v is None else int(v) for v in arrow_col.to_pylist()], dtype=np.uint8)
        if isinstance(col.spec, NDArraySpec):
            values = arrow_col.to_pylist()
            arr = CTable._coerce_ndarray_batch(col.name, col.spec, values, len(values))
            return arr.reshape((len(values), *col.spec.item_shape))
        if isinstance(col.spec, timestamp):
            arr = (
                arrow_col.to_numpy(zero_copy_only=False)
                .astype(f"datetime64[{col.spec.unit}]")
                .astype(np.int64)
            )
            if arrow_col.null_count and nv is not None and int(nv) != int(np.iinfo(np.int64).min):
                arr[arr == np.iinfo(np.int64).min] = int(nv)
            return arr.astype(col.dtype, copy=False)
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

    @staticmethod
    def _nested_metadata_from_column_names(
        column_names: list[str], *, empty_root_physical: str | None = None
    ) -> dict:
        logical_to_physical = {}
        physical_to_storage = {}
        for name in column_names:
            logical_to_physical[name] = name
            physical_to_storage[name] = f"_cols/{_column_name_to_relpath(name)}"
        nested = {
            "version": 1,
            "logical_root": "",
            "logical_to_physical": logical_to_physical,
            "physical_to_storage": physical_to_storage,
        }
        if empty_root_physical:
            logical_to_physical[""] = empty_root_physical
            nested["root"] = {"logical": "", "physical": empty_root_physical}
        return nested

    # ------------------------------------------------------------------
    # Unnamed-root list<struct<...>> detection and flattening helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_unnamed_root_list_struct(pa, schema) -> bool:
        """Return True iff *schema* qualifies for unnamed-root list<struct<...>> flattening.

        Conditions (all must hold):
        * exactly one top-level field;
        * field name is ``""`` (the canonical unnamed Arrow root);
        * field type is ``list<struct<...>>`` or ``large_list<struct<...>>``.
        """
        if len(schema) != 1:
            return False
        field = schema[0]
        if field.name != "":
            return False
        t = field.type
        if not (pa.types.is_list(t) or pa.types.is_large_list(t)):
            return False
        return pa.types.is_struct(t.value_type)

    @staticmethod
    def _inner_schema_for_unnamed_root(pa, schema):
        """Extract the inner struct schema from a single unnamed root list<struct<...>> schema.

        Returns a new Arrow schema whose top-level fields are the struct fields
        of the list value type.  The nullable flag of the original unnamed field
        is not propagated — individual struct child nullability applies.
        """
        field = schema[0]  # the unnamed "" field
        struct_type = field.type.value_type  # struct type inside the list
        return pa.schema(list(struct_type))

    @staticmethod
    def _flatten_root_list_struct_batches(pa, inner_schema, batches, max_rows: int | None = None):
        """Yield flattened :class:`pyarrow.RecordBatch` objects from an unnamed root stream.

        For each incoming batch (which has a single list<struct<...>> column),
        flatten the outer list using ``ListArray.flatten()`` — which skips null
        outer list rows — and convert the resulting struct array into a
        :class:`~pyarrow.RecordBatch` whose columns correspond to the struct fields.

        Parameters
        ----------
        pa:
            The ``pyarrow`` module.
        inner_schema:
            Arrow schema for the inner struct (output of
            :meth:`_inner_schema_for_unnamed_root`).
        batches:
            Iterable of incoming :class:`~pyarrow.RecordBatch` objects from the
            unnamed-root Parquet file.
        max_rows:
            Optional maximum number of flattened element rows to yield.
        """
        rows_seen = 0
        for batch in batches:
            if max_rows is not None and rows_seen >= max_rows:
                break
            list_array = batch.column(0)
            # flatten() skips null outer list rows and concatenates element values
            struct_values = list_array.flatten()
            if max_rows is not None:
                remaining = max_rows - rows_seen
                if len(struct_values) > remaining:
                    struct_values = struct_values.slice(0, remaining)
            n_values = len(struct_values)
            if n_values == 0:
                # Emit an empty record batch that still carries the inner schema
                empty_arrays = [pa.array([], type=f.type) for f in inner_schema]
                yield pa.record_batch(empty_arrays, schema=inner_schema)
                continue
            rows_seen += n_values
            yield pa.RecordBatch.from_struct_array(struct_values)

    @staticmethod
    def _flatten_arrow_struct_schema(pa, schema):
        """Flatten top-level struct fields into dotted leaf fields recursively."""

        out_fields = []

        def _walk(field, prefix: tuple[str, ...] = (), parent_nullable: bool = False):
            parts = (*prefix, field.name)
            name = join_field_path(parts)
            nullable = bool(parent_nullable or field.nullable)
            if pa.types.is_struct(field.type):
                for child in field.type:
                    _walk(pa.field(child.name, child.type, nullable=child.nullable), parts, nullable)
            else:
                out_fields.append(pa.field(name, field.type, nullable=nullable))

        for f in schema:
            _walk(f)
        return pa.schema(out_fields, metadata=schema.metadata)

    @staticmethod
    def _flatten_arrow_struct_batch(pa, batch, flat_schema):
        arrays = []

        def _extract(array, arr_type, parts):
            if not parts:
                return array
            head = parts[0]
            if pa.types.is_struct(arr_type):
                return _extract(array.field(head), arr_type[head].type, parts[1:])
            raise KeyError("Invalid flattened path")

        for field in flat_schema:
            parts = split_field_path(field.name)
            col = batch.column(batch.schema.get_field_index(parts[0]))
            arr = _extract(col, col.type, parts[1:])
            arrays.append(arr)
        return pa.RecordBatch.from_arrays(arrays, schema=flat_schema)

    @classmethod
    def _flatten_arrow_struct_input(cls, pa, schema, batches):
        """Return flattened (schema, batches, flattened) for struct-containing Arrow inputs."""
        if not any(pa.types.is_struct(f.type) for f in schema):
            return schema, batches, False
        flat_schema = cls._flatten_arrow_struct_schema(pa, schema)

        def _gen():
            for b in batches:
                yield cls._flatten_arrow_struct_batch(pa, b, flat_schema)

        return flat_schema, _gen(), True

    @classmethod
    def from_arrow(  # noqa: C901
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
        list_serializer: Literal["msgpack", "arrow"] = "msgpack",
        object_fallback: bool = False,
        column_cparams: Mapping[str, dict[str, Any]] | None = None,
        separate_nested_cols: bool = False,
    ) -> CTable:
        """Build a :class:`CTable` from an Arrow schema and iterable of record batches.

        **Nested struct flattening**: top-level Arrow ``struct<…>`` fields are
        automatically and recursively flattened into dotted leaf columns.  For
        example, a field ``trip: struct<begin: struct<lon: float64, lat: float64>>``
        becomes two CTable columns ``trip.begin.lon`` and ``trip.begin.lat``.
        Each leaf is stored as an independent compressed :class:`~blosc2.NDArray`.
        Row reads via ``t[i]`` reconstruct the original nested dict shape.  Use
        ``t["trip.begin.lon"]`` or ``t.trip.begin.lon`` to access a leaf::

            import pyarrow as pa, blosc2
            trip_type = pa.struct([("begin", pa.struct([("lon", pa.float64())]))])
            schema = pa.schema([pa.field("trip", trip_type)])
            t = blosc2.CTable.from_arrow(schema, batches)
            t.col_names          # ['trip.begin.lon']
            t["trip.begin.lon"].mean()
            t.trip.begin.lon.max()

        When *string_max_length* is ``None`` (the default), scalar Arrow
        ``string`` / ``large_string`` columns are imported as
        :func:`~blosc2.vlstring` columns and ``binary`` / ``large_binary``
        columns are imported as :func:`~blosc2.vlbytes` columns.  Non-struct
        ``struct`` columns (not containing only scalar leaves) are imported as
        :func:`~blosc2.struct` columns backed by batched variable-length
        storage.  Null values for these variable-length scalar columns are
        represented as native ``None`` with no sentinel needed.

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

        ``list_serializer`` selects the backend serializer for imported list
        columns. ``"msgpack"`` is the default; ``"arrow"`` stores Arrow list
        batches directly and can be much faster for deeply nested list columns.

        Unsupported Arrow types raise by default.  Pass ``object_fallback=True``
        to import such columns as schema-less :func:`~blosc2.object` columns.
        This fallback is intentionally not used by :meth:`from_parquet`.

        ``column_cparams`` optionally maps column names to per-column compression
        parameters. These override the table-level ``cparams`` for fixed-width
        columns imported from Arrow.
        """
        pa = cls._require_pyarrow("from_arrow()")
        if blosc2_batch_size is not None and blosc2_batch_size <= 0:
            raise ValueError("blosc2_batch_size must be a positive integer or None")
        if blosc2_items_per_block is not None and blosc2_items_per_block <= 0:
            raise ValueError("blosc2_items_per_block must be a positive integer or None")
        if list_serializer not in {"msgpack", "arrow"}:
            raise ValueError("list_serializer must be 'msgpack' or 'arrow'")

        # ------------------------------------------------------------------
        # Unnamed-root list<struct<...>> flattening (opt-in)
        # ------------------------------------------------------------------
        # When the source schema is a single unnamed "" field of type
        # list<struct<...>>, the outer list is a physical Parquet/Awkward
        # chunking artifact, not a semantic column.  Flatten it so that each
        # element becomes a CTable row.  The struct fields become ordinary
        # top-level columns and are further flattened by the struct-leaf
        # machinery below.
        original_root_metadata: dict | None = None
        if separate_nested_cols and cls._detect_unnamed_root_list_struct(pa, schema):
            inner_schema = cls._inner_schema_for_unnamed_root(pa, schema)
            batches = cls._flatten_root_list_struct_batches(pa, inner_schema, batches)
            schema = inner_schema
            original_root_metadata = {
                "kind": "unnamed_list_struct",
                "field_name": "",
                "preserve_grouping": False,
            }

        batches = iter(batches)
        first_batch = None
        table_for_inference = None
        original_top_level_struct_specs: dict[str, SchemaSpec] = {}
        for f in schema:
            if pa.types.is_struct(f.type):
                original_top_level_struct_specs[join_field_path((f.name,))] = cls._arrow_type_to_spec(
                    pa, f.type, nullable=f.nullable, object_fallback=object_fallback
                )
        if string_max_length is None or isinstance(string_max_length, Mapping):
            first_batch = next(batches, None)

        # Flatten top-level Arrow structs into dotted leaf columns so CTable can
        # persist nested scalar leaves as physical columns.
        flattened_structs = False
        if first_batch is not None:
            import itertools as _it

            schema, flat_batches, flattened_structs = cls._flatten_arrow_struct_input(
                pa, schema, _it.chain([first_batch], batches)
            )
            batches = iter(flat_batches)
            first_batch = next(batches, None)
        else:
            schema, batches, flattened_structs = cls._flatten_arrow_struct_input(pa, schema, batches)

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
        cls._apply_arrow_column_cparams(columns, column_cparams)
        for col in columns:
            if cls._is_list_column(col):
                if getattr(col.spec, "storage", None) == "batch":
                    col.spec.serializer = list_serializer
                    if blosc2_batch_size is not None:
                        col.spec.batch_rows = blosc2_batch_size
                    if blosc2_items_per_block is not None:
                        col.spec.items_per_block = blosc2_items_per_block
            elif cls._is_varlen_scalar_column(col):
                if blosc2_batch_size is not None:
                    col.spec.batch_rows = blosc2_batch_size
                if blosc2_items_per_block is not None:
                    col.spec.items_per_block = blosc2_items_per_block
        metadata = cls._arrow_schema_metadata(schema)
        empty_root_physical = None
        schema_meta = getattr(schema, "metadata", None) or {}
        root_key = b"blosc2_empty_root_physical"
        if root_key in schema_meta:
            raw = schema_meta[root_key]
            empty_root_physical = raw.decode() if isinstance(raw, bytes) else str(raw)
        metadata["nested"] = cls._nested_metadata_from_column_names(
            [col.name for col in columns], empty_root_physical=empty_root_physical
        )
        if flattened_structs:
            metadata["nested"]["reconstruct_rows"] = True
        if original_root_metadata is not None:
            metadata["nested"]["original_root"] = original_root_metadata
        compiled_columns_by_name = {col.name: col for col in columns}
        for name, spec in original_top_level_struct_specs.items():
            if name in compiled_columns_by_name:
                continue
            compiled_columns_by_name[name] = CompiledColumn(
                name=name,
                py_type=spec.python_type,
                spec=spec,
                dtype=getattr(spec, "dtype", None),
                default=MISSING,
                config=ColumnConfig(cparams=None, dparams=None, chunks=None, blocks=None),
                display_width=compute_display_width(spec),
            )

        compiled = CompiledSchema(
            row_cls=None,
            columns=columns,
            columns_by_name=compiled_columns_by_name,
            metadata=metadata,
        )
        if first_batch is not None:
            import itertools as _it

            batches = _it.chain([first_batch], batches)
        # Use capacity_hint to size initial NDArray chunks/blocks correctly.
        # When capacity_hint is None and we are in the unnamed-root flatten path,
        # fall back to _EXPECTED_SIZE_DEFAULT (1 M) so that compute_chunks_blocks
        # produces a reasonable block size instead of (1,) which causes catastrophic
        # storage fragmentation.  For non-unnamed-root imports capacity_hint is
        # always supplied by from_parquet (pf.metadata.num_rows), so the fallback
        # only matters for direct from_arrow() calls without a hint.
        if capacity_hint is None and original_root_metadata is not None:
            capacity = _EXPECTED_SIZE_DEFAULT
        else:
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
    def from_parquet(  # noqa: C901
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
        list_serializer: Literal["msgpack", "arrow"] = "arrow",
        separate_nested_cols: bool = True,
        max_rows: int | None = None,
        **kwargs,
    ) -> CTable:
        """Read a Parquet file into a :class:`CTable`.

        The Parquet file is streamed batch by batch through :mod:`pyarrow` and then
        converted into a typed :class:`CTable`. By default, the result is created in
        memory, but you can also persist it on disk via ``urlpath``.

        This method delegates the actual table construction to
        :meth:`CTable.from_arrow`, so Arrow schema handling, nullable-column support,
        and Blosc2 write tuning follow the same rules as that method.

        **Nested struct flattening**: top-level Parquet ``struct<…>`` fields are
        automatically and recursively flattened into dotted leaf columns — the same
        as in :meth:`from_arrow`.  For example, a Parquet file that contains a column
        ``trip: struct<begin: struct<lon: double, lat: double>>`` produces two CTable
        columns ``trip.begin.lon`` and ``trip.begin.lat``.  Row reads reconstruct the
        original nested dict shape; individual leaves are accessed via dotted names or
        attribute-chain proxies::

            t = blosc2.CTable.from_parquet("trips.parquet")
            t.col_names               # e.g. ['trip.begin.lon', 'trip.begin.lat', ...]
            t["trip.begin.lon"].mean()
            t.trip.begin.lon.max()

        Unsupported Parquet types are not silently imported as schema-less
        :func:`~blosc2.object` columns; they raise so callers can decide how to
        handle them explicitly.

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
            :meth:`CTable.from_arrow`.  In general, larger number of items
            favors compression ratios but make random access slower.

        list_serializer : {"msgpack", "arrow"}, optional
            Serializer used for imported list columns. The default, ``"arrow"``,
            stores Arrow list batches directly and is much faster for deeply nested
            or ``list<struct<...>>`` columns. The tradeoff is that accessing those
            list columns later requires PyArrow. Use ``"msgpack"`` to keep
            list-column stores independent of PyArrow at read time; it can be
            smaller for simple lists but is much slower and more memory-intensive
            for deeply nested data.

        separate_nested_cols : bool, optional
            Whether to separate qualifying nested columns during import. Defaults to
            ``True``. In particular, a single unnamed top-level
            ``list<struct<...>>`` field is treated as a root record stream: each list
            element becomes a CTable row and struct leaves become ordinary nested
            CTable columns. Use ``separate_nested_cols=False`` when closer fidelity to
            the original Parquet row/schema shape is more important than the separated
            column layout.

        max_rows : int or None, optional
            Maximum number of rows to import. For ordinary Parquet files this limits
            Parquet/CTable rows. For unnamed-root ``list<struct<...>>`` files imported
            with ``separate_nested_cols=True``, this limits flattened element rows.

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
            If ``max_rows`` is negative.
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
        if max_rows is not None and max_rows < 0:
            raise ValueError("max_rows must be non-negative")
        string_max_length = kwargs.pop("string_max_length", None)
        pf = pq.ParquetFile(path, **kwargs)
        arrow_schema = pf.schema_arrow
        if columns is not None:
            if len(set(columns)) != len(columns):
                raise ValueError("columns must be unique")
            fields = [arrow_schema.field(name) for name in columns]
            arrow_schema = pa.schema(fields)
        batches = pf.iter_batches(batch_size=batch_size, columns=columns)

        # Parquet files generated by Awkward-style pipelines may contain an
        # unnamed top-level field (""). When separate_nested_cols=True and the
        # schema qualifies as an unnamed-root list<struct<...>>, skip the
        # rename-to-root logic and pass the original schema directly to
        # from_arrow, which will perform the element-level flattening.
        # Otherwise, normalize empty column names to non-empty names as before.
        _is_unnamed_root_flatten = separate_nested_cols and cls._detect_unnamed_root_list_struct(
            pa, arrow_schema
        )
        if not _is_unnamed_root_flatten and any(name == "" for name in arrow_schema.names):
            used = {n for n in arrow_schema.names if n}

            def _fresh_root_name() -> str:
                base = "root"
                if base not in used:
                    used.add(base)
                    return base
                i = 1
                while True:
                    candidate = f"{base}_{i}"
                    if candidate not in used:
                        used.add(candidate)
                        return candidate
                    i += 1

            original_names = list(arrow_schema.names)
            renamed = [_fresh_root_name() if n == "" else n for n in original_names]
            arrow_schema = pa.schema(
                [arrow_schema.field(i).with_name(renamed[i]) for i in range(len(renamed))]
            )
            # Preserve canonical unnamed-root intent in schema metadata.
            try:
                first_root = next(renamed[i] for i, old in enumerate(original_names) if old == "")
            except StopIteration:
                first_root = renamed[0] if renamed else "root"
            current_meta = dict(arrow_schema.metadata or {})
            current_meta[b"blosc2_empty_root_physical"] = first_root.encode()
            arrow_schema = arrow_schema.with_metadata(current_meta)

            def _renamed_batches(batch_iter, names):
                for b in batch_iter:
                    yield b.rename_columns(names)

            batches = _renamed_batches(batches, renamed)

        def _limited_batches(batch_iter, limit: int):
            rows_seen = 0
            for batch in batch_iter:
                if rows_seen >= limit:
                    break
                remaining = limit - rows_seen
                if len(batch) > remaining:
                    batch = batch.slice(0, remaining)
                rows_seen += len(batch)
                yield batch

        # For unnamed-root flattening, max_rows applies to flattened element rows,
        # not to the outer Parquet rows.  Pre-flatten here when a limit is requested
        # so the limit can be enforced precisely before handing batches to from_arrow.
        if _is_unnamed_root_flatten and max_rows is not None:
            inner_schema = cls._inner_schema_for_unnamed_root(pa, arrow_schema)
            limited_flat_batches = cls._flatten_root_list_struct_batches(
                pa, inner_schema, batches, max_rows=max_rows
            )
            ct = cls.from_arrow(
                inner_schema,
                limited_flat_batches,
                urlpath=urlpath,
                mode=mode,
                cparams=cparams,
                dparams=dparams,
                validate=validate,
                capacity_hint=max_rows,
                string_max_length=string_max_length,
                auto_null_sentinels=auto_null_sentinels,
                blosc2_batch_size=blosc2_batch_size,
                blosc2_items_per_block=blosc2_items_per_block,
                list_serializer=list_serializer,
                separate_nested_cols=False,
            )
            nested_meta = ct._schema.metadata.get("nested", {})
            nested_meta["original_root"] = {
                "kind": "unnamed_list_struct",
                "field_name": "",
                "preserve_grouping": False,
            }
            ct._schema.metadata["nested"] = nested_meta
            ct._storage.save_schema(schema_to_dict(ct._schema))
            return ct

        if max_rows is not None:
            batches = _limited_batches(batches, max_rows)

        # When flattening a root list<struct<...>>, the actual element count is not
        # known ahead of time.  Pass capacity_hint=None so that from_arrow falls back
        # to _EXPECTED_SIZE_DEFAULT (1 M), which gives compute_chunks_blocks() a
        # reasonable block size instead of the catastrophic (1, 1) produced by
        # capacity=1.  The CLI path computes a better estimate by sampling.
        if _is_unnamed_root_flatten:
            _capacity_hint = None
        elif pf.metadata is not None:
            _capacity_hint = (
                pf.metadata.num_rows if max_rows is None else min(max_rows, pf.metadata.num_rows)
            )
        else:
            _capacity_hint = max_rows

        return cls.from_arrow(
            arrow_schema,
            batches,
            urlpath=urlpath,
            mode=mode,
            cparams=cparams,
            dparams=dparams,
            validate=validate,
            capacity_hint=_capacity_hint,
            string_max_length=string_max_length,
            auto_null_sentinels=auto_null_sentinels,
            blosc2_batch_size=blosc2_batch_size,
            blosc2_items_per_block=blosc2_items_per_block,
            list_serializer=list_serializer,
            separate_nested_cols=separate_nested_cols,
        )

    # ------------------------------------------------------------------
    # CSV interop
    # ------------------------------------------------------------------

    def to_csv(self, path: str, *, header: bool = True, sep: str = ",") -> None:
        """Write all live rows to a CSV file.

        Uses Python's stdlib ``csv`` module — no extra dependency required.
        Fixed-shape ndarray column cells are serialised as JSON arrays for
        readability and shape safety (e.g. ``"[1.0, 2.0, 3.0]"``).

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

        n = len(self)
        arrays: list = []
        for name in self.col_names:
            col = self[name]
            if col.is_ndarray:
                arr = col[:]
                null_mask = col._null_mask_for(arr)
                json_strings: list[str] = []
                for i in range(n):
                    if null_mask[i]:
                        json_strings.append("")
                    else:
                        json_strings.append(json.dumps(arr[i].tolist()))
                arrays.append(json_strings)
            else:
                arrays.append(col[:])

        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            if header:
                writer.writerow(self.col_names)
            for row in zip(*arrays, strict=True):
                writer.writerow(row)

    @staticmethod
    def _csv_ndarray_col_to_array(raw: list[str], col) -> np.ndarray:
        """Convert a list of JSON-array CSV strings to a stacked ndarray for an ndarray column."""
        spec = col.spec
        null_value = getattr(spec, "null_value", None)
        item_shape = spec.item_shape
        dtype = spec.dtype

        rows = []
        for val in raw:
            stripped = val.strip()
            if stripped == "":
                if null_value is not None:
                    rows.append(np.full(item_shape, null_value, dtype=dtype))
                    continue
                raise ValueError(f"Column {col.name!r}: non-nullable column got empty cell")

            try:
                arr = np.array(json.loads(stripped), dtype=dtype)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Column {col.name!r}: invalid JSON array cell {val!r}") from exc

            if arr.shape != item_shape:
                raise ValueError(f"Column {col.name!r}: expected item shape {item_shape}, got {arr.shape}")
            rows.append(arr)

        return np.ascontiguousarray(rows, dtype=dtype)

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
            shape = cls._column_physical_shape(col, capacity)
            chunks, blocks = cls._column_chunks_blocks(col, shape)
            new_cols[col.name] = mem_storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=shape,
                chunks=chunks,
                blocks=blocks,
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
                if isinstance(col.spec, NDArraySpec):
                    arr = cls._csv_ndarray_col_to_array(col_data[i], col)
                else:
                    nv = getattr(col.spec, "null_value", None)
                    arr = cls._csv_col_to_array(col_data[i], col, nv)
                new_cols[col.name][:n] = arr
            new_valid[:n] = True
            obj._n_rows = n
            obj._last_pos = n

        return obj

    # ------------------------------------------------------------------
    # Pandas / DataFrame interop
    # ------------------------------------------------------------------

    def to_pandas(self):
        """Convert to a `pandas <https://pandas.pydata.org>`_ DataFrame.

        Scalar columns become regular DataFrame columns.  Fixed-shape ndarray
        columns become ``object``-dtype columns whose cells hold NumPy arrays
        of per-row shape *item_shape*.

        Returns
        -------
        pandas.DataFrame

        Examples
        --------
        >>> import blosc2
        >>> from dataclasses import dataclass
        >>> import numpy as np
        >>> @dataclass
        ... class Row:
        ...     id: int = blosc2.field(blosc2.int64())
        ...     embedding: object = blosc2.field(blosc2.ndarray((3,), dtype=blosc2.float32()))
        >>> t = blosc2.CTable(Row, new_data=[
        ...     (1, np.array([1, 2, 3], dtype=np.float32)),
        ...     (2, np.array([4, 5, 6], dtype=np.float32)),
        ... ])
        >>> df = t.to_pandas()
        >>> df["id"].tolist()
        [1, 2]
        >>> df["embedding"].dtype
        dtype('O')
        >>> np.testing.assert_array_equal(df["embedding"][0], np.array([1, 2, 3], dtype=np.float32))
        """
        import pandas as pd

        data = {}
        for name in self.col_names:
            col = self[name]
            if col.is_ndarray:
                data[name] = list(col)
            else:
                data[name] = col[:]

        return pd.DataFrame(data)

    @classmethod
    def from_pandas(cls, df, row_cls) -> CTable:  # noqa: C901
        """Build a :class:`CTable` from a pandas DataFrame.

        Schema comes from *row_cls* (a dataclass) — CTable is always typed.
        Object-dtype DataFrame columns are **not** automatically inferred as
        ndarray columns; the *row_cls* must explicitly declare
        :func:`blosc2.ndarray` fields.

        Parameters
        ----------
        df:
            Source pandas DataFrame.
        row_cls:
            A dataclass whose fields define the column names and types.

        Returns
        -------
        CTable
            A new CTable containing all DataFrame rows.

        Raises
        ------
        TypeError
            If *row_cls* is not a dataclass.
        ValueError
            If DataFrame columns do not match the *row_cls* schema.
        """
        schema = compile_schema(row_cls)
        cls._resolve_nullable_specs(schema)

        # Validate column names
        schema_names = [col.name for col in schema.columns]
        missing = [name for name in schema_names if name not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns declared in row_cls: {missing}")
        extra = [name for name in df.columns if name not in schema_names]
        if extra:
            raise ValueError(f"DataFrame has extra columns not in row_cls: {extra}")

        n = len(df)
        capacity = max(n, 1)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        mem_storage = InMemoryTableStorage()

        new_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        new_cols: dict[str, Any] = {}
        for col in schema.columns:
            if cls._is_list_column(col):
                new_cols[col.name] = mem_storage.create_list_column(
                    col.name,
                    spec=col.spec,
                    cparams=None,
                    dparams=None,
                )
                continue
            if cls._is_varlen_scalar_column(col):
                new_cols[col.name] = mem_storage.create_varlen_scalar_column(
                    col.name,
                    spec=col.spec,
                    cparams=None,
                    dparams=None,
                )
                continue
            if cls._is_dictionary_column(col):
                dict_col = mem_storage.create_dictionary_column(
                    col.name,
                    spec=col.spec,
                    cparams=None,
                    dparams=None,
                )
                if len(dict_col.codes) < capacity:
                    dict_col.resize((capacity,))
                new_cols[col.name] = dict_col
                continue
            shape = cls._column_physical_shape(col, capacity)
            chunks, blocks = cls._column_chunks_blocks(col, shape)
            new_cols[col.name] = mem_storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=shape,
                chunks=chunks,
                blocks=blocks,
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
        obj._computed_cols = {}
        obj._materialized_cols = {}
        obj._expr_index_arrays = {}
        obj._valid_rows = new_valid
        obj._n_rows = 0
        obj._last_pos = 0

        if n > 0:

            def normalize_pandas_missing(value):
                if value is None:
                    return None
                if isinstance(value, float) and np.isnan(value):
                    return None
                # pandas.NA cannot be compared/coerced reliably; detect it by type name
                # without importing pandas here.
                if type(value).__name__ == "NAType":
                    return None
                return value

            raw_columns = {}
            for col in schema.columns:
                series = df[col.name]
                if isinstance(col.spec, NDArraySpec) and series.values.dtype != object:
                    raise ValueError(
                        f"Column {col.name!r}: expected object dtype in DataFrame "
                        f"for ndarray column, got {series.values.dtype}"
                    )
                if (
                    cls._is_list_column(col)
                    or cls._is_varlen_scalar_column(col)
                    or cls._is_dictionary_column(col)
                    or isinstance(col.spec, NDArraySpec)
                ):
                    raw_columns[col.name] = [normalize_pandas_missing(value) for value in series.tolist()]
                else:
                    raw_columns[col.name] = series.to_numpy(dtype=col.dtype)
            obj.extend(raw_columns, validate=True)

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

    def add_column(  # noqa: C901
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

        live_pos = self._live_positions_from_valid_rows_chunks()
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
                    if self._is_ndarray_column(compiled_col):
                        default_val = self._coerce_ndarray_value(name, spec, default)
                    else:
                        default_val = spec.dtype.type(default)
                except (ValueError, OverflowError) as exc:
                    raise TypeError(
                        f"Cannot coerce default {default!r} to dtype {spec.dtype!r}: {exc}"
                    ) from exc
            else:
                default_val = None

            capacity = len(self._valid_rows)
            shape = self._column_physical_shape(compiled_col, capacity)
            default_chunks, default_blocks = self._column_chunks_blocks(compiled_col, shape)
            col_storage = self._resolve_column_storage(compiled_col, default_chunks, default_blocks)
            new_col = self._storage.create_column(
                name,
                dtype=spec.dtype,
                shape=shape,
                chunks=col_storage["chunks"],
                blocks=col_storage["blocks"],
                cparams=col_storage.get("cparams"),
                dparams=col_storage.get("dparams"),
            )
            if len(live_pos) > 0:
                if self._is_ndarray_column(compiled_col):
                    new_col[live_pos] = np.broadcast_to(default_val, (len(live_pos), *spec.item_shape))
                else:
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
        dependents.extend(
            mat_name
            for mat_name, meta in self._materialized_cols.items()
            if name in meta.get("col_deps", ())
        )
        if dependents:
            raise ValueError(
                f"Cannot drop column {name!r}: it is used by computed/generated column(s) "
                + ", ".join(repr(d) for d in dependents)
                + ". Drop those columns first."
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

        Renaming a flat column to a dotted name (e.g. ``"trip.begin.lon"``)
        promotes it to a nested leaf column: it will be stored under the
        hierarchical path ``/_cols/trip/begin/lon`` on disk and can be
        accessed via ``t["trip.begin.lon"]`` or the attribute-chain proxy
        ``t.trip.begin.lon``.  This is the primary way to define nested
        columns when importing from non-Arrow sources::

            t.rename_column("trip_begin_lon", "trip.begin.lon")
            t["trip.begin.lon"].mean()   # works as a regular Column

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
        old_compiled = self._schema.columns_by_name[old]
        self._col_widths.pop(old)
        self._col_widths[new] = max(len(new), old_compiled.display_width)

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

    def _ensure_generated_column_not_stale(self, name: str) -> None:
        meta = self._root_table._materialized_cols.get(name)
        if meta is not None and meta.get("stale", False):
            raise ValueError(
                f"Generated column {name!r} is stale because one or more source columns were modified. "
                f"Call refresh_generated_column({name!r}) before using it, or use t[{name!r}].read_stale() "
                "to explicitly read the last stored stale values."
            )

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
        self._ensure_generated_column_not_stale(name)
        col = self._cols[name]
        spec = self._schema.columns_by_name[name].spec
        if self._is_list_spec(spec) or isinstance(
            spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec, DictionarySpec)
        ):
            return col[positions]
        values = col[positions]
        if isinstance(spec, timestamp):
            return np.asarray(values).astype(f"datetime64[{spec.unit}]")
        return values

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
            materialized = []
            for name, meta in self._materialized_cols.items():
                entry = {
                    "name": name,
                    "computed_column": meta.get("computed_column"),
                    "expression": meta.get("expression"),
                    "col_deps": meta["col_deps"],
                    "dtype": str(meta["dtype"]),
                    "transformer_kind": meta.get("transformer_kind", "expression"),
                    "stale": bool(meta.get("stale", False)),
                }
                if "transformer" in meta:
                    entry["transformer"] = meta["transformer"]
                materialized.append(entry)
            d["materialized_columns"] = materialized
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
            loaded = {
                "computed_column": meta.get("computed_column"),
                "expression": meta.get("expression"),
                "col_deps": list(meta["col_deps"]),
                "dtype": np.dtype(meta["dtype"]),
                "transformer_kind": meta.get("transformer_kind", "expression"),
                "stale": bool(meta.get("stale", False)),
            }
            if "transformer" in meta:
                loaded["transformer"] = dict(meta["transformer"])
            self._materialized_cols[meta["name"]] = loaded

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
            if meta.get("transformer_kind") == "row_transformer":
                transformer = RowTransformer.from_metadata(meta["transformer"])
                row[name] = np.asarray(transformer.evaluate_row(row), dtype=meta["dtype"])
            else:
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
            if meta.get("transformer_kind") == "row_transformer":
                transformer = RowTransformer.from_metadata(meta["transformer"])
                values = transformer.evaluate_batch(raw_columns)
            else:
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
    def _coerce_generated_spec(dtype_or_spec, sample: np.ndarray | None = None) -> SchemaSpec:
        """Resolve a generated-column dtype/spec, inferring ndarray shape when needed."""
        if isinstance(dtype_or_spec, SchemaSpec):
            return dtype_or_spec
        if dtype_or_spec is None:
            if sample is None:
                raise TypeError("dtype is required when a generated column has no rows to infer from.")
            arr = np.asarray(sample)
            if arr.ndim <= 1:
                return CTable._schema_spec_from_dtype(arr.dtype)
            return NDArraySpec(item_shape=arr.shape[1:], dtype=arr.dtype)
        return CTable._schema_spec_from_dtype(np.dtype(dtype_or_spec))

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
        dtype: np.dtype | None,
        *,
        spec: SchemaSpec | None = None,
        cparams: dict | None = None,
    ) -> None:
        """Create an empty stored column aligned with the table's physical row space."""
        if spec is None:
            if dtype is None:
                raise TypeError("dtype or spec is required")
            spec = self._schema_spec_from_dtype(dtype)
        dtype = np.dtype(spec.dtype)
        if isinstance(spec, NDArraySpec):
            default = np.zeros(spec.item_shape, dtype=dtype)
        else:
            default = np.array(0, dtype=dtype).item() if dtype.kind not in {"U", "S"} else dtype.type()

        capacity = len(self._valid_rows)
        compiled_col = CompiledColumn(
            name=name,
            py_type=spec.python_type,
            spec=spec,
            dtype=dtype,
            default=default,
            config=ColumnConfig(cparams=cparams, dparams=None, chunks=None, blocks=None),
            display_width=compute_display_width(spec),
        )
        shape = self._column_physical_shape(compiled_col, capacity)
        default_chunks, default_blocks = self._column_chunks_blocks(compiled_col, shape)
        new_col = self._storage.create_column(
            name,
            dtype=dtype,
            shape=shape,
            chunks=default_chunks,
            blocks=default_blocks,
            cparams=cparams,
            dparams=None,
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

    def _normalize_expression_transformer(self, expr) -> tuple[blosc2.LazyExpr, list[str]]:
        if isinstance(expr, RowTransformer):
            raise TypeError(
                "RowTransformer instances cannot be used for computed columns; use add_generated_column()."
            )
        if isinstance(expr, blosc2.LazyExpr):
            lazy = expr
        elif callable(expr):
            lazy = expr(self._cols)
        elif isinstance(expr, str):
            self._guard_scalar_expression(expr)
            operands = self._where_expression_operands()
            expr, operands = self._rewrite_nested_expression(expr, operands)
            lazy = blosc2.lazyexpr(expr, operands)
        else:
            raise TypeError(
                f"expr must be a callable or an expression string (or LazyExpr), got {type(expr).__name__!r}."
            )
        if not isinstance(lazy, blosc2.LazyExpr):
            raise TypeError(f"expr must return a blosc2.LazyExpr, got {type(lazy).__name__!r}.")

        owned_ids = {id(arr): cname for cname, arr in self._cols.items()}
        col_deps = []
        for key in sorted(lazy.operands.keys()):
            arr = lazy.operands[key]
            cname = owned_ids.get(id(arr))
            if cname is None:
                raise ValueError(
                    f"Operand {key!r} in the expression does not reference a stored column of this table."
                )
            self._ensure_generated_column_not_stale(cname)
            col_info = self._schema.columns_by_name.get(cname)
            if col_info is not None and self._is_ndarray_column(col_info):
                raise TypeError(
                    f"Column {cname!r} is a fixed-shape ndarray column. Expression transformers only "
                    "support scalar columns; use a RowTransformer for ndarray row reductions/projections."
                )
            col_deps.append(cname)
        return lazy, col_deps

    def _evaluate_expression_materialized_batch(
        self, meta: dict, raw_columns: Mapping[str, Any]
    ) -> np.ndarray:
        operands = {
            f"o{i}": blosc2.asarray(raw_columns[dep], dtype=self._cols[dep].dtype)
            for i, dep in enumerate(meta["col_deps"])
        }
        values = blosc2.lazyexpr(meta["expression"], operands)[:]
        return np.asarray(values, dtype=meta["dtype"])

    def _generated_dependency_closure(self, source: str) -> set[str]:
        """Return generated columns transitively depending on *source*."""
        affected: set[str] = set()
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for name, meta in self._materialized_cols.items():
                if name in affected:
                    continue
                if current in meta.get("col_deps", ()):
                    affected.add(name)
                    queue.append(name)
        return affected

    def _mark_generated_columns_stale(self, source: str) -> None:
        affected = self._generated_dependency_closure(source)
        changed = False
        for name in affected:
            meta = self._materialized_cols[name]
            if not meta.get("stale", False):
                meta["stale"] = True
                changed = True
        if changed and isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def refresh_generated_column(self, name: str) -> None:
        """Recompute a stored generated/materialized column from its source columns."""
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if name not in self._materialized_cols:
            raise KeyError(f"{name!r} is not a generated/materialized column.")
        meta = self._materialized_cols[name]
        live_pos = self._live_positions_from_valid_rows_chunks()
        if meta.get("transformer_kind") == "row_transformer":
            transformer = RowTransformer.from_metadata(meta["transformer"])
            values = np.asarray(transformer.evaluate_existing(self), dtype=meta["dtype"])
        else:
            raw_columns = {dep: self[dep][:] for dep in meta["col_deps"]}
            values = self._evaluate_expression_materialized_batch(meta, raw_columns)
        if len(values) != len(live_pos):
            raise ValueError(
                f"Generated column {name!r} produced {len(values)} values, expected {len(live_pos)}."
            )
        self._cols[name][live_pos] = values
        meta["stale"] = False
        self._mark_all_indexes_stale()
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def refresh_generated_columns(self, *, source: str | None = None) -> None:
        """Refresh all generated columns, optionally only those depending on *source*."""
        affected = None if source is None else self._generated_dependency_closure(source)
        for name in list(self._materialized_cols):
            if affected is None or name in affected:
                self.refresh_generated_column(name)

    def add_generated_column(  # noqa: C901
        self,
        name: str,
        *,
        values: str | blosc2.LazyExpr | Callable[[dict[str, Any]], blosc2.LazyExpr] | RowTransformer,
        dtype=None,
        create_index: bool = False,
    ) -> None:
        """Add a stored generated column maintained by the table.

        A generated column is physical storage, not a virtual expression.  The
        initial values are computed for all current live rows, and later
        ``append()`` / ``extend()`` calls automatically compute values for newly
        inserted rows when source columns are provided.  If a source column is
        modified in-place, dependent generated columns are marked stale; call
        :meth:`refresh_generated_column` or :meth:`refresh_generated_columns` to
        recompute them.

        Supported signatures are::

            add_generated_column(name, *, values="price * qty", dtype=..., create_index=False)
            add_generated_column(name, *, values=lazy_expr, dtype=..., create_index=False)
            add_generated_column(name, *, values=lambda cols: cols["price"] * 1.21, dtype=...)
            add_generated_column(name, *, values=t.embedding.row_transformer.norm(axis=0), dtype=...)
            add_generated_column(name, *, values=t.image.row_transformer.mean(axis=(0, 1)),
                                 dtype=blosc2.ndarray((3,), dtype=...))

        Parameters
        ----------
        name:
            Name of the generated column to create.  It must be a valid column
            name and must not collide with an existing stored or computed
            column.
        values:
            Definition used to compute the generated values.  Accepted forms:

            * ``str``: scalar expression over stored scalar columns, e.g.
              ``"price * qty"``.  The expression must produce one scalar value
              per row.
            * :class:`blosc2.LazyExpr`: scalar lazy expression over stored
              columns of this table.  It must produce a 1-D scalar stream.
            * callable: called as ``values(self._cols)`` and must return a
              :class:`blosc2.LazyExpr` over stored columns of this table.
            * :class:`RowTransformer`: row-wise projection/reduction bound to a
              fixed-shape ndarray column, e.g.
              ``t.embedding.row_transformer.norm(axis=0)`` or
              ``t.image.row_transformer.mean(axis=(0, 1))``.  Row transformers
              may produce either one scalar per row or one fixed-shape ndarray
              item per row.

            Expression forms currently cannot depend on computed columns and
            cannot directly consume fixed-shape ndarray columns; use a
            row-transformer for ndarray row projections/reductions.
        dtype:
            Output schema or dtype.  Scalar outputs may pass a NumPy dtype or a
            Blosc2 scalar spec such as ``blosc2.float64()``.  Fixed-shape
            ndarray outputs must pass an ndarray spec such as
            ``blosc2.ndarray((3,), dtype=blosc2.float32())`` unless the table has
            existing rows from which the output shape can be inferred.  When
            omitted, dtype and fixed-shape output shape are inferred from the
            current generated values; this is not possible for an empty table.
        create_index:
            If ``True``, create an index on the generated column immediately.
            Only scalar generated columns can be indexed; fixed-shape ndarray
            generated columns raise :class:`ValueError` when indexing is
            requested.

        Examples
        --------
        Create and index a scalar generated column from a string expression::

            t.add_generated_column(
                "total",
                values="price * qty",
                dtype=blosc2.float64(),
                create_index=True,
            )

        Use a callable when normal Python composition is more convenient::

            t.add_generated_column(
                "price_with_tax",
                values=lambda cols: cols["price"] * 1.21,
                dtype=blosc2.float64(),
            )

        Generate a scalar from each fixed-shape ndarray row.  For row
        transformers, axes refer to the per-row item shape, so ``axis=0`` is the
        embedding-coordinate axis for ``item_shape=(dim,)``::

            t.add_generated_column(
                "embedding_norm",
                values=t.embedding.row_transformer.norm(axis=0, ord=2),
                dtype=blosc2.float64(),
                create_index=True,
            )

        Generate a fixed-shape ndarray value per row.  Here an image column has
        ``item_shape=(height, width, 3)`` and the generated column stores one RGB
        vector per row::

            t.add_generated_column(
                "image_mean_rgb",
                values=t.image.row_transformer.mean(axis=(0, 1)),
                dtype=blosc2.ndarray((3,), dtype=blosc2.float32()),
            )

        Generated columns are maintained on append/extend::

            t.append((new_id, new_embedding, new_image))
            assert t.embedding_norm[-1] == np.linalg.norm(new_embedding)

        If source values are changed in place, refresh dependent generated
        columns before relying on them::

            t.embedding[0] = new_embedding
            t.refresh_generated_column("embedding_norm")

        Raises
        ------
        ValueError
            If called on a view or read-only table, if *name* already exists,
            if generated output length/shape is incompatible with the table, or
            if ``create_index=True`` is requested for an ndarray generated
            column.
        TypeError
            If *values* has an unsupported form, references unsupported source
            columns, or cannot be coerced to *dtype*.
        KeyError
            If a :class:`RowTransformer` references a missing source column.
        """
        if self.base is not None:
            raise ValueError("Cannot add a generated column to a view.")
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        _validate_column_name(name)
        if name in self._cols:
            raise ValueError(f"A stored column named {name!r} already exists.")
        if name in self._computed_cols:
            raise ValueError(f"A computed column named {name!r} already exists.")

        live_pos = self._live_positions_from_valid_rows_chunks()
        if isinstance(values, RowTransformer):
            transformer = values
            for dep in transformer.source_columns:
                if dep not in self._cols:
                    raise KeyError(f"No source column named {dep!r}.")
                col_info = self._schema.columns_by_name[dep]
                if not self._is_ndarray_column(col_info):
                    raise TypeError(f"RowTransformer source {dep!r} is not an ndarray column.")
            generated_values = (
                transformer.evaluate_existing(self)
                if len(live_pos)
                else transformer.evaluate_batch(
                    {
                        transformer.source: np.zeros(
                            (1, *self._schema.columns_by_name[transformer.source].spec.item_shape),
                            dtype=self._cols[transformer.source].dtype,
                        )
                    }
                )[:0]
            )
            spec = self._coerce_generated_spec(dtype, generated_values)
            metadata = {
                "computed_column": None,
                "expression": None,
                "col_deps": list(transformer.source_columns),
                "dtype": np.dtype(spec.dtype),
                "transformer_kind": "row_transformer",
                "transformer": transformer.to_metadata(),
                "stale": False,
            }
        else:
            lazy, col_deps = self._normalize_expression_transformer(values)
            generated_values = np.asarray(lazy[:])
            if generated_values.ndim != 1:
                raise TypeError("Expression generated columns must produce a 1-D scalar result.")
            generated_values = (
                generated_values[live_pos]
                if len(generated_values) == len(self._valid_rows)
                else generated_values
            )
            spec = self._coerce_generated_spec(dtype, generated_values)
            metadata = {
                "computed_column": None,
                "expression": lazy.expression,
                "col_deps": col_deps,
                "dtype": np.dtype(spec.dtype),
                "transformer_kind": "expression",
                "stale": False,
            }
        if create_index and isinstance(spec, NDArraySpec):
            raise ValueError("Generated columns intended for indexing must be 1-D scalar columns.")
        generated_values = np.asarray(generated_values, dtype=spec.dtype)
        if len(generated_values) != len(live_pos):
            raise ValueError(
                f"Generated column {name!r} produced {len(generated_values)} values, expected {len(live_pos)}."
            )
        if isinstance(spec, NDArraySpec) and generated_values.shape != (len(live_pos), *spec.item_shape):
            raise ValueError(
                f"Generated column {name!r} expected shape {(len(live_pos), *spec.item_shape)}, got {generated_values.shape}."
            )

        self._create_empty_stored_column(name, np.dtype(spec.dtype), spec=spec)
        self._materialized_cols[name] = metadata
        try:
            if len(live_pos):
                self._cols[name][live_pos] = generated_values
            if create_index:
                self.create_index(name)
        except Exception:
            with contextlib.suppress(Exception):
                self.drop_column(name)
            raise
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(self._schema_dict_with_computed())

    def add_computed_column(
        self,
        name: str,
        expr: str | blosc2.LazyExpr | Callable[[dict[str, Any]], blosc2.LazyExpr],
        *,
        dtype: np.dtype | None = None,
    ) -> None:
        """Add a read-only virtual column computed from stored columns.

        A computed column has no physical storage.  It is backed by a
        :class:`blosc2.LazyExpr` and is evaluated when values are read, filtered,
        displayed, exported, or aggregated.  Because it is virtual, it is
        read-only, cannot be indexed directly, and is not supplied in
        ``append()`` / ``extend()`` inputs.  To store and optionally index a
        computed result, use :meth:`add_generated_column` or materialize an
        existing computed column with :meth:`materialize_computed_column`.

        Supported signatures are::

            add_computed_column(name, "price * qty", dtype=None)
            add_computed_column(name, lazy_expr, dtype=None)
            add_computed_column(name, lambda cols: cols["price"] * cols["qty"], dtype=None)

        Parameters
        ----------
        name:
            Name of the virtual computed column.  It must be a valid column name
            and must not collide with an existing stored or computed column.
        expr:
            Definition of the virtual column.  Accepted forms:

            * ``str``: scalar expression over stored scalar columns, e.g.
              ``"price * qty"``.
            * :class:`blosc2.LazyExpr`: lazy expression over stored columns of
              this table.
            * callable: called as ``expr(self._cols)`` and must return a
              :class:`blosc2.LazyExpr` over stored columns of this table.

            Expressions must depend only on stored columns of this table;
            computed columns cannot depend on other computed columns in this
            version.  Fixed-shape ndarray columns are not accepted in computed
            column expressions yet.  For row-wise ndarray projections or
            reductions, use :meth:`add_generated_column` with
            ``values=t.ndarray_col.row_transformer...``.
        dtype:
            Optional dtype override for the computed values.  When omitted, the
            dtype is inferred from the resulting :class:`blosc2.LazyExpr`.
            This changes the dtype reported by the CTable column wrapper; it
            does not create physical storage.

        Examples
        --------
        Add a computed column from a string expression and use it like a normal
        read-only column::

            t.add_computed_column("total", "price * qty")
            assert t.total[:].shape == (t.nrows,)

        Add a computed column from a callable.  The callable receives the table's
        stored column mapping::

            t.add_computed_column(
                "price_with_tax",
                lambda cols: cols["price"] * 1.21,
                dtype=np.float64,
            )

        Callable expressions can use normal Python logic while still returning a
        lazy expression::

            def total_expr(cols):
                base = cols["price"] * cols["qty"]
                return base * 1.21 if include_tax else base

            t.add_computed_column("total", total_expr)

        They are also convenient for reusable, parameterized helpers::

            def ratio(num, den):
                return lambda cols: cols[num] / cols[den]

            t.add_computed_column("margin", ratio("profit", "revenue"))

        Computed columns participate in filters and aggregates::

            expensive = t.where(t.total > 100)
            total_revenue = t.total.sum()

        Computed columns are virtual and read-only.  Materialize one when a
        stored snapshot or an indexable column is needed::

            t.materialize_computed_column("total", new_name="total_stored")
            t.create_index("total_stored")

        For maintained stored results, prefer generated columns::

            t.add_generated_column(
                "total_stored",
                values="price * qty",
                dtype=blosc2.float64(),
                create_index=True,
            )

        Raises
        ------
        ValueError
            If called on a view or read-only table, if *name* already exists,
            or if an expression operand does not reference a stored column of
            this table.
        TypeError
            If *expr* has an unsupported form, does not produce a
            :class:`blosc2.LazyExpr`, references unsupported source columns, or
            if a :class:`RowTransformer` is passed.  Row transformers are only
            accepted by :meth:`add_generated_column`.
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

        lazy, col_deps = self._normalize_expression_transformer(expr)
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
            elif (
                self._is_list_column(col_info)
                or self._is_varlen_scalar_column(col_info)
                or self._is_dictionary_column(col_info)
            ):
                dtype = np.dtype(object)
            elif self._is_ndarray_column(col_info):
                fields.append((name, col_info.dtype, col_info.spec.item_shape))
                continue
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

    def _logical_to_physical_name(self, name: str) -> str:
        """Resolve a user/logical column path to a stored physical column name."""
        if name in self._cols or name in self._computed_cols:
            return name
        nested = self._schema.metadata.get("nested") if self._schema.metadata else None
        if isinstance(nested, dict):
            mapping = nested.get("logical_to_physical")
            if isinstance(mapping, dict):
                physical = mapping.get(name)
                if isinstance(physical, str) and (physical in self._cols or physical in self._computed_cols):
                    return physical
        return name

    def _expand_logical_column_selector(self, name: str) -> list[str]:
        """Resolve one logical selector to one or more physical column names.

        If *name* points to a scalar leaf, returns ``[leaf]``. If it points to
        a struct-like prefix (e.g. ``"trip"``), expands to descendant leaves.
        """
        physical = self._logical_to_physical_name(name)
        if physical in self._cols or physical in self._computed_cols:
            return [physical]
        prefix_parts = split_field_path(physical)
        expanded = [
            col for col in self.col_names if split_field_path(col)[: len(prefix_parts)] == prefix_parts
        ]
        if expanded:
            return expanded
        return [physical]

    def __getitem__(self, key):
        """Type-driven indexing for columns, rows, projections, and filters.

        Supported keys are:

        - ``str``: return a :class:`Column` when it matches a stored or computed
          column name; otherwise evaluate it as a boolean expression via
          :meth:`where`.  Dotted names (e.g. ``"trip.begin.lon"``) select
          nested leaf columns directly; a struct-prefix name
          (e.g. ``"trip.begin"``) that matches multiple descendant leaves returns
          a :class:`_StructPathColumn` view.  This item-access form is the
          canonical way to access columns and works for every column name,
          including names that are not valid Python identifiers or that collide
          with existing :class:`CTable` attributes or methods.
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

        Access a nested leaf column with a dotted name or an attribute chain::

            lons = t["trip.begin.lon"]   # Column for the nested leaf
            lons = t.trip.begin.lon      # equivalent attribute-chain form

        Attribute access is only a convenience fallback.  If a column name is
        not a valid identifier, or if it conflicts with an existing table
        attribute or method such as ``nrows``, ``where`` or ``sort_by``, use item
        access instead::

            col = t["where"]             # column named "where"
            method = t.where             # CTable.where method
        """
        if isinstance(key, str):
            physical = self._logical_to_physical_name(key)
            if physical in self._cols or physical in self._computed_cols:
                return Column(self, physical)
            expanded = self._expand_logical_column_selector(key)
            cc = self._schema.columns_by_name.get(physical)
            if len(expanded) > 1 or (expanded and cc is not None and isinstance(cc.spec, StructSpec)):
                return _StructPathColumn(self, physical, expanded)
            return self.where(key)
        if isinstance(key, (blosc2.NDArray, blosc2.LazyExpr)) and getattr(key, "dtype", None) == np.bool_:
            return self.where(key)
        if isinstance(key, tuple):
            raise TypeError("Tuple indexing is not supported for CTable in V1")
        return self._getitem_row_selector(key)

    def _nested_namespace(self, prefix: str):
        prefix_parts = split_field_path(prefix)
        for name in self.col_names:
            parts = split_field_path(name)
            if parts[: len(prefix_parts)] == prefix_parts and len(parts) > len(prefix_parts):
                return _NestedColumnNamespace(self, prefix)
        return None

    def __getattr__(self, s: str):
        """Convenience fallback for attribute-style column access.

        This is called only after normal Python attribute lookup fails.  Thus
        ``t.name`` can return a column only for non-conflicting identifier-like
        column names.  For columns whose names conflict with existing CTable
        attributes/methods, or are not valid identifiers, use the canonical item
        access form ``t["name"]``.
        """
        physical = self._logical_to_physical_name(s)
        if physical in self._cols or physical in self._computed_cols:
            return Column(self, physical)
        ns = self._nested_namespace(s)
        if ns is not None:
            return ns
        return super().__getattribute__(s)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self):
        """Physically rewrite every column array keeping only live rows.

        Closes the gaps left by prior :meth:`delete` calls by shuffling live
        data to the front of each column array.  The underlying NDArray
        allocations are **not resized** — each column retains its original
        capacity.  To actually reclaim memory, use :meth:`copy` with
        ``compact=True`` instead, which allocates fresh arrays sized to the
        live row count.  All existing indexes are dropped and must be
        recreated afterwards.  Raises ``ValueError`` if the table is
        read-only or a view.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot compact a view.")
        if self._last_pos is not None and self._last_pos == self._n_rows:
            return
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
            if self._is_dictionary_column(col):
                # Keep dictionary values intact; just compact the codes.
                live_codes = v.codes[real_poss[: self._n_rows]]
                v.codes[: self._n_rows] = live_codes
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

    @staticmethod
    def _column_selector_name(value: Any) -> str:
        """Return the column name represented by a string or Column-like selector."""
        name = getattr(value, "_col_name", value)
        if not isinstance(name, str):
            raise TypeError(f"Expected a column name or Column object, got {type(value)!r}")
        return name

    def _normalise_sort_keys(
        self,
        cols: str | list[str],
        ascending: bool | list[bool],
    ) -> tuple[list[str], list[bool]]:
        """Validate and normalise sort key arguments; return (cols, ascending)."""
        if isinstance(cols, str) or isinstance(getattr(cols, "_col_name", None), str):
            cols = [self._column_selector_name(cols)]
        else:
            cols = [self._column_selector_name(col) for col in cols]

        resolved_cols: list[str] = []
        for name in cols:
            expanded = self._expand_logical_column_selector(name)
            if len(expanded) != 1:
                raise ValueError(
                    f"Sort key {name!r} resolves to multiple columns {expanded!r}; please choose a leaf column."
                )
            resolved_cols.append(expanded[0])
        cols = resolved_cols
        if isinstance(ascending, bool):
            ascending = [ascending] * len(cols)
        if len(cols) != len(ascending):
            raise ValueError(
                f"'ascending' must have the same length as 'cols' ({len(cols)}), got {len(ascending)}."
            )
        for name in cols:
            if name not in self._cols and name not in self._computed_cols:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
            self._ensure_generated_column_not_stale(name)
            col_info = self._schema.columns_by_name.get(name)
            if col_info is not None and self._is_ndarray_column(col_info):
                raise TypeError(
                    f"Cannot sort by ndarray column {name!r} with per-row shape {col_info.spec.item_shape}. "
                    "Materialize a scalar generated column first, e.g. embedding_norm or embedding_max."
                )
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
                col_info = self._schema.columns_by_name.get(name)
                if col_info is not None and self._is_dictionary_column(col_info):
                    # Sort dictionary columns by decoded string values.
                    decoded = self._cols[name][live_pos]
                    raw = np.array(decoded, dtype=object)
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
            the tiebreaker, and so on.  For tables with **nested (dotted)
            column names**, pass the dotted leaf name directly::

                t.sort_by("trip.begin.lon")
                t.sort_by(["trip.begin.lon", "payment.fare"], ascending=[True, False])

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

        # Live physical positions.  Scan the validity NDArray chunk-wise to avoid
        # materialising the whole mask as a single NumPy array.
        live_pos = self._live_positions_from_valid_rows_chunks()
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

        # For small filtered views, materialise selected columns once and sort those
        # small arrays in memory.  This avoids gathering the sort keys and then
        # gathering the result columns again from the large backing arrays.
        if (
            sorted_pos is None
            and not inplace
            and self.base is not None
            and n <= _SMALL_SORT_MATERIALIZE_LIMIT
            and not any(
                self._is_list_column(col) or self._is_varlen_scalar_column(col)
                for col in self._schema.columns
            )
        ):
            return self._sorted_small_copy_from_live_positions(cols, ascending, live_pos, n)

        if sorted_pos is None:
            order = np.lexsort(self._build_lex_keys(cols, ascending, live_pos, n))
            sorted_pos = live_pos[order]

        if inplace:
            self._sort_by_inplace(sorted_pos, n)
            return self

        return self._sorted_copy_from_positions(sorted_pos, n)

    def _sorted_small_copy_from_live_positions(
        self, cols: list[str], ascending: list[bool], live_pos: np.ndarray, n: int
    ) -> CTable:
        """Materialise and sort a small filtered view, avoiding a second gather of sort keys."""
        gathered = {}
        for col in self._schema.columns:
            arr = self._cols[col.name]
            if self._is_dictionary_column(col):
                gathered[col.name] = arr.codes[live_pos]
            else:
                gathered[col.name] = arr[live_pos]

        lex_keys = []
        for name, asc in zip(reversed(cols), reversed(ascending), strict=True):
            col_info = self._schema.columns_by_name.get(name)
            if col_info is not None and self._is_dictionary_column(col_info):
                raw = np.array(self._cols[name][live_pos], dtype=object)
            else:
                raw = gathered[name]

            if not asc:
                if raw.dtype.kind in "USO":
                    rank = np.argsort(np.argsort(raw, kind="stable"), kind="stable")
                    lex_keys.append((n - 1 - rank).astype(np.intp))
                elif np.issubdtype(raw.dtype, np.unsignedinteger):
                    lex_keys.append(-raw.astype(np.int64))
                else:
                    lex_keys.append(-raw)
            else:
                lex_keys.append(raw)

            nv = getattr(col_info.spec, "null_value", None) if col_info else None
            if nv is not None:
                if isinstance(nv, float) and np.isnan(nv):
                    null_ind = np.isnan(raw).astype(np.intp)
                else:
                    null_ind = (raw == nv).astype(np.intp)
                lex_keys.append(null_ind)

        order = np.lexsort(lex_keys)
        result = self._empty_copy(capacity=n)
        for col in self._schema.columns:
            col_name = col.name
            if self._is_dictionary_column(col):
                for v in self._cols[col_name].dictionary:
                    result._cols[col_name].encode(v)
                result._cols[col_name].codes[:n] = gathered[col_name][order]
            else:
                result._cols[col_name][:n] = gathered[col_name][order]
        result._valid_rows[:n] = True
        result._valid_rows[n:] = False
        result._n_rows = n
        result._last_pos = n
        return result

    def _sort_by_inplace(self, sorted_pos: np.ndarray, n: int) -> None:
        for col in self._schema.columns:
            arr = self._cols[col.name]
            if self._is_list_column(col):
                new_arr = ListArray(spec=col.spec)
                new_arr.extend((arr[int(pos)] for pos in sorted_pos), validate=False)
                new_arr.flush()
                self._cols[col.name] = new_arr
            elif self._is_dictionary_column(col):
                sorted_codes = arr.codes[sorted_pos]
                arr.codes[:n] = sorted_codes
            else:
                arr[:n] = arr[sorted_pos]
        self._valid_rows[:n] = True
        self._valid_rows[n:] = False
        self._n_rows = n
        self._last_pos = n
        self._mark_all_indexes_stale()

    def _sorted_copy_from_positions(self, sorted_pos: np.ndarray, n: int) -> CTable:
        # Build a new in-memory table with the sorted rows
        result = self._empty_copy()
        for col in self._schema.columns:
            col_name = col.name
            arr = self._cols[col_name]
            if self._is_list_column(col):
                result._cols[col_name].extend((arr[int(pos)] for pos in sorted_pos), validate=False)
                result._cols[col_name].flush()
            elif self._is_dictionary_column(col):
                # Copy dictionary values, then sorted codes.
                for v in arr.dictionary:
                    result._cols[col_name].encode(v)
                sorted_codes = arr.codes[sorted_pos]
                result._cols[col_name].codes[:n] = sorted_codes
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

        This is the only operation that truly reclaims memory: when
        ``compact=True`` the new table allocates fresh arrays sized exactly
        to the live row count, discarding all deleted-row gaps and unused
        capacity.

        Parameters
        ----------
        compact:
            If ``True`` (default), only live (non-deleted) rows are copied.
            The result is a dense table with no tombstones and no parent
            dependency — ideal for materialising a filtered view or freeing
            memory after heavy deletions.
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
            elif self._is_dictionary_column(col):
                # Copy dictionary values, then copy (live) codes.
                for v in arr.dictionary:
                    result._cols[col_name].encode(v)
                pos_slice = live_pos if compact else np.arange(n, dtype=np.int64)
                raw_codes = arr.codes[pos_slice]
                result._cols[col_name].codes[:n] = raw_codes
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
            elif self._is_varlen_scalar_column(col):
                new_cols[col.name] = mem_storage.create_varlen_scalar_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
            elif self._is_dictionary_column(col):
                dict_col = mem_storage.create_dictionary_column(
                    col.name,
                    spec=col.spec,
                    cparams=col_storage.get("cparams"),
                    dparams=col_storage.get("dparams"),
                )
                if len(dict_col.codes) < capacity:
                    dict_col.codes.resize((capacity,))
                new_cols[col.name] = dict_col
            else:
                shape = self._column_physical_shape(col, capacity)
                chunks = col_storage["chunks"]
                blocks = col_storage["blocks"]
                if col.config.chunks is None and col.config.blocks is None:
                    chunks, blocks = self._column_chunks_blocks(col, shape)
                new_cols[col.name] = mem_storage.create_column(
                    col.name,
                    dtype=col.dtype,
                    shape=shape,
                    chunks=chunks,
                    blocks=blocks,
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

    # Cost-model constants for cross-column index refinement.
    # Calibrated from profiling with sparse-gather optimisations.
    #   _GATHER_COST_MS_PER_1K_ITEMS_PER_OP  ≈ ms to sparse-gather 1000 items from one operand column
    #   _SCAN_COST_MS_PER_1M_ROWS             ≈ ms to miniexpr-scan 1 million rows
    # If refinement cost exceeds scan cost, fall back to a full scan.
    _GATHER_COST_MS_PER_1K_ITEMS_PER_OP: float = 3.5
    _SCAN_COST_MS_PER_1M_ROWS: float = 4.3

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
        if kind not in {"summary", "bucket", "partial", "full", "opsi"}:
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
            col_name = self._logical_to_physical_name(col_name)
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
        """Build and register an index for a stored column or table expression.

        For tables with **nested (dotted) column names**, pass the dotted leaf
        name directly::

            t.create_index("trip.begin.lon")
            t.where("trip.begin.lon > -87.7").nrows   # index is used automatically

        .. rubric:: Choosing an index kind

        ``BUCKET`` (the default) is the cheapest to build and store.
        It accelerates single‑column ``where`` queries and ``sort_by``
        reuse with approximate ordering derived from value
        quantization.  Sufficient for most workloads.

        ``FULL`` builds a globally sorted index that returns exact
        row positions for any range predicate.  It enables the
        **cross‑column refinement** planner path: when a multi‑column
        conjunction such as ``(tips > 100) & (km > 0) & (sec > 0)``
        indexes only the most selective column, the planner obtains
        compact exact positions from ``FULL`` and evaluates the
        remaining predicates on just those rows.  ``FULL`` is also
        ideal for ``sort_by`` reuse because it carries a complete
        sort order.

        ``PARTIAL`` builds a chunk‑local sorted payload with segment
        navigation.  It is cheaper to build than ``FULL`` (roughly
        half the raw storage) while still providing exact positions
        for cross‑column refinement.  Its exact positions are most
        compact for equality or narrow range queries; wide ranges
        may scan proportionally more candidate segments.

        ``OPSI`` is a specialised tier for approximate ordering;
        prefer ``FULL`` when a globally sorted ordered index is
        needed to accelerate ``sort_by``.

        ``SUMMARY`` stores only per‑segment min/max and is the
        lightest kind; it may still skip chunks for broad range
        queries but cannot accelerate ``sort_by``.
        """
        if self.base is not None:
            raise ValueError("Cannot create an index on a view.")
        if col_name is not None and field is not None:
            raise ValueError("col_name and field are mutually exclusive")
        if expression is not None and (col_name is not None or field is not None):
            raise ValueError("column targets and expression are mutually exclusive")
        if operands is not None and expression is None:
            raise ValueError("operands can only be provided together with expression")
        col_name = field if field is not None else col_name
        if col_name is not None:
            col_name = self._logical_to_physical_name(col_name)

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
        self._ensure_generated_column_not_stale(col_name)
        if col_name in catalog:
            raise ValueError(
                f"Index already exists for column {col_name!r}. "
                "Call rebuild_index() to replace it or drop_index() first."
            )

        col_arr = self._cols[col_name]
        if isinstance(self._schema.columns_by_name[col_name].spec, NDArraySpec):
            spec = self._schema.columns_by_name[col_name].spec
            raise ValueError(
                f"Cannot create an index on ndarray column {col_name!r} with per-row shape {spec.item_shape}. "
                "Materialize a scalar generated column first, e.g. embedding_norm or embedding_max."
            )
        if isinstance(self._schema.columns_by_name[col_name].spec, ListSpec):
            raise ValueError(f"Cannot create an index on list column {col_name!r} in V1.")
        if isinstance(
            self._schema.columns_by_name[col_name].spec, (VLStringSpec, VLBytesSpec, StructSpec, ObjectSpec)
        ):
            raise NotImplementedError(
                f"Cannot create an index on variable-length scalar column {col_name!r}: "
                "indexing for vlstring/vlbytes/struct/object columns is not supported yet."
            )
        # Dictionary columns: index the underlying int32 codes array.
        is_dictionary = isinstance(self._schema.columns_by_name[col_name].spec, DictionarySpec)
        if is_dictionary:
            col_arr = col_arr.codes  # index the int32 codes NDArray
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
    def _evaluate_refine_predicate(col_values, refine_plan) -> np.ndarray:
        """Evaluate a single comparison predicated on *col_values*.

        ``refine_plan`` is an :class:`~blosc2.indexing.ExactPredicatePlan`
        that carries ``lower`` / ``upper`` bounds and their inclusiveness.
        Returns a boolean mask of the same length as *col_values*.
        """
        mask = np.ones(len(col_values), dtype=bool)
        if refine_plan.lower is not None:
            if refine_plan.lower_inclusive:
                mask &= col_values >= refine_plan.lower
            else:
                mask &= col_values > refine_plan.lower
        if refine_plan.upper is not None:
            if refine_plan.upper_inclusive:
                mask &= col_values <= refine_plan.upper
            else:
                mask &= col_values < refine_plan.upper
        return mask

    @staticmethod
    def _evaluate_expression_at(expr_result, candidates, *, prefetched: dict | None = None):
        """Evaluate *expr_result* on the operand rows at *candidates*.

        Returns a boolean ``numpy.ndarray`` the same length as *candidates*,
        or ``None`` if evaluation fails.

        Parameters
        ----------
        prefetched:
            Optional dict mapping operand variable names to already-gathered
            NumPy arrays.  When provided, those operands are reused instead of
            re-read from storage.
        """
        try:
            operands = {}
            for var_name, arr in expr_result.operands.items():
                if prefetched is not None and var_name in prefetched:
                    sliced = prefetched[var_name]
                else:
                    sliced = arr[candidates]
                    if hasattr(sliced, "__array__"):
                        sliced = np.asarray(sliced)
                operands[var_name] = sliced
            return blosc2.evaluate(expr_result.expression, operands)
        except Exception:
            return None

    @staticmethod
    def _find_indexed_columns(root_cols, catalog, operands):
        """Return live indexed columns referenced by *operands* in expression order.

        Avoid iterating over ``root_cols.items()`` here: for lazy persistent tables
        that would open every column just to find the indexed operands.
        """
        indexed = []
        seen = set()
        indexed_arrays = {}
        for col_name, descriptor in catalog.items():
            if col_name in root_cols:
                indexed_arrays[col_name] = (root_cols[col_name], descriptor)

        for operand in operands.values():
            if not isinstance(operand, blosc2.NDArray):
                continue
            for col_name, (col_arr, descriptor) in indexed_arrays.items():
                if col_name in seen or col_arr is not operand:
                    continue
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

        for _col_name, col_arr, descriptor in indexed_columns[:1]:
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

        if plan.partial_exact_positions is not None:
            # Cross-column refinement: the FULL index on one column gave us
            # exact positions, but the expression has additional predicates on
            # other columns.  Refinement reads every operand column at those
            # candidate positions using sparse/fancy indexing.  For compressed
            # columns this can touch many chunks and be slower than the regular
            # sequential miniexpr scan, which is very fast for simple predicates.
            # Use a cost model to compare refinement vs full scan.
            candidates = np.asarray(plan.partial_exact_positions, dtype=np.int64)
            n_candidates = len(candidates)
            n_operands = len(expr_result.operands)
            target_len = len(root._valid_rows)

            estimated_refine_ms = (
                (n_candidates / 1000.0) * CTable._GATHER_COST_MS_PER_1K_ITEMS_PER_OP * n_operands
            )
            estimated_scan_ms = (target_len / 1_000_000.0) * CTable._SCAN_COST_MS_PER_1M_ROWS
            if estimated_refine_ms > estimated_scan_ms:
                return None

            # Read the primary column once and reuse for both null filtering
            # and refinement, avoiding a second sparse gather later.
            primary_op_name = next(
                (vn for vn, va in expr_result.operands.items() if va is primary_col_arr), None
            )
            prefetched = None
            if nullable_indexed and primary_op_name is not None:
                raw = primary_col_arr[candidates]
                raw = np.asarray(raw) if hasattr(raw, "__array__") else raw
                pos = candidates
                for name in nullable_indexed:
                    if name == primary_col_name:
                        nv = getattr(root._schema.columns_by_name[name].spec, "null_value", None)
                        if isinstance(nv, float) and np.isnan(nv):
                            keep = ~np.isnan(raw)
                        else:
                            keep = raw != nv
                        pos = pos[keep]
                        raw = raw[keep]  # already filtered for refinement reuse
                    else:
                        col = root._schema.columns_by_name[name]
                        vals = root._cols[name][pos]
                        nv = getattr(col.spec, "null_value", None)
                        if isinstance(nv, float) and np.isnan(nv):
                            keep = ~np.isnan(vals)
                        else:
                            keep = vals != nv
                        pos = pos[keep]
                candidates = pos
                prefetched = {primary_op_name: raw}
            else:
                candidates = _exclude_null_positions(candidates)

            restricted = self._evaluate_expression_at(expr_result, candidates, prefetched=prefetched)
            if restricted is not None and restricted.dtype == np.bool_:
                refined = candidates[np.asarray(restricted, dtype=bool)]
                return _exclude_null_positions(refined)
            # Fall through to full scan if refinement fails

        if plan.bucket_masks is not None:
            # When bucket pruning covers all units (100 % of chunks are
            # candidates), the per‑chunk evaluation overhead outweighs the
            # benefit over a plain scan.  Fall back to the scan path.
            if plan.total_units > 0 and plan.selected_units >= plan.total_units:
                return None
            _, positions = evaluate_bucket_query(
                expression, merged_operands, {}, where_dict, plan, return_positions=True
            )
            return _exclude_null_positions(positions)

        if plan.candidate_units is not None and plan.segment_len is not None:
            # When segment summaries prune fewer than half the candidate
            # units, the per‑segment evaluation overhead outweighs a plain
            # scan.  Fall back to the scan path.
            if plan.total_units > 0 and plan.selected_units / plan.total_units > 0.5:
                return None
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
        if isinstance(spec, DictionarySpec):
            ordered_tag = ", ordered" if spec.ordered else ""
            return f"dictionary[str{ordered_tag}]"
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
        if isinstance(spec, NDArraySpec):
            return spec.display_label()
        if isinstance(spec, timestamp):
            return (
                f"timestamp[{spec.unit}]"
                if spec.timezone is None
                else f"timestamp[{spec.unit}, {spec.timezone}]"
            )
        if dtype is None:
            return "None"
        if dtype.kind == "U":
            nchars = dtype.itemsize // 4
            return f"U{nchars} (Unicode)"
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

        For tables with **nested (dotted) column names** the row dict may be
        supplied either as a flat mapping of dotted keys or as a nested dict
        that mirrors the original struct shape — both are accepted and
        automatically flattened to the physical dotted leaf names::

            # flat dotted keys
            t.append({"trip.begin.lon": -87.6, "trip.begin.lat": 41.8,
                      "payment.fare": 12.5})

            # original nested dict (auto-flattened)
            t.append({"trip": {"begin": {"lon": -87.6, "lat": 41.8}},
                      "payment": {"fare": 12.5}})
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
            elif self._is_dictionary_column(col):
                col_array[pos] = row[name]  # DictionaryColumn encodes on __setitem__
            else:
                col_array[pos] = row[name]

        n_rows = self.nrows
        self._valid_rows[pos] = True
        self._last_pos = pos + 1
        self._n_rows = n_rows + 1
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
        true_pos = self._live_positions_from_valid_rows_chunks()

        if isinstance(ind, Iterable) and not isinstance(ind, (str, bytes)):
            ind = list(ind)
        elif not isinstance(ind, int) and not isinstance(ind, slice):
            raise TypeError(f"Invalid type '{type(ind)}'")

        false_pos = true_pos[ind]
        n_deleted = len(np.unique(false_pos))
        n_rows = self.nrows

        self._valid_rows[false_pos] = False
        self._n_rows = n_rows - n_deleted
        if self._last_pos is None or np.any(false_pos == self._last_pos - 1):
            self._last_pos = None  # last live row deleted; recalculate on next write
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

        For tables with **nested (dotted) column names** both the dict-of-arrays
        and list-of-dicts forms accept the original nested dict shape and
        auto-flatten it to physical dotted leaf names::

            # nested dict of arrays
            t.extend({
                "trip": {"begin": {"lon": lons, "lat": lats}},
                "payment": {"fare": fares},
            })

            # list of nested dicts
            t.extend([
                {"trip": {"begin": {"lon": -87.6, "lat": 41.8}}, "payment": {"fare": 12.5}},
                {"trip": {"begin": {"lon": -87.5, "lat": 41.7}}, "payment": {"fare": 8.0}},
            ])
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
                if any(isinstance(v, dict) for v in data.values()):
                    data = self._flatten_nested_dict(data)
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
            elif data and isinstance(data[0], dict):
                # List of dicts: flatten any nested dicts and pivot to column arrays.
                flat_rows = [
                    self._flatten_nested_dict(row) if any(isinstance(v, dict) for v in row.values()) else row
                    for row in data
                ]
                new_nrows = len(flat_rows)
                col_set = set(input_col_names)
                raw_columns = {
                    name: [row[name] for row in flat_rows]
                    for name in input_col_names
                    if name in flat_rows[0]
                }
                provided_names = set(raw_columns)
                # Fill any remaining columns from the rows (may include extra keys)
                for row in flat_rows:
                    for key in row:
                        if key in col_set and key not in raw_columns:
                            raw_columns[key] = [r.get(key) for r in flat_rows]
                            provided_names.add(key)
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
        dict_processed_cols: dict[str, list] = {}
        for name in current_col_names:
            col_meta = self._schema.columns_by_name[name]
            if self._is_list_column(col_meta):
                list_processed_cols[name] = list(raw_columns[name])
            elif self._is_varlen_scalar_column(col_meta):
                varlen_scalar_processed_cols[name] = list(raw_columns[name])
            elif self._is_dictionary_column(col_meta):
                dict_processed_cols[name] = list(raw_columns[name])
            else:
                target_dtype = self._cols[name].dtype
                if isinstance(col_meta.spec, timestamp):
                    values = np.asarray(raw_columns[name])
                    if np.issubdtype(values.dtype, np.datetime64):
                        values = values.astype(f"datetime64[{col_meta.spec.unit}]").astype(np.int64)
                    elif values.dtype.kind in "OUS":
                        values = np.array(
                            [
                                col_meta.spec.null_value
                                if v is None
                                else np.datetime64(v)
                                .astype(f"datetime64[{col_meta.spec.unit}]")
                                .astype(np.int64)
                                if isinstance(v, (np.datetime64, str)) or hasattr(v, "isoformat")
                                else v
                                for v in values
                            ],
                            dtype=target_dtype,
                        )
                    scalar_processed_cols[name] = np.ascontiguousarray(values, dtype=target_dtype)
                elif self._is_ndarray_column(col_meta):
                    scalar_processed_cols[name] = self._coerce_ndarray_batch(
                        name, col_meta.spec, raw_columns[name], new_nrows
                    )
                else:
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
            elif self._is_dictionary_column(col_meta):
                # DictionaryColumn.__setitem__ with a slice encodes all values.
                self._cols[name][start_pos:end_pos] = dict_processed_cols[name]
            else:
                self._cols[name][start_pos:end_pos] = scalar_processed_cols[name][:]

        n_rows = self.nrows
        self._valid_rows[start_pos:end_pos] = True
        self._last_pos = end_pos
        self._n_rows = n_rows + new_nrows
        self._mark_all_indexes_stale()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _where_expression_operands(self) -> dict[str, blosc2.NDArray | blosc2.LazyExpr]:
        operands = {}
        for name, arr in self._cols.items():
            col = self._schema.columns_by_name.get(name)
            if col is not None and not (
                self._is_list_column(col)
                or self._is_varlen_scalar_column(col)
                or self._is_dictionary_column(col)
                or self._is_ndarray_column(col)
            ):
                operands[name] = arr
        operands.update({name: cc["lazy"] for name, cc in self._computed_cols.items()})
        return operands

    def _rewrite_nested_expression(
        self, expr: str, operands: dict[str, blosc2.NDArray | blosc2.LazyExpr]
    ) -> tuple[str, dict[str, blosc2.NDArray | blosc2.LazyExpr]]:
        """Rewrite dotted nested names in *expr* to safe identifiers.

        `blosc2.lazyexpr` does not accept dotted identifiers, but nested leaf
        columns are naturally addressed as dotted paths (e.g. ``trip.begin.lon``).
        This maps them to temporary aliases and returns rewritten expression and
        operand mapping.
        """
        dotted = [name for name in operands if "." in name]
        if not dotted:
            return expr, operands

        rewritten = expr
        new_operands = dict(operands)
        # Longest names first so trip.begin.lon is rewritten before trip.begin.
        for i, name in enumerate(sorted(dotted, key=len, reverse=True)):
            alias = f"__nf{i}"
            pattern = rf"(?<![\w.]){re.escape(name)}(?![\w.])"
            replaced = re.sub(pattern, alias, rewritten)
            if replaced != rewritten:
                rewritten = replaced
                new_operands[alias] = new_operands.pop(name)
        return rewritten, new_operands

    @staticmethod
    def _expression_references_name(expr: str, name: str) -> bool:
        return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w.])", expr) is not None

    def _guard_scalar_expression(self, expr: str) -> None:
        for name, meta in self._root_table._materialized_cols.items():
            if meta.get("stale", False) and self._expression_references_name(expr, name):
                raise ValueError(
                    f"Generated column {name!r} is stale because one or more source columns were modified. "
                    f"Call refresh_generated_column({name!r}) before using it in expressions, or use "
                    f"t[{name!r}].read_stale() to explicitly read the last stored stale values."
                )
        for col in self._schema.columns:
            if self._is_ndarray_column(col) and self._expression_references_name(expr, col.name):
                raise TypeError(
                    f"Column {col.name!r} is a fixed-shape ndarray column. String expressions only "
                    "support scalar columns. Use an element projection or a row-wise reduction first."
                )
            if self._is_varlen_scalar_column(col) and self._expression_references_name(expr, col.name):
                raise NotImplementedError(
                    f"Column {col.name!r} is a variable-length scalar column (vlstring/vlbytes/struct/object); "
                    "lazy expressions are not supported yet."
                )

    def _guard_varlen_scalar_expression(self, expr: str) -> None:
        self._guard_scalar_expression(expr)

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

        For tables with **nested (dotted) column names**, dotted leaf names and
        attribute-chain proxies work in both string and expression forms::

            view = t.where("trip.begin.lon > -87.7 and payment.fare > 10")
            view = t.where(t.trip.begin.lon > -87.7)

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
            operands = self._where_expression_operands()
            expr_result, operands = self._rewrite_nested_expression(expr_result, operands)
            expr_result = blosc2.lazyexpr(expr_result, operands)
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

        target_len = len(self._valid_rows)
        known_n_rows = self._known_n_rows()
        all_rows_valid = known_n_rows == target_len
        filter_intersected = False

        # For moderately-sized boolean filters, prefer a NumPy materialization.
        # LazyExpr.compute() creates a compressed NDArray and a non-compacted table
        # still needs a second pass to intersect it with _valid_rows.  Evaluating to
        # NumPy lets us do that intersection in-memory and only compress the final
        # mask once in view().  Above the threshold, keep the compressed path so peak
        # memory does not scale too aggressively with the column size.
        if isinstance(expr_result, blosc2.LazyExpr):
            filter = expr_result[:] if target_len <= _WHERE_NUMPY_MASK_LIMIT else expr_result.compute()
        else:
            filter = expr_result

        if getattr(filter, "ndim", 1) != 1:
            raise ValueError(
                "CTable.where() requires a 1-D row mask. Reduce ndarray-column predicates to one "
                "boolean per row before filtering."
            )

        filter_len = len(filter)
        if filter_len != target_len:
            if filter_len == self.nrows:
                physical = blosc2.zeros(target_len, dtype=np.bool_)
                live_pos = self._live_positions_from_valid_rows_chunks()
                physical[live_pos] = filter[:]
                filter = physical
                filter_intersected = True
            elif filter_len > target_len:
                filter = filter[:target_len]
                filter_intersected = False
            else:
                padding = blosc2.zeros(target_len, dtype=np.bool_)
                padding[:filter_len] = filter[:]
                filter = padding
                filter_intersected = False

        if not filter_intersected and not all_rows_valid:
            if isinstance(filter, np.ndarray):
                filter &= self._valid_rows[:]
            else:
                filter = (filter & self._valid_rows).compute()

        result = self.view(filter)
        return result if columns is None else result.select(list(columns))

    def _run_row_logic(self, ind: int | slice | str | Iterable) -> CTable:
        true_pos = self._live_positions_from_valid_rows_chunks()

        if isinstance(ind, Iterable) and not isinstance(ind, (str, bytes)):
            ind = list(ind)

        mant_pos = true_pos[ind]

        new_mask_np = np.zeros(len(self._valid_rows), dtype=bool)
        new_mask_np[mant_pos] = True

        new_mask = blosc2.asarray(new_mask_np)
        return self.view(new_mask)
