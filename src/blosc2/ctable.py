#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""CTable: a columnar compressed table built on top of blosc2.NDArray."""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pprint
import shutil
from collections.abc import Iterable
from dataclasses import MISSING
from textwrap import TextWrapper
from typing import Any, Generic, TypeVar

import numpy as np

from blosc2 import compute_chunks_blocks
from blosc2.ctable_storage import FileTableStorage, InMemoryTableStorage, TableStorage
from blosc2.schema_compiler import schema_from_dict, schema_to_dict

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        return wrapper


import blosc2
from blosc2.info import InfoReporter, format_nbytes_info
from blosc2.schema import SchemaSpec
from blosc2.schema_compiler import (
    ColumnConfig,
    CompiledColumn,
    CompiledSchema,
    _validate_column_name,
    compile_schema,
    compute_display_width,
)

# ---------------------------------------------------------------------------
# Index proxy and CTableIndex
# ---------------------------------------------------------------------------


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


class _CTableIndexProxy:
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


class CTableIndex:
    """A handle on an index attached to a :class:`CTable` column.

    Returned by :meth:`CTable.index` and items of :attr:`CTable.indexes`.
    Provides :meth:`drop`, :meth:`rebuild`, and :meth:`compact` convenience
    methods that delegate back to the owning table.
    """

    def __init__(self, table: CTable, col_name: str, descriptor: dict) -> None:
        self._table = table
        self._col_name = col_name
        self._descriptor = descriptor

    @property
    def col_name(self) -> str:
        """Column name this index targets."""
        return self._col_name

    @property
    def kind(self) -> str:
        """Index kind string (``'bucket'``, ``'partial'``, or ``'full'``)."""
        return self._descriptor.get("kind", "")

    @property
    def stale(self) -> bool:
        """True if the index is stale and needs rebuilding."""
        return bool(self._descriptor.get("stale", False))

    @property
    def name(self) -> str | None:
        """Optional human-readable name assigned at creation time."""
        return self._descriptor.get("name") or None

    @property
    def nbytes(self) -> int:
        """Total uncompressed size in bytes for this index payload."""
        from blosc2.indexing import _component_nbytes, iter_index_components

        root = self._table._root_table
        col_arr = root._cols[self._col_name]
        descriptor = self._descriptor
        return sum(
            _component_nbytes(col_arr, descriptor, component)
            for component in iter_index_components(col_arr, descriptor)
        )

    @property
    def cbytes(self) -> int:
        """Total compressed size in bytes for this index payload."""
        from blosc2.indexing import _component_cbytes, iter_index_components

        root = self._table._root_table
        col_arr = root._cols[self._col_name]
        descriptor = self._descriptor
        return sum(
            _component_cbytes(col_arr, descriptor, component)
            for component in iter_index_components(col_arr, descriptor)
        )

    @property
    def cratio(self) -> float:
        """Compression ratio for this index payload."""
        cbytes = self.cbytes
        if cbytes == 0:
            return float("inf")
        return self.nbytes / cbytes

    def storage_stats(self) -> tuple[int, int, float] | None:
        """Return ``(nbytes, cbytes, cratio)`` when sidecars are directly measurable."""
        try:
            nbytes = self.nbytes
            cbytes = self.cbytes
        except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError):
            root = self._table._root_table
            if not isinstance(root._storage, FileTableStorage):
                return None

            from blosc2.indexing import iter_index_components

            descriptor = self._descriptor
            col_arr = root._cols[self._col_name]
            store = root._storage._open_store()
            nbytes = 0
            cbytes = 0
            try:
                for component in iter_index_components(col_arr, descriptor):
                    if component.path is None:
                        return None
                    key = self._component_store_key(component.path)
                    obj = store[key]
                    nbytes += int(obj.nbytes)
                    cbytes += int(obj.cbytes)
            except (FileNotFoundError, OSError, RuntimeError, KeyError, ValueError):
                return None
        cratio = float("inf") if cbytes == 0 else nbytes / cbytes
        return nbytes, cbytes, cratio

    @staticmethod
    def _component_store_key(path: str) -> str:
        """Return the logical TreeStore key for an index component path."""
        normalized = path.replace("\\", "/")
        marker = "_indexes/"
        idx = normalized.find(marker)
        if idx < 0:
            raise KeyError(f"Cannot resolve index component path {path!r} inside table store.")
        relpath = normalized[idx:]
        for suffix in (".b2nd", ".b2f"):
            if relpath.endswith(suffix):
                relpath = relpath[: -len(suffix)]
                break
        return "/" + relpath.lstrip("/")

    def drop(self) -> None:
        """Drop this index from the owning table."""
        self._table.drop_index(self._col_name)

    def rebuild(self) -> CTableIndex:
        """Rebuild this index and return the updated handle."""
        return self._table.rebuild_index(self._col_name)

    def compact(self) -> CTableIndex:
        """Compact this index (merge incremental runs) and return the updated handle."""
        return self._table.compact_index(self._col_name)

    def __repr__(self) -> str:
        stale_str = " (stale)" if self.stale else ""
        name_str = f" name={self.name!r}" if self.name else ""
        return f"<CTableIndex col={self._col_name!r} kind={self.kind!r}{name_str}{stale_str}>"


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


class _RowIndexer:
    def __init__(self, table):
        self._table = table

    def __getitem__(self, item):
        return self._table._run_row_logic(item)


class _Row:
    def __init__(self, table: CTable, nrow: int):
        self._table = table
        self._nrow = nrow
        self._real_pos = None

    def _get_real_pos(self) -> int:
        self._real_pos = _find_physical_index(self._table._valid_rows, self._nrow)
        return self._real_pos

    def __getitem__(self, col_name: str):
        if self._real_pos is None:
            self._get_real_pos()
        return self._table._cols[col_name][self._real_pos]


# ---------------------------------------------------------------------------
# Column
# ---------------------------------------------------------------------------


class Column:
    def __init__(self, table: CTable, col_name: str, mask=None):
        self._table = table
        self._col_name = col_name
        self._mask = mask

    @property
    def _raw_col(self):
        return self._table._cols[self._col_name]

    @property
    def _valid_rows(self):
        if self._mask is None:
            return self._table._valid_rows

        return (self._table._valid_rows & self._mask).compute()

    def __getitem__(self, key: int | slice | list | np.ndarray):
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            return self._raw_col[int(pos_true)]

        elif isinstance(key, slice):
            real_pos = blosc2.where(self._valid_rows, _arange(len(self._valid_rows))).compute()
            start, stop, step = key.indices(len(real_pos))
            mask = blosc2.zeros(len(self._table._valid_rows), dtype=np.bool_)
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
            # Boolean mask in logical space — same convention as numpy/pandas.
            # key[i] == True means "include logical row i".
            n_live = len(self)
            if len(key) != n_live:
                raise IndexError(
                    f"Boolean mask length {len(key)} does not match number of live rows {n_live}."
                )
            all_pos = np.where(self._valid_rows[:])[0]
            phys_indices = all_pos[key]
            return self._raw_col[phys_indices]

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, _arange(len(self._valid_rows))).compute()
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            return self._raw_col[phys_indices]

        raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key: int | slice | list | np.ndarray, value):
        if self._table._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            self._raw_col[int(pos_true)] = value

        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # Boolean mask in logical space.
            n_live = len(self)
            if len(key) != n_live:
                raise IndexError(
                    f"Boolean mask length {len(key)} does not match number of live rows {n_live}."
                )
            all_pos = np.where(self._valid_rows[:])[0]
            phys_indices = all_pos[key]
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

            if isinstance(value, (list, tuple)):
                value = np.array(value, dtype=self._raw_col.dtype)
            self._raw_col[phys_indices] = value

        else:
            raise TypeError(f"Invalid index type: {type(key)}")
        self._table._root_table._mark_all_indexes_stale()

    def __iter__(self):
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

    def __len__(self):
        return blosc2.count_nonzero(self._valid_rows)

    def __lt__(self, other):
        return self._raw_col < other

    def __le__(self, other):
        return self._raw_col <= other

    def __eq__(self, other):
        return self._raw_col == other

    def __ne__(self, other):
        return self._raw_col != other

    def __gt__(self, other):
        return self._raw_col > other

    def __ge__(self, other):
        return self._raw_col >= other

    @property
    def dtype(self):
        return self._raw_col.dtype

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

    def to_numpy(self) -> np.ndarray:
        """Return all live values as a NumPy array."""
        parts = list(self.iter_chunks(size=max(1, len(self))))
        if not parts:
            return np.array([], dtype=self.dtype)
        return np.concatenate(parts) if len(parts) > 1 else parts[0]

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

    def unique(self) -> np.ndarray:
        """Return sorted array of unique live values.

        Processes data in chunks — never loads the full column at once.
        """
        seen: set = set()
        for chunk in self.iter_chunks():
            seen.update(chunk.tolist())
        return np.array(sorted(seen), dtype=self.dtype)

    def value_counts(self) -> dict:
        """Return a ``{value: count}`` dict sorted by count descending.

        Processes data in chunks — never loads the full column at once.

        Example
        -------
        >>> t["active"].value_counts()
        {True: 8432, False: 1568}
        """
        counts: dict = {}
        for chunk in self.iter_chunks():
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

    def sum(self):
        """Sum of all live values.

        Supported dtypes: bool, int, uint, float, complex.
        Bool values are counted as 0 / 1.
        """
        self._require_kind("biufc", "sum")
        self._require_nonempty("sum")
        # Use a wide accumulator to reduce overflow risk
        acc_dtype = (
            np.float64
            if self.dtype.kind == "f"
            else (
                np.complex128 if self.dtype.kind == "c" else np.int64 if self.dtype.kind in "biu" else None
            )
        )
        result = acc_dtype(0)
        for chunk in self.iter_chunks():
            result += chunk.sum(dtype=acc_dtype)
        # Return in the column's natural dtype when it fits, else keep wide
        if self.dtype.kind in "biu":
            return int(result)
        return result

    def min(self):
        """Minimum live value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        """
        self._require_kind("biufUS", "min")
        self._require_nonempty("min")
        result = None
        is_str = self.dtype.kind in "US"
        for chunk in self.iter_chunks():
            # numpy .min()/.max() don't support string dtypes in recent NumPy;
            # fall back to Python's built-in min/max which work on any comparable type.
            chunk_min = min(chunk) if is_str else chunk.min()
            if result is None or chunk_min < result:
                result = chunk_min
        return result

    def max(self):
        """Maximum live value.

        Supported dtypes: bool, int, uint, float, string, bytes.
        Strings are compared lexicographically.
        """
        self._require_kind("biufUS", "max")
        self._require_nonempty("max")
        result = None
        is_str = self.dtype.kind in "US"
        for chunk in self.iter_chunks():
            chunk_max = max(chunk) if is_str else chunk.max()
            if result is None or chunk_max > result:
                result = chunk_max
        return result

    def mean(self) -> float:
        """Arithmetic mean of all live values.

        Supported dtypes: bool, int, uint, float.
        Always returns a Python float.
        """
        self._require_kind("biuf", "mean")
        self._require_nonempty("mean")
        total = np.float64(0)
        count = 0
        for chunk in self.iter_chunks():
            total += chunk.sum(dtype=np.float64)
            count += len(chunk)
        return float(total / count)

    def std(self, ddof: int = 0) -> float:
        """Standard deviation of all live values (single-pass, Welford's algorithm).

        Parameters
        ----------
        ddof:
            Delta degrees of freedom.  ``0`` (default) gives the population
            std; ``1`` gives the sample std (divides by N-1).

        Supported dtypes: bool, int, uint, float.
        Always returns a Python float.
        """
        self._require_kind("biuf", "std")
        self._require_nonempty("std")

        # Chan's parallel update — combines per-chunk (n, mean, M2) tuples.
        # This is numerically stable and requires only a single pass.
        n_total = np.int64(0)
        mean_total = np.float64(0)
        M2_total = np.float64(0)

        for chunk in self.iter_chunks():
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
        """Return True if at least one live value is True.

        Supported dtypes: bool.
        Short-circuits on the first True found.
        """
        self._require_kind("b", "any")
        return any(chunk.any() for chunk in self.iter_chunks())

    def all(self) -> bool:
        """Return True if every live value is True.

        Supported dtypes: bool.
        Short-circuits on the first False found.
        """
        self._require_kind("b", "all")
        return all(chunk.all() for chunk in self.iter_chunks())


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


class CTable(Generic[RowT]):
    def __init__(
        self,
        row_type: type[RowT],
        new_data=None,
        *,
        urlpath: str | None = None,
        mode: str = "a",
        expected_size: int = 1_048_576,
        compact: bool = False,
        validate: bool = True,
        cparams: dict[str, Any] | None = None,
        dparams: dict[str, Any] | None = None,
    ) -> None:
        self._row_type = row_type
        self._validate = validate
        self._table_cparams = cparams
        self._table_dparams = dparams
        self._cols: dict[str, blosc2.NDArray] = {}
        self._col_widths: dict[str, int] = {}
        self.col_names: list[str] = []
        self.row = _RowIndexer(self)
        self.auto_compact = compact
        self.base = None

        # Choose storage backend
        if urlpath is not None:
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
                col = storage.open_column(name)
                self._cols[name] = col
                cc = self._schema.columns_by_name[name]
                self._col_widths[name] = max(len(name), cc.display_width)
            self._n_rows = int(blosc2.count_nonzero(self._valid_rows))
            self._last_pos = None  # resolve lazily on first write
        else:
            # ---- Create new table ----
            if storage.is_read_only():
                raise FileNotFoundError(f"No CTable found at {urlpath!r}")

            # Build compiled schema from either a dataclass or a legacy Pydantic model
            if dataclasses.is_dataclass(row_type) and isinstance(row_type, type):
                self._schema = compile_schema(row_type)
            else:
                self._schema = _compile_pydantic_schema(row_type)

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
        if storage is not None and hasattr(storage, "close"):
            storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        with contextlib.suppress(Exception):
            self.close()

    def _init_columns(
        self, expected_size: int, default_chunks, default_blocks, storage: TableStorage
    ) -> None:
        """Create one NDArray per column using the compiled schema."""
        for col in self._schema.columns:
            self.col_names.append(col.name)
            self._col_widths[col.name] = max(len(col.name), col.display_width)
            col_storage = self._resolve_column_storage(col, default_chunks, default_blocks)
            self._cols[col.name] = storage.create_column(
                col.name,
                dtype=col.dtype,
                shape=(expected_size,),
                chunks=col_storage["chunks"],
                blocks=col_storage["blocks"],
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
        - list / tuple  → positional, zipped with ``col_names``
        - dict          → used as-is
        - dataclass     → ``dataclasses.asdict``
        - np.void / structured scalar → field-name access
        """
        if isinstance(data, dict):
            return data
        if isinstance(data, (list, tuple)):
            return dict(zip(self.col_names, data, strict=False))
        if dataclasses.is_dataclass(data) and not isinstance(data, type):
            return dataclasses.asdict(data)
        if isinstance(data, (np.void, np.record)):
            return {name: data[name] for name in self.col_names}
        # Fallback: try positional indexing
        return {name: data[i] for i, name in enumerate(self.col_names)}

    def _coerce_row_to_storage(self, row: dict[str, Any]) -> dict[str, Any]:
        """Coerce each value in *row* to the column's storage dtype."""
        result = {}
        for col in self._schema.columns:
            val = row[col.name]
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
        """Double the physical capacity of all columns and the valid_rows mask."""
        c = len(self._valid_rows)
        for col_arr in self._cols.values():
            col_arr.resize((c * 2,))
        self._valid_rows.resize((c * 2,))

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        _HEAD_TAIL = 10  # rows shown at each end

        nrows = self._n_rows
        ncols = len(self.col_names)
        hidden = max(0, nrows - _HEAD_TAIL * 2)

        # -- physical positions for head and tail rows --
        valid_np = self._valid_rows[:]
        all_pos = np.where(valid_np)[0]

        if nrows <= _HEAD_TAIL * 2:
            head_pos = all_pos
            tail_pos = np.array([], dtype=all_pos.dtype)
            hidden = 0
        else:
            head_pos = all_pos[:_HEAD_TAIL]
            tail_pos = all_pos[-_HEAD_TAIL:]

        # -- per-column display widths --
        widths: dict[str, int] = {}
        for name in self.col_names:
            widths[name] = max(
                self._col_widths[name],
                len(str(self._cols[name].dtype)),
            )

        sep = "  ".join("─" * (w + 2) for w in widths.values())

        def fmt_row(values: dict) -> str:
            return "  ".join(f" {values[n]!s:<{widths[n]}} " for n in self.col_names)

        # -- batch-fetch values (one read per column, not one per cell) --
        def rows_to_dicts(positions) -> list[dict]:
            if len(positions) == 0:
                return []
            col_data = {n: self._cols[n][positions] for n in self.col_names}
            return [{n: col_data[n][i].item() for n in self.col_names} for i in range(len(positions))]

        lines = [
            fmt_row({n: n for n in self.col_names}),
            fmt_row({n: str(self._cols[n].dtype) for n in self.col_names}),
            sep,
        ]

        for row in rows_to_dicts(head_pos):
            lines.append(fmt_row(row))

        if hidden > 0:
            lines.append(fmt_row(dict.fromkeys(self.col_names, "...")))

        for row in rows_to_dicts(tail_pos):
            lines.append(fmt_row(row))

        lines.append(sep)
        footer = f"{nrows:,} rows × {ncols} columns"
        if hidden > 0:
            footer += f"  ({hidden:,} rows hidden)"
        lines.append(footer)

        return "\n".join(lines)

    def __repr__(self) -> str:
        cols = ", ".join(self.col_names)
        return f"CTable<{cols}>({self._n_rows:,} rows, {_fmt_bytes(self.cbytes)} compressed)"

    def __len__(self):
        return self._n_rows

    def __iter__(self):
        for i in range(self.nrows):
            yield _Row(self, i)

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
        obj.row = _RowIndexer(obj)
        obj.auto_compact = False
        obj.base = None

        obj._valid_rows = storage.open_valid_rows()
        for name in col_names:
            obj._cols[name] = storage.open_column(name)
            cc = schema.columns_by_name[name]
            obj._col_widths[name] = max(len(name), cc.display_width)

        obj._n_rows = int(blosc2.count_nonzero(obj._valid_rows))
        obj._last_pos = None  # resolve lazily on first write
        return obj

    # ------------------------------------------------------------------
    # Save / Load (in-memory ↔ disk)
    # ------------------------------------------------------------------

    def save(self, urlpath: str, *, overwrite: bool = False) -> None:
        """Copy this (in-memory) table to disk at *urlpath*.

        Only live rows are written — the on-disk table is always compacted.

        Parameters
        ----------
        urlpath:
            Destination directory path.
        overwrite:
            If ``False`` (default), raise :exc:`ValueError` when *urlpath*
            already exists.  Set to ``True`` to replace an existing table.

        Raises
        ------
        ValueError
            If *urlpath* already exists and ``overwrite=False``, or if called
            on a view.
        """
        if self.base is not None:
            raise ValueError("Cannot save a view — save the parent table instead.")
        file_storage = FileTableStorage(urlpath, "w")
        target_path = file_storage._root
        if os.path.exists(target_path):
            if not overwrite:
                raise ValueError(f"Path {target_path!r} already exists. Use overwrite=True to replace.")
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)

        # Collect live physical positions
        valid_np = self._valid_rows[:]
        live_pos = np.where(valid_np)[0]
        n_live = len(live_pos)
        capacity = max(n_live, 1)

        default_chunks, default_blocks = compute_chunks_blocks((capacity,))

        # --- valid_rows (all True, compacted) ---
        disk_valid = file_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        if n_live > 0:
            disk_valid[:n_live] = True

        # --- columns ---
        for col in self._schema.columns:
            name = col.name
            col_storage = self._resolve_column_storage(col, default_chunks, default_blocks)
            disk_col = file_storage.create_column(
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

        file_storage.save_schema(schema_to_dict(self._schema))
        file_storage.close()

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
        disk_cols = {name: file_storage.open_column(name) for name in col_names}
        phys_size = len(disk_valid)
        n_live = int(blosc2.count_nonzero(disk_valid))
        capacity = max(phys_size, 1)

        mem_storage = InMemoryTableStorage()
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))

        mem_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        if phys_size > 0:
            mem_valid[:phys_size] = disk_valid[:]

        mem_cols: dict[str, blosc2.NDArray] = {}
        for col in schema.columns:
            name = col.name
            mem_col = mem_storage.create_column(
                name,
                dtype=col.dtype,
                shape=(capacity,),
                chunks=default_chunks,
                blocks=default_blocks,
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
        obj.row = _RowIndexer(obj)
        obj.auto_compact = False
        obj.base = None
        obj._valid_rows = mem_valid
        obj._n_rows = n_live
        obj._last_pos = None  # resolve lazily on first write
        return obj

    # ------------------------------------------------------------------
    # View / filtering
    # ------------------------------------------------------------------

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
        obj._col_widths = parent._col_widths
        obj.col_names = parent.col_names
        obj.row = _RowIndexer(obj)
        obj.auto_compact = parent.auto_compact
        obj.base = parent
        obj._valid_rows = new_valid_rows
        obj._n_rows = int(blosc2.count_nonzero(new_valid_rows))
        obj._last_pos = None
        return obj

    def view(self, new_valid_rows):
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
            if name not in self._cols:
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

        # Subset of columns — same NDArray objects, no copy
        obj._cols = {name: self._cols[name] for name in cols}
        obj.col_names = list(cols)

        # Rebuild schema for the selected columns only
        sel_set = set(cols)
        sel_compiled = [c for c in self._schema.columns if c.name in sel_set]
        # Preserve caller-specified order
        order = {name: i for i, name in enumerate(cols)}
        sel_compiled.sort(key=lambda c: order[c.name])
        obj._schema = CompiledSchema(
            columns=sel_compiled,
            columns_by_name={c.name: c for c in sel_compiled},
            row_cls=self._schema.row_cls,
        )
        obj._col_widths = {name: self._col_widths[name] for name in cols if name in self._col_widths}
        obj.row = _RowIndexer(obj)
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
            lines.append(f"  {name}  [{dtype}]")

            if n == 0:
                lines.append("    (empty)")
                lines.append("")
                continue

            if dtype.kind in "biufc" and dtype.kind != "c":
                # numeric + bool
                if dtype.kind == "b":
                    arr = col.to_numpy()
                    true_n = int(arr.sum())
                    lines.append(f"    count : {n:,}")
                    lines.append(f"    true  : {true_n:,}  ({true_n / n * 100:.1f} %)")
                    lines.append(f"    false : {n - true_n:,}  ({(n - true_n) / n * 100:.1f} %)")
                else:
                    mn = col.min()
                    mx = col.max()
                    avg = col.mean()
                    sd = col.std()
                    fmt = ".4g"
                    lines.append(f"    count : {n:,}")
                    lines.append(f"    mean  : {avg:{fmt}}")
                    lines.append(f"    std   : {sd:{fmt}}")
                    lines.append(f"    min   : {mn:{fmt}}")
                    lines.append(f"    max   : {mx:{fmt}}")
            elif dtype.kind in "US":
                mn = col.min()
                mx = col.max()
                nu = len(col.unique())
                lines.append(f"    count   : {n:,}")
                lines.append(f"    unique  : {nu:,}")
                lines.append(f"    min     : {mn!r}")
                lines.append(f"    max     : {mx!r}")
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
            dtype = self._cols[name].dtype
            if not (
                np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating) or dtype == np.bool_
            ):
                raise TypeError(
                    f"Column {name!r} has dtype {dtype} which is not supported by cov(). "
                    "Only int, float, and bool columns are allowed."
                )

        if self._n_rows < 2:
            raise ValueError(f"cov() requires at least 2 live rows, got {self._n_rows}.")

        # Build (n_cols, n_rows) matrix — one row per column
        arrays = []
        for name in self.col_names:
            arr = self[name].to_numpy()
            if arr.dtype == np.bool_:
                arr = arr.astype(np.int8)
            arrays.append(arr.astype(np.float64))

        data = np.stack(arrays, axis=0)  # shape (ncols, n_live)
        return np.atleast_2d(np.cov(data))

    # ------------------------------------------------------------------
    # Arrow interop
    # ------------------------------------------------------------------

    def to_arrow(self):
        """Convert all live rows to a :class:`pyarrow.Table`.

        Each column is materialized via :meth:`Column.to_numpy` and wrapped
        in a ``pyarrow.array``.  String columns are emitted as ``pa.string()``
        (variable-length UTF-8); bytes columns as ``pa.large_binary()``.

        Raises
        ------
        ImportError
            If ``pyarrow`` is not installed.
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for to_arrow(). Install it with: pip install pyarrow"
            ) from None

        arrays = {}
        for name in self.col_names:
            col = self[name]
            arr = col.to_numpy()
            kind = arr.dtype.kind
            if kind == "U":
                pa_arr = pa.array(arr.tolist(), type=pa.string())
            elif kind == "S":
                pa_arr = pa.array(arr.tolist(), type=pa.large_binary())
            else:
                pa_arr = pa.array(arr)
            arrays[name] = pa_arr

        return pa.table(arrays)

    @classmethod
    def from_arrow(cls, arrow_table) -> CTable:
        """Build a :class:`CTable` from a :class:`pyarrow.Table`.

        Schema is inferred from the Arrow field types.  String columns
        (``pa.string()``, ``pa.large_string()``) are stored with
        ``max_length`` set to the longest value found in the data.

        Parameters
        ----------
        arrow_table:
            A ``pyarrow.Table`` instance.

        Returns
        -------
        CTable
            A new in-memory CTable containing all rows from *arrow_table*.

        Raises
        ------
        ImportError
            If ``pyarrow`` is not installed.
        TypeError
            If an Arrow field type has no corresponding blosc2 spec.
        """
        try:
            import pyarrow as pa
        except ImportError:
            raise ImportError(
                "pyarrow is required for from_arrow(). Install it with: pip install pyarrow"
            ) from None

        import blosc2.schema as b2s

        def _arrow_type_to_spec(pa_type, arrow_col):
            """Map a pyarrow DataType to a blosc2 SchemaSpec."""
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
                    return spec_cls()

            # String types: determine max_length from the data
            if pa_type in (pa.string(), pa.large_string(), pa.utf8(), pa.large_utf8()):
                values = [v for v in arrow_col.to_pylist() if v is not None]
                max_len = max((len(v) for v in values), default=1)
                return b2s.string(max_length=max(max_len, 1))

            raise TypeError(
                f"No blosc2 spec for Arrow type {pa_type!r}. "
                "Supported: int8/16/32/64, uint8/16/32/64, float32/64, bool, string."
            )

        # Build CompiledSchema from Arrow schema
        columns: list[CompiledColumn] = []
        for field in arrow_table.schema:
            name = field.name
            _validate_column_name(name)
            spec = _arrow_type_to_spec(field.type, arrow_table.column(name))
            col_config = ColumnConfig(cparams=None, dparams=None, chunks=None, blocks=None)
            columns.append(
                CompiledColumn(
                    name=name,
                    py_type=spec.python_type,
                    spec=spec,
                    dtype=spec.dtype,
                    default=MISSING,
                    config=col_config,
                    display_width=compute_display_width(spec),
                )
            )

        schema = CompiledSchema(
            row_cls=None,
            columns=columns,
            columns_by_name={col.name: col for col in columns},
        )

        n = len(arrow_table)
        capacity = max(n, 1)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        mem_storage = InMemoryTableStorage()

        new_valid = mem_storage.create_valid_rows(
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
        )
        new_cols: dict[str, blosc2.NDArray] = {}
        for col in columns:
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
        obj._row_type = None
        obj._validate = False
        obj._table_cparams = None
        obj._table_dparams = None
        obj._storage = mem_storage
        obj._read_only = False
        obj._schema = schema
        obj._cols = new_cols
        obj._col_widths = {col.name: max(len(col.name), col.display_width) for col in columns}
        obj.col_names = [col.name for col in columns]
        obj.row = _RowIndexer(obj)
        obj.auto_compact = False
        obj.base = None
        obj._valid_rows = new_valid
        obj._n_rows = 0
        obj._last_pos = 0

        if n > 0:
            # Write each column directly — one bulk slice assignment per column.
            # String columns (dtype.kind == 'U') can't go through Arrow's zero-copy
            # to_numpy(), so we convert via to_pylist() and let NumPy handle the
            # fixed-width unicode coercion.  All other types use zero-copy numpy.
            for col in columns:
                arrow_col = arrow_table.column(col.name)
                if col.dtype.kind in "US":
                    arr = np.array(arrow_col.to_pylist(), dtype=col.dtype)
                else:
                    arr = arrow_col.to_numpy(zero_copy_only=False).astype(col.dtype)
                new_cols[col.name][:n] = arr

            new_valid[:n] = True
            obj._n_rows = n
            obj._last_pos = n

        return obj

    # ------------------------------------------------------------------
    # CSV interop
    # ------------------------------------------------------------------

    def to_csv(self, path: str, *, header: bool = True, sep: str = ",") -> None:
        """Write all live rows to a CSV file.

        Uses Python's stdlib ``csv`` module — no extra dependency required.
        Each column is materialised once via :meth:`Column.to_numpy`; rows
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

        arrays = [self[name].to_numpy() for name in self.col_names]

        with open(path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=sep)
            if header:
                writer.writerow(self.col_names)
            for row in zip(*arrays, strict=True):
                writer.writerow(row)

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
        obj.row = _RowIndexer(obj)
        obj.auto_compact = False
        obj.base = None
        obj._valid_rows = new_valid
        obj._n_rows = 0
        obj._last_pos = 0

        if n > 0:
            for i, col in enumerate(schema.columns):
                if col.dtype == np.bool_:
                    # np.array(["False"], dtype=bool) treats any non-empty
                    # string as True.  Parse "True"/"False"/"1"/"0" explicitly.
                    arr = np.array(
                        [v.strip() in ("True", "true", "1") for v in col_data[i]],
                        dtype=np.bool_,
                    )
                else:
                    arr = np.array(col_data[i], dtype=col.dtype)
                new_cols[col.name][:n] = arr
            new_valid[:n] = True
            obj._n_rows = n
            obj._last_pos = n

        return obj

    # ------------------------------------------------------------------
    # Schema mutations: add / drop / rename columns
    # ------------------------------------------------------------------

    def add_column(
        self,
        name: str,
        spec: SchemaSpec,
        default,
        *,
        cparams: dict | None = None,
    ) -> None:
        """Add a new column filled with *default* for every existing live row.

        Parameters
        ----------
        name:
            Column name.  Must follow the same naming rules as schema fields.
        spec:
            A schema descriptor such as ``b2.int64(ge=0)`` or ``b2.string()``.
        default:
            Value written to every existing live row.  Must be coercible to
            *spec*'s dtype.
        cparams:
            Optional compression parameters for this column's NDArray.

        Raises
        ------
        ValueError
            If the table is read-only, is a view, or the column already exists.
        TypeError
            If *default* cannot be coerced to *spec*'s dtype.
        """
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot add a column to a view.")
        _validate_column_name(name)
        if name in self._cols:
            raise ValueError(f"Column {name!r} already exists.")

        try:
            default_val = spec.dtype.type(default)
        except (ValueError, OverflowError) as exc:
            raise TypeError(f"Cannot coerce default {default!r} to dtype {spec.dtype!r}: {exc}") from exc

        capacity = len(self._valid_rows)
        default_chunks, default_blocks = compute_chunks_blocks((capacity,))
        new_col = self._storage.create_column(
            name,
            dtype=spec.dtype,
            shape=(capacity,),
            chunks=default_chunks,
            blocks=default_blocks,
            cparams=cparams,
            dparams=None,
        )

        live_pos = np.where(self._valid_rows[:])[0]
        if len(live_pos) > 0:
            new_col[live_pos] = default_val

        compiled_col = CompiledColumn(
            name=name,
            py_type=spec.python_type,
            spec=spec,
            dtype=spec.dtype,
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
            self._storage.save_schema(schema_to_dict(self._schema))

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
        if len(self.col_names) == 1:
            raise ValueError("Cannot drop the last column.")

        catalog = self._storage.load_index_catalog()
        if name in catalog:
            descriptor = catalog.pop(name)
            self._validate_index_descriptor(name, descriptor)
            self._drop_index_descriptor(name, descriptor)
            self._storage.save_index_catalog(catalog)

        if isinstance(self._storage, FileTableStorage):
            self._storage.delete_column(name)

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
            self._storage.save_schema(schema_to_dict(self._schema))

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
        if old not in self._cols:
            raise KeyError(f"No column named {old!r}. Available: {self.col_names}")
        if new in self._cols:
            raise ValueError(f"Column {new!r} already exists.")
        _validate_column_name(new)

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
        if isinstance(self._storage, FileTableStorage):
            self._storage.save_schema(schema_to_dict(self._schema))
        if rebuild_kwargs is not None:
            self.create_index(new, **rebuild_kwargs)

    # ------------------------------------------------------------------
    # Column access
    # ------------------------------------------------------------------

    def __getitem__(self, s: str):
        if s in self._cols:
            return Column(self, s)
        return None

    def __getattr__(self, s: str):
        if s in self._cols:
            return Column(self, s)
        return super().__getattribute__(s)

    # ------------------------------------------------------------------
    # Compaction
    # ------------------------------------------------------------------

    def compact(self):
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise ValueError("Cannot compact a view.")
        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        start = 0
        block_size = self._valid_rows.blocks[0]
        end = min(block_size, self._n_rows)
        while start < end:
            for _k, v in self._cols.items():
                v[start:end] = v[real_poss[start:end]]
            start += block_size
            end = min(end + block_size, self._n_rows)

        self._valid_rows[: self._n_rows] = True
        self._valid_rows[self._n_rows :] = False
        self._last_pos = self._n_rows  # next write goes right after live rows
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
            if name not in self._cols:
                raise KeyError(f"No column named {name!r}. Available: {self.col_names}")
            dtype = self._cols[name].dtype
            if np.issubdtype(dtype, np.complexfloating):
                raise TypeError(
                    f"Column {name!r} has complex dtype {dtype} which does not support ordering."
                )
        return cols, ascending

    def _build_lex_keys(
        self,
        cols: list[str],
        ascending: list[bool],
        live_pos: np.ndarray,
        n: int,
    ) -> list[np.ndarray]:
        """Build the key list for np.lexsort (innermost = last = primary key)."""
        lex_keys = []
        for name, asc in zip(reversed(cols), reversed(ascending), strict=True):
            raw = self._cols[name][live_pos]
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
        if self.base is not None:
            raise ValueError("Cannot sort a view. Materialise it first with .to_table() or sort the parent.")
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

        order = np.lexsort(self._build_lex_keys(cols, ascending, live_pos, n))

        sorted_pos = live_pos[order]

        if inplace:
            for _col_name, arr in self._cols.items():
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
            for col_name, arr in self._cols.items():
                result._cols[col_name][:n] = arr[sorted_pos]
            result._valid_rows[:n] = True
            result._valid_rows[n:] = False
            result._n_rows = n
            result._last_pos = n
            return result

    def _empty_copy(self) -> CTable:
        """Return a new empty in-memory CTable with the same schema and capacity."""
        from blosc2 import compute_chunks_blocks

        capacity = max(self._n_rows, 1)
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
        obj._col_widths = self._col_widths
        obj.col_names = [col.name for col in self._schema.columns]
        obj.row = _RowIndexer(obj)
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
        return len(self._cols)

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

        col_arr = self._cols.get(col_name)
        token = descriptor["token"]

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

        anchor = self._storage.index_anchor_path(col_name)
        if anchor is not None:
            proxy_key = ("persistent", str(Path(anchor).resolve()))
            _PERSISTENT_INDEXES.pop(proxy_key, None)
            with contextlib.suppress(OSError):
                os.rmdir(os.path.dirname(anchor))

    def _index_create_kwargs_from_descriptor(self, descriptor: dict) -> dict[str, Any]:
        """Return create_index kwargs that rebuild an existing descriptor."""
        build = "ooc" if bool(descriptor.get("ooc", False)) else "memory"
        return {
            "kind": descriptor["kind"],
            "optlevel": int(descriptor.get("optlevel", 5)),
            "name": descriptor.get("name") or None,
            "build": build,
            "cparams": descriptor.get("cparams"),
        }

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
        proxy = _CTableIndexProxy(col_arr, anchor)
        proxy_key = _array_key(proxy)
        _PERSISTENT_INDEXES.pop(proxy_key, None)  # clear any stale cache entry

        target = _field_target_descriptor(None)
        token = _target_token(target)
        persistent = True
        dtype = col_arr.dtype
        use_ooc = _resolve_ooc_mode(kind, build)

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
            if kind == "full":
                with tempfile.TemporaryDirectory(prefix="blosc2-index-ooc-", dir=resolved_tmpdir) as td:
                    full = _build_full_descriptor_ooc(
                        proxy, target, token, kind, dtype, persistent, Path(td), cparams_obj
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
            full = (
                _build_full_descriptor(proxy, token, kind, values, persistent, cparams_obj)
                if kind == "full"
                else None
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
            )

        result = _copy_descriptor(descriptor)
        _PERSISTENT_INDEXES.pop(proxy_key, None)  # evict proxy to avoid memory leak
        return result

    def create_index(
        self,
        col_name: str,
        *,
        kind: blosc2.IndexKind = blosc2.IndexKind.BUCKET,
        optlevel: int = 5,
        name: str | None = None,
        build: str = "auto",
        tmpdir: str | None = None,
        **kwargs,
    ) -> CTableIndex:
        """Build and register an index for a column.

        Parameters
        ----------
        col_name:
            Name of the column to index.
        kind:
            Index kind.  One of :attr:`blosc2.IndexKind.BUCKET` (default),
            :attr:`blosc2.IndexKind.PARTIAL`, or :attr:`blosc2.IndexKind.FULL`.
        optlevel:
            Optimisation level (1–9).  Higher values give more precise pruning
            at the cost of larger index files.  Default is 5.
        name:
            Optional human-readable label for the index.
        build:
            Build strategy: ``'auto'``, ``'memory'``, or ``'ooc'`` (out-of-core).
        tmpdir:
            Temporary directory for out-of-core builds.  ``None`` means use the
            column's own directory (persistent tables) or the system temporary
            directory (in-memory tables).
        **kwargs:
            Pass ``cparams=<CParams or dict>`` to customise index compression.

        Returns
        -------
        CTableIndex
            A handle on the newly created index.

        Raises
        ------
        ValueError
            If called on a view.
        KeyError
            If *col_name* is not a column of this table.
        """
        if self.base is not None:
            raise ValueError("Cannot create an index on a view.")
        if col_name not in self._cols:
            raise KeyError(f"No column named {col_name!r}. Available: {self.col_names}")
        catalog = self._storage.load_index_catalog()
        if col_name in catalog:
            raise ValueError(
                f"Index already exists for column {col_name!r}. "
                "Call rebuild_index() to replace it or drop_index() first."
            )

        from blosc2.indexing import (
            _IN_MEMORY_INDEXES,
            _copy_descriptor,
            _normalize_build_mode,
            _normalize_index_cparams,
            _normalize_index_kind,
        )
        from blosc2.indexing import (
            create_index as _ix_create_index,
        )

        cparams_obj = _normalize_index_cparams(kwargs.pop("cparams", None))
        if kwargs:
            raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(kwargs))}")

        kind_str = _normalize_index_kind(kind)
        build_str = _normalize_build_mode(build)
        col_arr = self._cols[col_name]
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
            )
            store = _IN_MEMORY_INDEXES[id(col_arr)]
            descriptor = _copy_descriptor(store["indexes"]["__self__"])

        value_epoch, _ = self._storage.get_epoch_counters()
        descriptor["built_value_epoch"] = value_epoch

        catalog = self._storage.load_index_catalog()
        catalog[col_name] = descriptor
        self._storage.save_index_catalog(catalog)
        return CTableIndex(self, col_name, descriptor)

    def drop_index(self, col_name: str) -> None:
        """Remove the index for *col_name* and delete any sidecar files.

        Parameters
        ----------
        col_name:
            Column whose index should be dropped.

        Raises
        ------
        ValueError
            If called on a view.
        KeyError
            If no index exists for *col_name*.
        """
        if self.base is not None:
            raise ValueError("Cannot drop an index from a view.")

        catalog = self._storage.load_index_catalog()
        if col_name not in catalog:
            raise KeyError(f"No index found for column {col_name!r}.")

        descriptor = catalog.pop(col_name)
        self._validate_index_descriptor(col_name, descriptor)
        self._drop_index_descriptor(col_name, descriptor)
        self._storage.save_index_catalog(catalog)

    def rebuild_index(self, col_name: str) -> CTableIndex:
        """Drop and recreate the index for *col_name* with the same parameters.

        Parameters
        ----------
        col_name:
            Column whose index should be rebuilt.

        Returns
        -------
        CTableIndex
            A handle on the newly built index.

        Raises
        ------
        ValueError
            If called on a view.
        KeyError
            If no index exists for *col_name*.
        """
        if self.base is not None:
            raise ValueError("Cannot rebuild an index on a view.")

        catalog = self._storage.load_index_catalog()
        if col_name not in catalog:
            raise KeyError(f"No index found for column {col_name!r}.")

        old_desc = catalog[col_name]
        self._validate_index_descriptor(col_name, old_desc)
        create_kwargs = self._index_create_kwargs_from_descriptor(old_desc)

        self.drop_index(col_name)
        return self.create_index(col_name, **create_kwargs)

    def compact_index(self, col_name: str) -> CTableIndex:
        """Compact the index for *col_name*, merging any incremental append runs.

        Only meaningful for ``kind='full'`` indexes.  For other kinds the call
        is a no-op and returns the current handle.

        Parameters
        ----------
        col_name:
            Column whose index should be compacted.

        Returns
        -------
        CTableIndex
            A handle reflecting the (possibly updated) index descriptor.

        Raises
        ------
        ValueError
            If called on a view.
        KeyError
            If no index exists for *col_name*.
        """
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
        from blosc2.indexing import (
            compact_index as _ix_compact_index,
        )

        catalog = self._storage.load_index_catalog()
        if col_name not in catalog:
            raise KeyError(f"No index found for column {col_name!r}.")

        col_arr = self._cols[col_name]
        descriptor = catalog[col_name]

        if _is_persistent_array(col_arr):
            anchor = self._storage.index_anchor_path(col_name)
            proxy = _CTableIndexProxy(col_arr, anchor)
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
            catalog[col_name] = updated_desc
            self._storage.save_index_catalog(catalog)
            return CTableIndex(self, col_name, updated_desc)
        else:
            _ix_compact_index(col_arr)
            store = _IN_MEMORY_INDEXES.get(id(col_arr))
            if store:
                token = descriptor["token"]
                updated_desc = _copy_descriptor(store["indexes"].get(token, descriptor))
                updated_desc["built_value_epoch"] = descriptor.get("built_value_epoch", 0)
                catalog[col_name] = updated_desc
                self._storage.save_index_catalog(catalog)
                return CTableIndex(self, col_name, updated_desc)
            return CTableIndex(self, col_name, descriptor)

    def index(self, col_name: str) -> CTableIndex:
        """Return the index handle for *col_name*.

        Parameters
        ----------
        col_name:
            Column name to look up.

        Returns
        -------
        CTableIndex

        Raises
        ------
        KeyError
            If no index exists for *col_name*.
        """
        catalog = self._root_table._storage.load_index_catalog()
        if col_name not in catalog:
            raise KeyError(f"No index found for column {col_name!r}.")
        return CTableIndex(self, col_name, catalog[col_name])

    @property
    def indexes(self) -> list[CTableIndex]:
        """Return a list of :class:`CTableIndex` handles for all active indexes."""
        catalog = self._root_table._storage.load_index_catalog()
        return [CTableIndex(self, col_name, desc) for col_name, desc in catalog.items()]

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

    def _try_index_where(self, expr_result: blosc2.LazyExpr) -> np.ndarray | None:
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

        expression = expr_result.expression
        operands = dict(expr_result.operands)

        indexed_columns = self._find_indexed_columns(root._cols, catalog, operands)
        if not indexed_columns:
            return None

        primary_col_name, primary_col_arr, _ = indexed_columns[0]

        # Inject every usable table-owned descriptor so plan_query can combine them.
        for _col_name, col_arr, descriptor in indexed_columns:
            arr_key = _array_key(col_arr)
            if _is_persistent_array(col_arr):
                store = _PERSISTENT_INDEXES.get(arr_key) or _default_index_store()
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

        if plan.exact_positions is not None:
            return np.asarray(plan.exact_positions, dtype=np.int64)

        if plan.bucket_masks is not None:
            _, positions = evaluate_bucket_query(
                expression, merged_operands, {}, where_dict, plan, return_positions=True
            )
            return np.asarray(positions, dtype=np.int64)

        if plan.candidate_units is not None and plan.segment_len is not None:
            _, positions = evaluate_segment_query(
                expression, merged_operands, {}, where_dict, plan, return_positions=True
            )
            return np.asarray(positions, dtype=np.int64)

        return None

    @property
    def info_items(self) -> list[tuple[str, object]]:
        """Structured summary items used by :meth:`info`."""
        storage_type = "persistent" if isinstance(self._storage, FileTableStorage) else "in-memory"
        urlpath = self._storage._root if isinstance(self._storage, FileTableStorage) else None
        schema_summary = {
            name: _InfoLiteral(self._dtype_info_label(self._cols[name].dtype)) for name in self.col_names
        }

        index_summary = {}
        for idx in self.indexes:
            stale = " stale" if idx.stale else ""
            label = f" name={idx.name!r}" if idx.name and idx.name != "__self__" else ""
            stats = idx.storage_stats()
            if stats is None:
                suffix = "size=n/a (sidecars not directly addressable)"
            else:
                nbytes, cbytes, cratio = stats
                suffix = (
                    f"nbytes={format_nbytes_info(nbytes)}, "
                    f"cbytes={format_nbytes_info(cbytes)}, cratio={cratio:.2f}x"
                )
            index_summary[idx.col_name] = f"[{idx.kind}{stale}{label}] {suffix}"

        items = [
            ("type", self.__class__.__name__),
            ("storage", storage_type),
            ("rows", self.nrows),
            ("columns", self.ncols),
            ("capacity", len(self._valid_rows)),
            ("view", self.base is not None),
            ("read_only", self._read_only),
            ("nbytes", format_nbytes_info(self.nbytes)),
            ("cbytes", format_nbytes_info(self.cbytes)),
            ("cratio", f"{self.cratio:.2f}"),
            ("schema", schema_summary),
            (
                "valid_rows_mask",
                f"nbytes={format_nbytes_info(self._valid_rows.nbytes)}, cbytes={format_nbytes_info(self._valid_rows.cbytes)}",
            ),
            ("indexes", index_summary if index_summary else "none"),
        ]
        if urlpath is not None:
            items.insert(2, ("urlpath", urlpath))
        return items

    @staticmethod
    def _dtype_info_label(dtype: np.dtype) -> str:
        """Return a compact dtype label for info reports."""
        if dtype.kind == "U":
            return f"U{dtype.itemsize // 4}"
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
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise TypeError("Cannot extend view.")

        # Normalize → validate → coerce
        row = self._normalize_row_input(data)
        if self._validate:
            from blosc2.schema_validation import validate_row

            row = validate_row(self._schema, row)
        row = self._coerce_row_to_storage(row)

        pos = self._resolve_last_pos()
        if pos >= len(self._valid_rows):
            self._grow()

        for name, col_array in self._cols.items():
            col_array[pos] = row[name]

        self._valid_rows[pos] = True
        self._last_pos = pos + 1
        self._n_rows += 1
        self._mark_all_indexes_stale()

    def delete(self, ind: int | slice | str | Iterable) -> None:
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

    def extend(self, data: list | CTable | Any, *, validate: bool | None = None) -> None:
        if self._read_only:
            raise ValueError("Table is read-only (opened with mode='r').")
        if self.base is not None:
            raise TypeError("Cannot extend view.")
        if len(data) <= 0:
            return

        # Resolve effective validate flag: per-call override takes precedence
        do_validate = self._validate if validate is None else validate

        start_pos = self._resolve_last_pos()

        current_col_names = self.col_names
        columns_to_insert = []
        new_nrows = 0

        if hasattr(data, "_cols") and hasattr(data, "_n_rows"):
            for name in current_col_names:
                col = data._cols[name][: data._n_rows]
                columns_to_insert.append(col)
            new_nrows = data._n_rows
        else:
            if isinstance(data, np.ndarray) and data.dtype.names is not None:
                for name in current_col_names:
                    columns_to_insert.append(data[name])
                new_nrows = len(data)
            else:
                columns_to_insert = list(zip(*data, strict=False))
                new_nrows = len(data)

        # Validate constraints column-by-column before writing
        if do_validate:
            from blosc2.schema_vectorized import validate_column_batch

            raw_columns = {current_col_names[i]: columns_to_insert[i] for i in range(len(current_col_names))}
            validate_column_batch(self._schema, raw_columns)

        processed_cols = []
        for i, raw_col in enumerate(columns_to_insert):
            target_dtype = self._cols[current_col_names[i]].dtype
            b2_arr = blosc2.asarray(raw_col, dtype=target_dtype)
            processed_cols.append(b2_arr)

        end_pos = start_pos + new_nrows

        if self.auto_compact and end_pos >= len(self._valid_rows):
            self.compact()  # sets _last_pos = _n_rows
            start_pos = self._last_pos
            end_pos = start_pos + new_nrows

        while end_pos > len(self._valid_rows):
            self._grow()

        for j, name in enumerate(current_col_names):
            self._cols[name][start_pos:end_pos] = processed_cols[j][:]

        self._valid_rows[start_pos:end_pos] = True
        self._last_pos = end_pos
        self._n_rows += new_nrows
        self._mark_all_indexes_stale()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @profile
    def where(self, expr_result) -> CTable:
        if isinstance(expr_result, Column):
            expr_result = expr_result._raw_col

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
                return self.view(blosc2.asarray(mask))

        filter = expr_result.compute() if isinstance(expr_result, blosc2.LazyExpr) else expr_result

        target_len = len(self._valid_rows)

        if len(filter) > target_len:
            filter = filter[:target_len]
        elif len(filter) < target_len:
            padding = blosc2.zeros(target_len, dtype=np.bool_)
            padding[: len(filter)] = filter[:]
            filter = padding

        filter = (filter & self._valid_rows).compute()

        return self.view(filter)

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
