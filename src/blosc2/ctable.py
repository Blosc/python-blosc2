#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""CTable: a columnar compressed table built on top of blosc2.NDArray."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import MISSING
from typing import Any, Generic, TypeVar

import numpy as np

from blosc2 import compute_chunks_blocks

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        return wrapper


import blosc2
from blosc2.schema import SchemaSpec
from blosc2.schema_compiler import (
    ColumnConfig,
    CompiledColumn,
    CompiledSchema,
    compile_schema,
)

# RowT is intentionally left unbound so CTable works with both dataclasses
# and legacy Pydantic models during the transition period.
RowT = TypeVar("RowT")


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
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
            start, stop, step = key.indices(len(real_pos))
            mask = blosc2.zeros(len(self._table._valid_rows), dtype=np.bool_)
            if step == 1:
                phys_start = real_pos[start]
                phys_stop = real_pos[stop - 1]
                mask[phys_start : phys_stop + 1] = True
            else:
                lindices = np.arange(start, stop, step)
                phys_indices = real_pos[lindices]
                mask[phys_indices[:]] = True
            return Column(self._table, self._col_name, mask=mask)

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            return self._raw_col[phys_indices]

        raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key: int | slice | list | np.ndarray, value):
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")
            pos_true = _find_physical_index(self._valid_rows, key)
            self._raw_col[int(pos_true)] = value

        elif isinstance(key, (slice, list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
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

    def to_numpy(self):
        real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
        return self._raw_col[real_pos[:]]


# ---------------------------------------------------------------------------
# CTable
# ---------------------------------------------------------------------------


class CTable(Generic[RowT]):
    def __init__(
        self,
        row_type: type[RowT],
        new_data=None,
        *,
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

        # Build compiled schema from either a dataclass or a legacy Pydantic model
        if dataclasses.is_dataclass(row_type) and isinstance(row_type, type):
            self._schema: CompiledSchema = compile_schema(row_type)
        else:
            self._schema = _compile_pydantic_schema(row_type)

        self._cols: dict[str, blosc2.NDArray] = {}
        self._n_rows: int = 0
        self._last_pos: int | None = 0  # physical index of next write slot;
        # None means it must be recalculated
        # (set after any deletion)
        self._col_widths: dict[str, int] = {}
        self.col_names: list[str] = []
        self.row = _RowIndexer(self)
        self.auto_compact = compact
        self.base = None

        default_chunks, default_blocks = compute_chunks_blocks((expected_size,))
        self._valid_rows = blosc2.zeros(
            shape=(expected_size,),
            dtype=np.bool_,
            chunks=default_chunks,
            blocks=default_blocks,
        )

        self._init_columns(expected_size, default_chunks, default_blocks)

        if new_data is not None:
            self._load_initial_data(new_data)

    def _init_columns(self, expected_size: int, default_chunks, default_blocks) -> None:
        """Create one NDArray per column using the compiled schema."""
        for col in self._schema.columns:
            self.col_names.append(col.name)
            self._col_widths[col.name] = max(len(col.name), col.display_width)
            storage = self._resolve_column_storage(col, default_chunks, default_blocks)
            self._cols[col.name] = blosc2.zeros(
                shape=(expected_size,),
                dtype=col.dtype,
                **storage,
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

    def __str__(self):
        retval = []
        cont = 0

        for name in self._cols:
            retval.append(f"{name:^{self._col_widths[name]}} |")
            cont += self._col_widths[name] + 2
        retval.append("\n")
        for _i in range(cont):
            retval.append("-")
        retval.append("\n")

        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()

        for j in real_poss:
            for name in self._cols:
                retval.append(f"{self._cols[name][j]:^{self._col_widths[name]}}")
                retval.append(" |")
            retval.append("\n")
            for _ in range(cont):
                retval.append("-")
            retval.append("\n")
        return "".join(retval)

    def __len__(self):
        return self._n_rows

    def __iter__(self):
        for i in range(self.nrows):
            yield _Row(self, i)

    # ------------------------------------------------------------------
    # View / filtering
    # ------------------------------------------------------------------

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

        retval = CTable(
            self._row_type,
            compact=self.auto_compact,
            expected_size=len(self._valid_rows),
        )
        retval._cols = self._cols
        retval._n_rows = blosc2.count_nonzero(new_valid_rows)
        retval._col_widths = self._col_widths
        retval.col_names = self.col_names
        retval.base = self
        retval._valid_rows = new_valid_rows

        return retval

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
        """Return a JSON-compatible dict describing this table's schema.

        Suitable for debugging, serialization, and future ``save()``/``load()``
        support.  See :func:`~blosc2.schema_compiler.schema_to_dict`.
        """
        from blosc2.schema_compiler import schema_to_dict

        return schema_to_dict(self._schema)

    def info(self) -> None:
        """Print a concise summary of the CTable."""
        n_cols = len(self._cols)
        n_rows = len(self)

        cbytes = sum(col.cbytes for col in self._cols.values()) + self._valid_rows.cbytes
        nbytes = sum(col.nbytes for col in self._cols.values()) + self._valid_rows.nbytes

        def format_bytes(bytes_size: float) -> str:
            if bytes_size < 1024:
                return f"{bytes_size} B"
            elif bytes_size < 1024**2:
                return f"{bytes_size / 1024:.2f} KB"
            elif bytes_size < 1024**3:
                return f"{bytes_size / (1024**2):.2f} MB"
            else:
                return f"{bytes_size / (1024**3):.2f} GB"

        ratio = (nbytes / cbytes) if cbytes > 0 else 0.0

        lines = []
        lines.append("<class 'CTable'>")
        lines.append(f"nºColumns: {n_cols}")
        lines.append(f"nºRows: {n_rows}")
        lines.append("")

        header = f" {'#':>3}   {'Column':<15} {'Itemsize':<12} {'Dtype':<15}"
        lines.append(header)
        lines.append(f" {'---':>3}  {'------':<15} {'--------':<12} {'-----':<15}")

        for i, name in enumerate(self.col_names):
            col_array = self._cols[name]
            dtype_str = str(col_array.dtype)
            itemsize = f"{col_array.dtype.itemsize} B"
            lines.append(f" {i:>3}   {name:<15} {itemsize:<12} {dtype_str:<15}")

        lines.append("")
        lines.append(f"memory usage: {format_bytes(cbytes)}")
        lines.append(f"uncompressed size: {format_bytes(nbytes)}")
        lines.append(f"compression ratio: {ratio:.2f}x")
        lines.append("")

        print("\n".join(lines))

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

    def delete(self, ind: int | slice | str | Iterable) -> None:
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

    def extend(self, data: list | CTable | Any, *, validate: bool | None = None) -> None:
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

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    @profile
    def where(self, expr_result) -> CTable:
        if not (
            isinstance(expr_result, (blosc2.NDArray, blosc2.LazyExpr))
            and (getattr(expr_result, "dtype", None) == np.bool_)
        ):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(expr_result).__name__}")

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

    # ------------------------------------------------------------------
    # Persistence (not yet implemented)
    # ------------------------------------------------------------------

    def save(self, urlpath: str, group: str = "table") -> None: ...

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> CTable: ...
