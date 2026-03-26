#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Imports for CTable"""

from __future__ import annotations

from collections.abc import Iterable
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


from pydantic import BaseModel

import blosc2

RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class MaxLen:
    def __init__(self, length: int):
        self.length = int(length)


#############################
####  Row model examples  ###
#############################
"""
class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

class RowModel2(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    # name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")
    score: Annotated[float, NumpyDtype(np.float32)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

class RowModel3(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    #name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")"""


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



def _resolve_field_dtype(field) -> tuple[np.dtype, int]:
    """Return (numpy dtype, display_width) for a pydantic model field.

    Extracts dtype from NumpyDtype metadata when present, otherwise falls
    back to a sensible default for each Python primitive type.
    """
    annotation = field.annotation
    origin = getattr(annotation, "__origin__", annotation)

    # str / bytes: look for MaxLen metadata, build fixed-width dtype
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

    # Check for explicit NumpyDtype metadata (overrides primitive defaults)
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


def _default_display_width(origin) -> int:
    """Return a sensible display column width for a given Python type."""
    return {int: 12, float: 15, bool: 6, complex: 25}.get(origin, 20)


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
        # < (Less than)
        return self._raw_col < other

    def __le__(self, other):
        # <= (Less than or equal to)
        return self._raw_col <= other

    def __eq__(self, other):
        # == (Equal to)
        return self._raw_col == other

    def __ne__(self, other):
        # != (Not equal to)
        return self._raw_col != other

    def __gt__(self, other):
        # > (Greater than)
        return self._raw_col > other

    def __ge__(self, other):
        # >= (Greater than or equal to)
        return self._raw_col >= other

    @property
    def dtype(self):
        return self._raw_col.dtype

    def to_numpy(self):
        real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
        return self._raw_col[real_pos[:]]


class CTable(Generic[RowT]):
    def __init__(
        self, row_type: type[RowT], new_data=None, expected_size: int = 1_048_576, compact: bool = False
    ) -> None:
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}
        self.col_names = []
        self.row = _RowIndexer(self)
        self.auto_compact = compact
        self.base = None

        c, b = compute_chunks_blocks((expected_size,))
        self._valid_rows = blosc2.zeros(shape=(expected_size,), dtype=np.bool_, chunks=c, blocks=b)

        for name, field in row_type.model_fields.items():
            self.col_names.append(name)
            dt, display_width = _resolve_field_dtype(field)
            final_width = max(len(name), display_width)
            self._col_widths[name] = final_width
            self._cols[name] = blosc2.zeros(shape=(expected_size,), dtype=dt, chunks=c, blocks=b)

        if new_data is not None:
            self._load_initial_data(new_data)

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

    def __str__(self):
        retval = []
        cont = 0

        # We print the header
        for name in self._cols:
            retval.append(f"{name:^{self._col_widths[name]}} |")
            cont += self._col_widths[name] + 2
        retval.append("\n")
        for _i in range(cont):
            retval.append("-")
        retval.append("\n")

        # We print the rows

        """Change this. Use where"""
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

        retval = CTable(self._row_type, compact=self.auto_compact, expected_size=len(self._valid_rows))
        retval._cols = self._cols
        retval._n_rows = blosc2.count_nonzero(new_valid_rows)
        retval._col_widths = self._col_widths
        retval.col_names = self.col_names
        retval.base = self
        retval._valid_rows = new_valid_rows

        return retval

    def head(self, N: int = 5) -> CTable:
        """
        # Alternative code, slower with big data
        if n <= 0:
            return CTable(self._row_type, compact=self.auto_compact)

        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        n_take = min(n, self._n_rows)

        retval = CTable(self._row_type, compact=self.auto_compact)
        retval._n_rows = n_take
        retval._valid_rows[:n_take] = True

        for k in self._cols.keys():
            retval._cols[k][:n_take] = self._cols[k][real_poss[:n_take]]

        return retval"""
        if N <= 0:
            # If N is 0 or negative, return an empty table
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))

        arr = self._valid_rows
        count = 0
        chunk_size = arr.chunks[0]
        pos_N_true = -1
        if N <= 0:
            return self.view(blosc2.zeros(shape=len(arr), dtype=np.bool_))
        for info in arr.iterchunks_info():
            actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
            chunk_start = info.nchunk * chunk_size

            # All False without decompressing -> skip
            if info.special == blosc2.SpecialValue.ZERO:
                continue

            # Repeated value -> check if True or False
            if info.special == blosc2.SpecialValue.VALUE:
                val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                if not val:
                    continue  # all False, skip
                # All True: target is at offset (N - count - 1) within the chunk
                if count + actual_size < N:
                    count += actual_size
                    continue
                pos_N_true = chunk_start + (N - count - 1)
                break

            # General case: decompress only this chunk
            chunk_data = arr[chunk_start : chunk_start + actual_size]

            n_true = int(np.count_nonzero(chunk_data))
            if count + n_true < N:
                count += n_true
                continue

            # The N-th True is in this chunk
            pos_N_true = chunk_start + int(np.flatnonzero(chunk_data)[N - count - 1])
            break

        if pos_N_true == -1:
            return self.view(self._valid_rows)

        if pos_N_true < len(self._valid_rows) // 2:
            mask_arr = blosc2.zeros(shape=len(arr), dtype=np.bool_)
            mask_arr[: pos_N_true + 1] = True
        else:
            mask_arr = blosc2.ones(shape=len(arr), dtype=np.bool_)
            mask_arr[pos_N_true + 1 :] = False

        mask_arr = (mask_arr & self._valid_rows).compute()
        return self.view(mask_arr)

    def tail(self, N: int = 5) -> CTable:
        if N <= 0:
            # If N is 0 or negative, return an empty table
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))

        arr = self._valid_rows
        count = 0
        chunk_size = arr.chunks[0]
        pos_N_true = -1

        # Convert to list to iterate chunks in reverse order (metadata only, ~0 memory)
        for info in reversed(list(arr.iterchunks_info())):
            actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
            chunk_start = info.nchunk * chunk_size

            # All False without decompressing -> skip
            if info.special == blosc2.SpecialValue.ZERO:
                continue

            # Repeated value -> check if True or False
            if info.special == blosc2.SpecialValue.VALUE:
                val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                if not val:
                    continue  # all False, skip

                # All True: target is at offset 'actual_size - (N - count)' from chunk start
                if count + actual_size < N:
                    count += actual_size
                    continue
                pos_N_true = chunk_start + actual_size - (N - count)
                break

            # General case: decompress only this chunk
            chunk_data = arr[chunk_start : chunk_start + actual_size]

            n_true = int(np.count_nonzero(chunk_data))
            if count + n_true < N:
                count += n_true
                continue

            # The N-th True from the end is in this chunk
            # We use negative indexing [-(N - count)] to get elements from the back
            pos_N_true = chunk_start + int(np.flatnonzero(chunk_data)[-(N - count)])
            break

        if pos_N_true == -1:
            return self.view(self._valid_rows)

        # Mask creation logic reversed: keep everything from pos_N_true to the end
        if pos_N_true > len(arr) // 2:
            # We keep a small tail (less than half the array): start with zeros
            mask_arr = blosc2.zeros(shape=len(arr), dtype=np.bool_)
            mask_arr[pos_N_true:] = True
        else:
            # We keep a large tail (more than half the array): start with ones
            mask_arr = blosc2.ones(shape=len(arr), dtype=np.bool_)
            if pos_N_true > 0:
                mask_arr[:pos_N_true] = False

        # Compute intersection with existing valid rows and creating view
        mask_arr = (mask_arr & self._valid_rows).compute()
        return self.view(mask_arr)

    def __getitem__(self, s: str):
        if s in self._cols:
            return Column(self, s)
        return None

    def __getattr__(self, s: str):
        if s in self._cols:
            return Column(self, s)
        return super().__getattribute__(s)

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

    @property
    def nrows(self) -> int:
        return self._n_rows

    @property
    def ncols(self) -> int:
        return len(self._cols)

    def info(self) -> None:
        """
        Prints a concise summary of the CTable, including the column names,
        their data types, and memory layout.
        """
        n_cols = len(self._cols)
        n_rows = len(self)

        # Calculate global memory usage
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

        # New Header: replaced "Non-Null Count" with internal Array length & Itemsize
        header = f" {'#':>3}   {'Column':<15} {'Itemsize':<12} {'Dtype':<15}"
        lines.append(header)
        lines.append(f" {'---':>3}  {'------':<15} {'--------':<12} {'-----':<15}")

        for i, name in enumerate(self.col_names):
            col_array = self._cols[name]
            dtype_str = str(col_array.dtype)
            itemsize = f"{col_array.dtype.itemsize} B"

            line = f" {i:>3}   {name:<15} {itemsize:<12} {dtype_str:<15}"
            lines.append(line)

        lines.append("")
        lines.append(f"memory usage: {format_bytes(cbytes)}")
        lines.append(f"uncompressed size: {format_bytes(nbytes)}")
        lines.append(f"compression ratio: {ratio:.2f}x")
        lines.append("")

        print("\n".join(lines))

    def append(self, data: list | np.void | np.ndarray) -> None:
        if self.base is not None:
            raise TypeError("Cannot extend view.")

        is_list = isinstance(data, (list, tuple))

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

        pos = last_true_pos + 1

        if pos >= len(self._valid_rows):
            c = len(self._valid_rows)
            for v in self._cols.values():
                v.resize((c * 2,))
            self._valid_rows.resize((c * 2,))

        if is_list:
            for i, col_array in enumerate(self._cols.values()):
                col_array[pos] = data[i]
        else:
            for name, col_array in self._cols.items():
                col_array[pos] = data[name]

        self._valid_rows[pos] = True
        self._n_rows += 1

    def delete(self, ind: int | slice | str | Iterable) -> blosc2.NDArray:
        valid_rows_np = self._valid_rows[:]
        true_pos = np.where(valid_rows_np)[0]

        if isinstance(ind, Iterable) and not isinstance(ind, (str, bytes)):
            ind = list(ind)
        elif not isinstance(ind, int) and not isinstance(ind, slice):
            raise TypeError(f"Invalid type '{type(ind)}'")

        false_pos = true_pos[ind]

        new_mask_np = valid_rows_np.copy()
        new_mask_np[false_pos] = False

        new_mask = blosc2.asarray(new_mask_np)
        self._valid_rows = new_mask
        self._n_rows = blosc2.count_nonzero(self._valid_rows)

    def extend(self, data: list | CTable | Any) -> None:
        if self.base is not None:
            raise TypeError("Cannot extend view.")
        if len(data) <= 0:
            return
        ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0

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

        processed_cols = []
        for i, raw_col in enumerate(columns_to_insert):
            target_dtype = self._cols[current_col_names[i]].dtype
            b2_arr = blosc2.asarray(raw_col, dtype=target_dtype)
            processed_cols.append(b2_arr)

        end_pos = start_pos + new_nrows

        if self.auto_compact and end_pos >= len(self._valid_rows):
            self.compact()
            ultimas_validas = blosc2.where(
                self._valid_rows, np.array(range(len(self._valid_rows)))
            ).compute()
            start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0
            end_pos = start_pos + new_nrows

        while end_pos > len(self._valid_rows):
            c = len(self._valid_rows)
            for name in current_col_names:
                self._cols[name].resize((c * 2,))
            self._valid_rows.resize((c * 2,))

        # Do this per chunks
        for j, name in enumerate(current_col_names):
            self._cols[name][start_pos:end_pos] = processed_cols[j][:]

        self._valid_rows[start_pos:end_pos] = True
        self._n_rows = blosc2.count_nonzero(self._valid_rows)

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

    """Save & load are blank for now"""

    def save(self, urlpath: str, group: str = "table") -> None: ...

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> CTable: ...
