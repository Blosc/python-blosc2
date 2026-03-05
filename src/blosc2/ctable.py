#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""Imports for CTable"""

from __future__ import annotations  # ✅ PRIMERO (después de docstring)

from collections.abc import Iterable  # ✅ AHORA SÍ
from typing import Any, Generic, TypeVar

import numpy as np
from line_profiler import profile
from pydantic import BaseModel

import blosc2
from blosc2 import compute_chunks_blocks

RowT = TypeVar("RowT", bound=BaseModel)



class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class MaxLen:
    def __init__(self, length: int):
        self.length = int(length)


"""
class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True

'''class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    # name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")
    score: Annotated[float, NumpyDtype(np.float32)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True'''

class RowModel2(BaseModel):
    id: Annotated[int, NumpyDtype(np.int16)] = Field(ge=0)
    #name: Annotated[str, MaxLen(10)] = Field(default="unknown")
    name: Annotated[bytes, MaxLen(10)] = Field(default=b"unknown")"""





class _RowIndexer:
        def __init__(self, table):
            self._table = table

        def __getitem__(self, item):
            return self._table._run_row_logic(item)


class Column:
    def __init__(self, table: CTable, col_name: str):
        self._table = table
        self._col_name = col_name

    @property
    def _raw_col(self):
        return self._table._cols[self._col_name]

    @property
    def _valid_rows(self):
        return self._table._valid_rows

    def __getitem__(self, key: int | slice | list | np.ndarray):
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")

            arr = self._valid_rows
            count = 0
            chunk_size = arr.chunks[0]
            pos_true = -1

            for info in arr.iterchunks_info():
                actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
                chunk_start = info.nchunk * chunk_size

                if info.special == blosc2.SpecialValue.ZERO:
                    continue

                if info.special == blosc2.SpecialValue.VALUE:
                    val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                    if not val:
                        continue

                    if count + actual_size <= key:
                        count += actual_size
                        continue

                    pos_true = chunk_start + (key - count)
                    break

                chunk_data = arr[chunk_start: chunk_start + actual_size]
                n_true = int(np.count_nonzero(chunk_data))

                if count + n_true <= key:
                    count += n_true
                    continue

                pos_true = chunk_start + int(np.flatnonzero(chunk_data)[key - count])
                break

            if pos_true == -1:
                raise IndexError("Unexpected error finding physical index.")

            return self._raw_col[int(pos_true)]

        elif isinstance(key, slice):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
            lindices = range(*key.indices(len(real_pos)))
            phys_indices = np.array([real_pos[i] for i in lindices], dtype=np.int64)
            return self._raw_col[phys_indices]

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
            phys_indices = np.array([real_pos[i] for i in key], dtype=np.int64)
            return self._raw_col[phys_indices]

        raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, key: int | slice | list | np.ndarray, value): # noqa: C901
        if isinstance(key, int):
            n_rows = len(self)
            if key < 0:
                key += n_rows
            if not (0 <= key < n_rows):
                raise IndexError(f"index {key} is out of bounds for column with size {n_rows}")

            arr = self._valid_rows
            count = 0
            chunk_size = arr.chunks[0]
            pos_true = -1

            for info in arr.iterchunks_info():
                actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
                chunk_start = info.nchunk * chunk_size

                if info.special == blosc2.SpecialValue.ZERO:
                    continue

                if info.special == blosc2.SpecialValue.VALUE:
                    val = np.frombuffer(info.repeated_value, dtype=arr.dtype)[0]
                    if not val:
                        continue
                    if count + actual_size <= key:
                        count += actual_size
                        continue
                    pos_true = chunk_start + (key - count)
                    break

                chunk_data = arr[chunk_start: chunk_start + actual_size]
                n_true = int(np.count_nonzero(chunk_data))
                if count + n_true <= key:
                    count += n_true
                    continue

                pos_true = chunk_start + int(np.flatnonzero(chunk_data)[key - count])
                break

            self._raw_col[int(pos_true)] = value

        elif isinstance(key, slice):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
            lindices = range(*key.indices(len(real_pos)))
            phys_indices = np.array([real_pos[i] for i in lindices], dtype=np.int64)

            if isinstance(value, (list, tuple)):
                value = np.array(value, dtype=self._raw_col.dtype)

            self._raw_col[phys_indices] = value

        elif isinstance(key, (list, tuple, np.ndarray)):
            real_pos = blosc2.where(self._valid_rows, np.arange(len(self._valid_rows))).compute()
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

                data_chunk = self._raw_col[chunk_start: chunk_start + actual_size]
                yield from data_chunk
                continue

            mask_chunk = arr[chunk_start: chunk_start + actual_size]
            true_offsets = np.flatnonzero(mask_chunk)

            if len(true_offsets) == 0:
                continue

            physical_indices = chunk_start + true_offsets
            valid_data = self._raw_col[physical_indices.tolist()]

            yield from valid_data

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


class CTable(Generic[RowT]):


    def __init__(self, row_type: type[RowT], new_data = None, expected_size: int = 1_048_576, compact: bool = False) -> None:  # noqa: C901
        self._row_type = row_type
        self._cols: dict[str, blosc2.NDArray] = {}
        self._n_rows: int = 0
        self._col_widths: dict[str, int] = {}
        self.col_names = []
        self.row = _RowIndexer(self)
        self.auto_compact = compact
        self.base = None


        c, b = compute_chunks_blocks((expected_size,))
        self._valid_rows = blosc2.zeros(shape=(expected_size,), dtype = np.bool_ , chunks=c, blocks=b)


        for name, field in row_type.model_fields.items():
            self.col_names.append(name)
            origin = getattr(field.annotation, "__origin__", field.annotation)

            if origin is str or field.annotation is str:
                max_len = 32  # Default MaxLen
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"U{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin is bytes or field.annotation is bytes:
                max_len = 32    # Default MaxLen
                if hasattr(field.annotation, "__metadata__"):
                    for meta in field.annotation.__metadata__:
                        if isinstance(meta, MaxLen):
                            max_len = meta.max_length
                            break
                dt = np.dtype(f"S{max_len}")
                display_width = max(10, min(max_len, 50))

            elif origin is int or field.annotation is int:
                dt = np.int64
                display_width = 12

            elif origin is float or field.annotation is float:
                dt = np.float64
                display_width = 15

            elif origin is bool or field.annotation is bool:
                dt = np.bool_
                display_width = 6  # "True" / "False" fit in 5-6 chars

            elif origin is complex or field.annotation is complex:
                dt = np.complex128
                display_width = 25
            else:
                dt = np.object_
                display_width = 20

            final_width = max(len(name), display_width)
            self._col_widths[name] = final_width        # Usefull in __str__

            self._cols[name] = blosc2.zeros(shape=(expected_size,), dtype=dt, chunks=c, blocks=b)

        if new_data is not None:
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

        for name in self._cols:
            retval.append(f"{name:^{self._col_widths[name]}} |")
            cont += self._col_widths[name]+2
        retval.append("\n")
        for _ in range(cont):
            retval.append("-")
        retval.append("\n")


        # We print the rows

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

    def view(self, new_valid_rows):
        if not (isinstance(new_valid_rows, (blosc2.NDArray, blosc2.LazyExpr)) and
                (getattr(new_valid_rows, 'dtype', None) == np.bool_)):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(new_valid_rows).__name__}")

        new_valid_rows = new_valid_rows.compute() if isinstance(new_valid_rows, blosc2.LazyExpr) else new_valid_rows

        if len(self._valid_rows) != len(new_valid_rows):
            raise ValueError()

        retval = CTable(self._row_type, compact=self.auto_compact, expected_size=len(self._valid_rows))
        retval._cols = self._cols
        retval._n_rows= blosc2.count_nonzero(new_valid_rows)
        retval._col_widths= self._col_widths
        retval.col_names = self.col_names
        retval.base = self
        retval._valid_rows = new_valid_rows

        return retval

    def head(self, N: int = 5) -> CTable:
        '''
        # Alternative code, slowe with big data
        if n <= 0:
            return CTable(self._row_type, compact=self.auto_compact)

        real_poss = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        n_take = min(n, self._n_rows)

        retval = CTable(self._row_type, compact=self.auto_compact)
        retval._n_rows = n_take
        retval._valid_rows[:n_take] = True

        for k in self._cols.keys():
            retval._cols[k][:n_take] = self._cols[k][real_poss[:n_take]]

        return retval'''
        if N <= 0:
            return self.view(blosc2.zeros(shape=len(self._valid_rows), dtype=np.bool_))

        arr = self._valid_rows
        count = 0
        chunk_size = arr.chunks[0]
        pos_N_true = -1
        if (N<=0):
            return self.view(blosc2.zeros(shape=len(arr), dtype=np.bool_))

        for info in arr.iterchunks_info():
            actual_size = min(chunk_size, arr.shape[0] - info.nchunk * chunk_size)
            chunk_start = info.nchunk * chunk_size

            # All False without decompressing → skip
            if info.special == blosc2.SpecialValue.ZERO:
                continue

            # Repeated value → check if True or False
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
            chunk_data = arr[chunk_start: chunk_start + actual_size]

            n_true = int(np.count_nonzero(chunk_data))
            if count + n_true < N:
                count += n_true
                continue

            # The N-th True is in this chunk
            pos_N_true = chunk_start + int(np.flatnonzero(chunk_data)[N - count - 1])
            break

        if pos_N_true == -1:
            return self.view(self._valid_rows)

        if pos_N_true < len(self._valid_rows)//2:
            mask_arr = blosc2.zeros(shape=len(arr), dtype=np.bool_)
            mask_arr[:pos_N_true+1] = True
        else:
            mask_arr = blosc2.ones(shape=len(arr), dtype=np.bool_)
            mask_arr[pos_N_true+1:] = False

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

            # All False without decompressing → skip
            if info.special == blosc2.SpecialValue.ZERO:
                continue

            # Repeated value → check if True or False
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
            chunk_data = arr[chunk_start: chunk_start + actual_size]

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
        block_size= self._valid_rows.blocks[0]
        end = min(block_size, self._n_rows)
        while start < end:
            for _, v in self._cols.items():
                v[start:end] = v[real_poss[start:end]]
            start += block_size
            end = min(end + block_size, self._n_rows)

        self._valid_rows[:self._n_rows] = True
        self._valid_rows[self._n_rows:] = False

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
            elif bytes_size < 1024 ** 2:
                return f"{bytes_size / 1024:.2f} KB"
            elif bytes_size < 1024 ** 3:
                return f"{bytes_size / (1024 ** 2):.2f} MB"
            else:
                return f"{bytes_size / (1024 ** 3):.2f} GB"

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

    def append(self, data: list | np.void | np.ndarray) -> None: # noqa: C901
        if self.base is not None:
            raise TypeError("Cannot extend view.")

        is_list = isinstance(data, (list, tuple))
        col_values = list(self._cols.values())
        col_names = self.col_names

        if isinstance(data, dict):
            raise TypeError("Dictionaries are not supported in append.")

        if is_list and len(data) != len(col_values):
            raise ValueError(f"Expected {len(col_values)} values, received {len(data)}")

        if is_list:
            for i, val in enumerate(data):
                target_dtype = col_values[i].dtype
                try:
                    np.array(val, dtype=target_dtype)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"Value '{val}' is not compatible with column '{col_names[i]}' of type {target_dtype}") from None
        else:
            for name, arr in self._cols.items():
                try:
                    val = data[name]
                except (IndexError, KeyError, ValueError):
                    raise ValueError(f"Input data does not contain required field '{name}'") from None
                try:
                    np.array(val, dtype=arr.dtype)
                except (ValueError, TypeError):
                    raise TypeError(f"Value '{val}' in field '{name}' is not compatible with type {arr.dtype}") from None

        ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0
        if pos >= len(self._valid_rows):
            c = len(self._valid_rows)
            for _,v in self._cols.items():
                v.resize((c * 2,))
            self._valid_rows.resize((c * 2,))

        if is_list:
            for i, col_array in enumerate(col_values):
                col_array[pos] = data[i]
        else:
            for name, col_array in self._cols.items():
                col_array[pos] = data[name]
        self._valid_rows[pos] = True

        self._n_rows += 1

    def delete(self, ind: int | slice | str | Iterable):
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
        if len(data) <=0:
            return
        ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
        start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0

        current_col_names = self.col_names
        columns_to_insert = []
        new_nrows = 0

        if hasattr(data, "_cols") and hasattr(data, "_n_rows"):
            for name in current_col_names:
                col = data._cols[name][:data._n_rows]
                columns_to_insert.append(col)
            new_nrows = data._n_rows
        else:
            if isinstance(data, np.ndarray) and data.dtype.names is not None:
                for name in current_col_names:
                    columns_to_insert.append(data[name])
                new_nrows = len(data)
            else:
                columns_to_insert = list(zip(*data, strict=True))
                new_nrows = len(data)

        processed_cols = []
        for i, raw_col in enumerate(columns_to_insert):
            target_dtype = self._cols[current_col_names[i]].dtype
            b2_arr = blosc2.asarray(raw_col, dtype=target_dtype)
            processed_cols.append(b2_arr)

        end_pos = start_pos + new_nrows

        if self.auto_compact and end_pos >= len(self._valid_rows):
            self.compact()
            ultimas_validas = blosc2.where(self._valid_rows, np.array(range(len(self._valid_rows)))).compute()
            start_pos = ultimas_validas[-1] + 1 if len(ultimas_validas) > 0 else 0
            end_pos = start_pos + new_nrows

        while end_pos > len(self._valid_rows):
            c = len(self._valid_rows)
            for name in current_col_names:
                self._cols[name].resize((c*2,))
            self._valid_rows.resize((c*2,))




        # Do this per chunks
        for j, name in enumerate(current_col_names):
            self._cols[name][start_pos:end_pos] = processed_cols[j][:]

        self._valid_rows[start_pos:end_pos] = True
        self._n_rows = blosc2.count_nonzero(self._valid_rows)

    @profile
    def where(self, expr_result) -> CTable:
        if not (isinstance(expr_result, (blosc2.NDArray, blosc2.LazyExpr)) and
                (getattr(expr_result, 'dtype', None) == np.bool_)):
            raise TypeError(f"Expected boolean blosc2.NDArray or LazyExpr, got {type(expr_result).__name__}")

        filter = expr_result.compute() if isinstance(expr_result, blosc2.LazyExpr) else expr_result

        target_len = len(self._valid_rows)

        if len(filter) > target_len:
            filter = filter[:target_len]
        elif len(filter) < target_len:
            padding = blosc2.zeros(target_len, dtype=np.bool_)
            padding[:len(filter)] = filter[:]
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

    """Save & load are blank"""

    def save(self, urlpath: str, group: str = "table") -> None:
        ...

    @classmethod
    def load(cls, urlpath: str, group: str = "table", row_type: type[RowT] | None = None) -> CTable:
        ...