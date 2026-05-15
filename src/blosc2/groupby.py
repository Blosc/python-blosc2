#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Group-by support for :class:`blosc2.CTable`.

This module contains the Phase-1, NumPy-based implementation.  It is deliberately
chunked and columnar: only grouping columns, aggregation columns, and the
live-row mask are read from the source table.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from blosc2.schema import DictionarySpec, SchemaSpec, float64, int64
from blosc2.schema import bool as b2_bool
from blosc2.schema import field as b2_field

if TYPE_CHECKING:  # pragma: no cover
    from blosc2.ctable import CTable


AggName = Literal["size", "count", "sum", "mean", "min", "max"]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*$")
_NAN_KEY = ("__blosc2_groupby_nan__",)


@dataclasses.dataclass
class _AggSpec:
    input_col: str | None
    op: AggName
    output_col: str


@dataclasses.dataclass
class _AggState:
    op: AggName
    value: Any = None
    count: int = 0


class CTableGroupBy:
    """Deferred group-by operation returned by :meth:`CTable.group_by`.

    The object stores the source table, grouping keys, and execution options.
    It is not a :class:`CTable` view and does not materialize grouped data until
    a terminal method such as :meth:`size`, :meth:`count`, or :meth:`agg` is
    called.
    """

    def __init__(
        self,
        table: CTable,
        keys: str | Sequence[str],
        *,
        sort: bool = False,
        dropna: bool = True,
        engine: str = "auto",
        chunk_size: int | None = None,
    ) -> None:
        if isinstance(keys, str):
            keys = [keys]
        else:
            keys = list(keys)
        if not keys:
            raise ValueError("group_by() requires at least one key column")

        self.table = table
        self.keys = [table._logical_to_physical_name(k) for k in keys]
        self.sort = bool(sort)
        self.dropna = bool(dropna)
        self.engine = engine
        self.chunk_size = chunk_size

        for name in self.keys:
            if name in table._computed_cols:
                raise NotImplementedError("group_by() over computed columns is not supported yet")
            if name not in table._cols:
                raise KeyError(f"No column named {name!r}. Available: {table.col_names}")
            col_info = table._schema.columns_by_name[name]
            if table._is_list_column(col_info) or table._is_varlen_scalar_column(col_info):
                raise TypeError(f"Cannot group by variable-length/list column {name!r} in Phase 1")

    def size(self):
        """Return row counts per group as a new :class:`CTable`.

        This is equivalent to SQL ``COUNT(*)``: it counts rows in each group and
        is independent of null values in non-key columns.
        """
        return self._execute([_AggSpec(None, "size", "size")])

    def count(self, column: str):
        """Return non-null value counts for *column* per group.

        This is equivalent to SQL ``COUNT(column)`` and to
        ``group_by(...).agg({column: "count"})``.
        """
        col = self.table._logical_to_physical_name(column)
        return self._execute([_AggSpec(col, "count", f"{col}_count")])

    def agg(self, aggregations: Mapping[str, str | Sequence[str]]):
        """Aggregate value columns per group.

        Parameters
        ----------
        aggregations:
            Mapping from input column name to an aggregation name or list of
            names.  Supported operations in Phase 1 are ``"count"``, ``"sum"``,
            ``"mean"``, ``"min"``, ``"max"`` and the special row-count spelling
            ``{"*": "size"``}.
        """
        specs = self._normalize_aggs(aggregations)
        return self._execute(specs)

    def _normalize_aggs(self, aggregations: Mapping[str, str | Sequence[str]]) -> list[_AggSpec]:
        if not isinstance(aggregations, Mapping) or not aggregations:
            raise ValueError("agg() requires a non-empty mapping")
        specs: list[_AggSpec] = []
        for col_name, ops in aggregations.items():
            if isinstance(ops, str):
                op_list = [ops]
            else:
                op_list = list(ops)
            if not op_list:
                raise ValueError(f"No aggregations specified for column {col_name!r}")

            if col_name == "*":
                for op in op_list:
                    if op != "size":
                        raise ValueError("Only the 'size' aggregation is supported for '*' input")
                    specs.append(_AggSpec(None, "size", "size"))
                continue

            physical = self.table._logical_to_physical_name(col_name)
            self._validate_value_column(physical)
            for op in op_list:
                if op not in {"count", "sum", "mean", "min", "max"}:
                    raise ValueError(f"Unsupported aggregation {op!r}")
                self._validate_agg_for_column(physical, op)
                specs.append(_AggSpec(physical, op, f"{physical}_{op}"))
        output_names = [s.output_col for s in specs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("Aggregation output column names must be unique")
        return specs

    def _validate_agg_for_column(self, name: str, op: str) -> None:
        dtype = getattr(self.table._schema.columns_by_name[name].spec, "dtype", None)
        if op in {"sum", "mean"} and dtype is not None and dtype.kind not in "biuf":
            raise TypeError(f"Aggregation {op!r} is not supported for column {name!r} with dtype {dtype}")
        if op in {"min", "max"} and dtype is not None and dtype.kind == "c":
            raise TypeError(f"Aggregation {op!r} is not supported for complex column {name!r}")

    def _validate_value_column(self, name: str) -> None:
        if name in self.table._computed_cols:
            raise NotImplementedError("group_by() aggregations over computed columns are not supported yet")
        if name not in self.table._cols:
            raise KeyError(f"No column named {name!r}. Available: {self.table.col_names}")
        col_info = self.table._schema.columns_by_name[name]
        if self.table._is_list_column(col_info) or self.table._is_varlen_scalar_column(col_info):
            raise TypeError(f"Cannot aggregate variable-length/list column {name!r} in Phase 1")
        if self.table._is_dictionary_column(col_info):
            raise TypeError(f"Cannot aggregate dictionary column {name!r} in Phase 1")

    def _execute(self, specs: list[_AggSpec]):
        self._validate_output_names(specs)
        fast = self._try_execute_cython_i32_f64_sum(specs)
        if fast is not None:
            return fast
        fast = self._try_execute_cython_float_integral_key_f64_sum(specs)
        if fast is not None:
            return fast
        fast = self._try_execute_dense_single_int_key(specs)
        if fast is not None:
            return fast

        acc: dict[Any, dict[str, _AggState]] = {}
        key_values: dict[Any, tuple[Any, ...]] = {}

        phys_len = len(self.table._valid_rows)
        chunk_size = self._chunk_size()
        value_cols = sorted({s.input_col for s in specs if s.input_col is not None})

        for start in range(0, phys_len, chunk_size):
            stop = min(start + chunk_size, phys_len)
            valid = np.asarray(self.table._valid_rows[start:stop], dtype=bool)
            if not np.any(valid):
                continue

            raw_keys = [self._read_key_chunk(name, start, stop) for name in self.keys]
            live_mask = valid.copy()
            if self.dropna:
                for name, values in zip(self.keys, raw_keys, strict=True):
                    live_mask &= ~self._null_mask(name, values, is_key=True)
            if not np.any(live_mask):
                continue

            keys_live = [np.asarray(values)[live_mask] for values in raw_keys]
            n_live = len(keys_live[0])
            if n_live == 0:
                continue

            unique_keys, inverse = self._factorize_keys(keys_live)
            value_chunks = {
                name: np.asarray(self.table._cols[name][start:stop])[live_mask] for name in value_cols
            }

            partials = self._compute_partials(specs, unique_keys, inverse, value_chunks)
            display_keys = self._display_keys(unique_keys)
            normalized_keys = self._normalized_keys(display_keys)
            self._merge_partials(acc, key_values, normalized_keys, display_keys, partials, specs)

        rows = self._final_rows(acc, key_values, specs)
        return self._build_result(rows, specs)

    def _try_execute_cython_i32_f64_sum(self, specs: list[_AggSpec]):  # noqa: C901
        """Cython fast path for one int32 key and one non-null float64 sum."""
        if len(self.keys) != 1 or len(specs) != 1 or specs[0].op != "sum" or self.sort:
            return None
        spec = specs[0]
        if spec.input_col is None:
            return None
        key_name = self.keys[0]
        key_info = self.table._schema.columns_by_name[key_name]
        value_info = self.table._schema.columns_by_name[spec.input_col]
        if self.table._is_dictionary_column(key_info):
            key_arr = self.table._cols[key_name].codes
            key_is_dict = True
            key_null = int(key_info.spec.null_code)
            skip_key_null = self.dropna
        else:
            key_arr = self.table._cols[key_name]
            key_is_dict = False
            key_dtype = getattr(key_info.spec, "dtype", None)
            if key_dtype != np.dtype(np.int32):
                return None
            key_null_value = getattr(key_info.spec, "null_value", None)
            skip_key_null = self.dropna and key_null_value is not None
            key_null = 0 if key_null_value is None else int(key_null_value)
        value_dtype = getattr(value_info.spec, "dtype", None)
        if value_dtype != np.dtype(np.float64) or getattr(value_info.spec, "null_value", None) is not None:
            return None
        try:
            from blosc2 import indexing_ext
        except ImportError:
            return None
        kernel = getattr(indexing_ext, "groupby_dense_i32_f64_sum_checked", None)
        if kernel is None:
            return None

        compact_limit = 10_000_000
        sums = np.zeros(0, dtype=np.float64)
        present = np.zeros(0, dtype=bool)

        def ensure_size(size: int) -> bool:
            nonlocal sums, present
            if size > compact_limit:
                return False
            if size <= len(sums):
                return True
            old = len(sums)
            sums = np.pad(sums, (0, size - old), constant_values=0)
            present = np.pad(present, (0, size - old), constant_values=False)
            return True

        phys_len = len(self.table._valid_rows)
        chunk_size = self._chunk_size()
        for start in range(0, phys_len, chunk_size):
            stop = min(start + chunk_size, phys_len)
            valid = np.asarray(self.table._valid_rows[start:stop], dtype=bool)
            if not np.any(valid):
                continue
            keys = np.asarray(key_arr[start:stop], dtype=np.int32)
            values = np.asarray(self.table._cols[spec.input_col][start:stop], dtype=np.float64)
            status = int(kernel(keys, values, valid, sums, present, skip_key_null, key_null, False))
            if status == -1:
                return None
            if status > 0:
                if not ensure_size(status):
                    return None
                status = int(kernel(keys, values, valid, sums, present, skip_key_null, key_null, False))
                if status != 0:
                    return None

        rows = []
        for code in np.nonzero(present)[0]:
            key_value = self.table._cols[key_name].decode(int(code)) if key_is_dict else int(code)
            rows.append({key_name: key_value, spec.output_col: float(sums[code])})
        return self._build_result(rows, specs)

    def _try_execute_cython_float_integral_key_f64_sum(self, specs: list[_AggSpec]):  # noqa: C901
        """Cython fast path for integral float32/float64 keys and one non-null float64 sum."""
        if len(self.keys) != 1 or len(specs) != 1 or specs[0].op != "sum" or self.sort:
            return None
        spec = specs[0]
        if spec.input_col is None:
            return None
        key_name = self.keys[0]
        key_info = self.table._schema.columns_by_name[key_name]
        value_info = self.table._schema.columns_by_name[spec.input_col]
        key_dtype = getattr(key_info.spec, "dtype", None)
        value_dtype = getattr(value_info.spec, "dtype", None)
        if key_dtype not in {np.dtype(np.float32), np.dtype(np.float64)} or value_dtype != np.dtype(
            np.float64
        ):
            return None
        if getattr(value_info.spec, "null_value", None) is not None:
            return None
        # The fast path can skip NaNs.  If dropna=False and NaNs are present,
        # the Cython kernel reports unsupported and we fall back to generic
        # grouping, which can materialize a NaN group.
        skip_key_nan = self.dropna
        try:
            from blosc2 import indexing_ext
        except ImportError:
            return None
        kernel_name = (
            "groupby_dense_f32_integral_key_f64_sum_checked"
            if key_dtype == np.dtype(np.float32)
            else "groupby_dense_f64_integral_key_f64_sum_checked"
        )
        kernel = getattr(indexing_ext, kernel_name, None)
        if kernel is None:
            return None

        compact_limit = 10_000_000
        sums = np.zeros(0, dtype=np.float64)
        present = np.zeros(0, dtype=bool)

        def ensure_size(size: int) -> bool:
            nonlocal sums, present
            if size > compact_limit:
                return False
            if size <= len(sums):
                return True
            old = len(sums)
            sums = np.pad(sums, (0, size - old), constant_values=0)
            present = np.pad(present, (0, size - old), constant_values=False)
            return True

        phys_len = len(self.table._valid_rows)
        chunk_size = self._chunk_size()
        for start in range(0, phys_len, chunk_size):
            stop = min(start + chunk_size, phys_len)
            valid = np.asarray(self.table._valid_rows[start:stop], dtype=bool)
            if not np.any(valid):
                continue
            keys = np.asarray(self.table._cols[key_name][start:stop], dtype=key_dtype)
            values = np.asarray(self.table._cols[spec.input_col][start:stop], dtype=np.float64)
            status = int(kernel(keys, values, valid, sums, present, skip_key_nan, False))
            if status == -1:
                return None
            if status > 0:
                if not ensure_size(status):
                    return None
                status = int(kernel(keys, values, valid, sums, present, skip_key_nan, False))
                if status != 0:
                    return None

        rows = [
            {key_name: float(code), spec.output_col: float(sums[code])} for code in np.nonzero(present)[0]
        ]
        return self._build_result(rows, specs)

    def _try_execute_dense_single_int_key(self, specs: list[_AggSpec]):  # noqa: C901
        """Fast path for one dense integer/dictionary-code key.

        This avoids per-chunk ``np.unique`` and Python dictionary merging.  It is
        intentionally conservative: keys must be non-negative and the observed
        key range must stay reasonably compact.
        """
        if len(self.keys) != 1:
            return None
        key_name = self.keys[0]
        key_info = self.table._schema.columns_by_name[key_name]
        key_is_dict = self.table._is_dictionary_column(key_info)
        key_dtype = np.dtype(np.int32) if key_is_dict else getattr(key_info.spec, "dtype", None)
        if key_dtype is None or key_dtype.kind not in "biu":
            return None
        if any(spec.op in {"min", "max"} and spec.input_col is not None for spec in specs):
            for spec in specs:
                if spec.op in {"min", "max"} and spec.input_col is not None:
                    dtype = getattr(self.table._schema.columns_by_name[spec.input_col].spec, "dtype", None)
                    if dtype is None or np.dtype(dtype).kind not in "biufmM":
                        return None

        compact_limit = 10_000_000
        present = np.zeros(0, dtype=bool)
        states: dict[str, Any] = {}
        for spec in specs:
            if spec.op in {"size", "count"}:
                states[spec.output_col] = np.zeros(0, dtype=np.int64)
            elif spec.op == "sum":
                out_dtype = np.int64
                if spec.input_col is not None:
                    dtype = np.dtype(self.table._schema.columns_by_name[spec.input_col].spec.dtype)
                    out_dtype = np.float64 if dtype.kind == "f" else np.int64
                states[spec.output_col] = np.zeros(0, dtype=out_dtype)
            elif spec.op == "mean":
                states[spec.output_col] = (np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64))
            elif spec.op in {"min", "max"}:
                assert spec.input_col is not None
                dtype = np.dtype(self.table._schema.columns_by_name[spec.input_col].spec.dtype)
                identity = _max_identity(dtype) if spec.op == "min" else _min_identity(dtype)
                states[spec.output_col] = (np.full(0, identity, dtype=dtype), np.zeros(0, dtype=bool))

        def ensure_size(size: int) -> bool:
            nonlocal present, states
            if size > compact_limit:
                return False
            if size <= len(present):
                return True
            old = len(present)
            present = np.pad(present, (0, size - old), constant_values=False)
            for spec in specs:
                state = states[spec.output_col]
                if spec.op in {"size", "count", "sum"}:
                    states[spec.output_col] = np.pad(state, (0, size - old), constant_values=0)
                elif spec.op == "mean":
                    sums, counts = state
                    states[spec.output_col] = (
                        np.pad(sums, (0, size - old), constant_values=0),
                        np.pad(counts, (0, size - old), constant_values=0),
                    )
                elif spec.op in {"min", "max"}:
                    values, has = state
                    dtype = values.dtype
                    identity = _max_identity(dtype) if spec.op == "min" else _min_identity(dtype)
                    states[spec.output_col] = (
                        np.pad(values, (0, size - old), constant_values=identity),
                        np.pad(has, (0, size - old), constant_values=False),
                    )
            return True

        phys_len = len(self.table._valid_rows)
        chunk_size = self._chunk_size()
        value_cols = sorted({s.input_col for s in specs if s.input_col is not None})
        for start in range(0, phys_len, chunk_size):
            stop = min(start + chunk_size, phys_len)
            valid = np.asarray(self.table._valid_rows[start:stop], dtype=bool)
            if not np.any(valid):
                continue
            raw_keys = self._read_key_chunk(key_name, start, stop)
            live_mask = valid.copy()
            if self.dropna:
                live_mask &= ~self._null_mask(key_name, raw_keys, is_key=True)
            if not np.any(live_mask):
                continue
            keys = np.asarray(raw_keys[live_mask])
            if keys.dtype.kind == "b":
                keys = keys.astype(np.int8, copy=False)
            if len(keys) == 0:
                continue
            min_key = int(np.min(keys))
            if min_key < 0:
                return None
            max_key = int(np.max(keys))
            if not ensure_size(max_key + 1):
                return None
            present[keys] = True
            value_chunks = {
                name: np.asarray(self.table._cols[name][start:stop])[live_mask] for name in value_cols
            }

            for spec in specs:
                if spec.op == "size":
                    states[spec.output_col] += np.bincount(keys, minlength=len(present)).astype(np.int64)
                    continue
                assert spec.input_col is not None
                values = value_chunks[spec.input_col]
                non_null = ~self._null_mask(spec.input_col, values, is_key=False)
                if spec.op == "count":
                    states[spec.output_col] += np.bincount(
                        keys, weights=non_null.astype(np.int64), minlength=len(present)
                    ).astype(np.int64)
                elif spec.op == "sum":
                    state = states[spec.output_col]
                    if values.dtype.kind in "biu":
                        np.add.at(state, keys[non_null], values[non_null].astype(np.int64, copy=False))
                    else:
                        state += np.bincount(
                            keys, weights=np.where(non_null, values, 0), minlength=len(present)
                        ).astype(state.dtype, copy=False)
                elif spec.op == "mean":
                    sums, counts = states[spec.output_col]
                    sums += np.bincount(keys, weights=np.where(non_null, values, 0), minlength=len(present))
                    counts += np.bincount(
                        keys, weights=non_null.astype(np.int64), minlength=len(present)
                    ).astype(np.int64)
                elif spec.op in {"min", "max"}:
                    values_state, has_state = states[spec.output_col]
                    if spec.op == "min":
                        np.minimum.at(values_state, keys[non_null], values[non_null])
                    else:
                        np.maximum.at(values_state, keys[non_null], values[non_null])
                    has_state[keys[non_null]] = True

        group_codes = np.nonzero(present)[0]
        rows = []
        for code in group_codes:
            key_value = self.table._cols[key_name].decode(int(code)) if key_is_dict else _python_scalar(code)
            row = {key_name: key_value}
            for spec in specs:
                state = states[spec.output_col]
                if spec.op == "mean":
                    sums, counts = state
                    row[spec.output_col] = (
                        math.nan if counts[code] == 0 else float(sums[code]) / int(counts[code])
                    )
                elif spec.op in {"min", "max"}:
                    values_state, has_state = state
                    row[spec.output_col] = (
                        _python_scalar(values_state[code])
                        if has_state[code]
                        else _null_output_value(self._result_spec_for_agg(spec))
                    )
                else:
                    row[spec.output_col] = _python_scalar(state[code])
            rows.append(row)
        return self._build_result(rows, specs)

    def _chunk_size(self) -> int:
        if self.chunk_size is not None:
            if self.chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            return int(self.chunk_size)
        chunks = getattr(self.table._valid_rows, "chunks", None)
        if chunks:
            return max(int(chunks[0]), 1)
        return 65536

    def _read_key_chunk(self, name: str, start: int, stop: int) -> np.ndarray:
        col_info = self.table._schema.columns_by_name[name]
        if self.table._is_dictionary_column(col_info):
            return np.asarray(self.table._cols[name].codes[start:stop], dtype=np.int32)
        return np.asarray(self.table._cols[name][start:stop])

    def _factorize_keys(
        self, keys_live: list[np.ndarray]
    ) -> tuple[np.ndarray | list[np.ndarray], np.ndarray]:
        if len(keys_live) == 1:
            unique, inverse = np.unique(keys_live[0], return_inverse=True)
            return unique, inverse

        dtype = [(f"k{i}", arr.dtype) for i, arr in enumerate(keys_live)]
        packed = np.empty(len(keys_live[0]), dtype=dtype)
        for i, arr in enumerate(keys_live):
            packed[f"k{i}"] = arr
        unique, inverse = np.unique(packed, return_inverse=True)
        return unique, inverse

    def _display_keys(self, unique_keys: np.ndarray | list[np.ndarray]) -> list[tuple[Any, ...]]:
        if len(self.keys) == 1:
            name = self.keys[0]
            col_info = self.table._schema.columns_by_name[name]
            values = []
            for value in np.asarray(unique_keys):
                if self.table._is_dictionary_column(col_info):
                    values.append((self.table._cols[name].decode(int(value)),))
                else:
                    values.append((_python_scalar(value),))
            return values

        result = []
        assert isinstance(unique_keys, np.ndarray)
        for row in unique_keys:
            vals = []
            for i, name in enumerate(self.keys):
                value = row[f"k{i}"]
                col_info = self.table._schema.columns_by_name[name]
                if self.table._is_dictionary_column(col_info):
                    vals.append(self.table._cols[name].decode(int(value)))
                else:
                    vals.append(_python_scalar(value))
            result.append(tuple(vals))
        return result

    def _normalized_keys(self, display_keys: list[tuple[Any, ...]]) -> list[Any]:
        normalized = []
        for key in display_keys:
            norm = tuple(_normalize_key_part(v) for v in key)
            normalized.append(norm[0] if len(norm) == 1 else norm)
        return normalized

    def _compute_partials(
        self,
        specs: list[_AggSpec],
        unique_keys: np.ndarray | list[np.ndarray],
        inverse: np.ndarray,
        value_chunks: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        n_groups = len(unique_keys)
        partials: dict[str, Any] = {}
        for spec in specs:
            if spec.op == "size":
                partials[spec.output_col] = np.bincount(inverse, minlength=n_groups).astype(np.int64)
                continue

            assert spec.input_col is not None
            values = value_chunks[spec.input_col]
            non_null = ~self._null_mask(spec.input_col, values, is_key=False)

            if spec.op == "count":
                partials[spec.output_col] = np.bincount(
                    inverse, weights=non_null.astype(np.int64), minlength=n_groups
                ).astype(np.int64)
            elif spec.op in {"sum", "mean"}:
                counts = np.bincount(inverse, weights=non_null.astype(np.int64), minlength=n_groups).astype(
                    np.int64
                )
                if spec.op == "sum" and values.dtype.kind in "biu":
                    sums = np.zeros(n_groups, dtype=np.int64)
                    np.add.at(sums, inverse[non_null], values[non_null].astype(np.int64, copy=False))
                else:
                    weights = np.where(non_null, values, 0)
                    sums = np.bincount(inverse, weights=weights, minlength=n_groups)
                partials[spec.output_col] = (sums, counts)
            elif spec.op in {"min", "max"}:
                partials[spec.output_col] = self._minmax_partials(
                    spec.op, inverse, values, non_null, n_groups
                )
        return partials

    def _minmax_partials(
        self, op: AggName, inverse: np.ndarray, values: np.ndarray, non_null: np.ndarray, n_groups: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if values.dtype.kind in "biufcmM":
            if op == "min":
                identity = _max_identity(values.dtype)
                out = np.full(n_groups, identity, dtype=values.dtype)
                np.minimum.at(out, inverse[non_null], values[non_null])
            else:
                identity = _min_identity(values.dtype)
                out = np.full(n_groups, identity, dtype=values.dtype)
                np.maximum.at(out, inverse[non_null], values[non_null])
        else:
            out = np.empty(n_groups, dtype=values.dtype)
            has = np.zeros(n_groups, dtype=bool)
            for group, value, ok in zip(inverse, values, non_null, strict=True):
                if not ok:
                    continue
                if not has[group] or (value < out[group] if op == "min" else value > out[group]):
                    out[group] = value
                    has[group] = True
            return out, has
        has_value = np.bincount(inverse, weights=non_null.astype(np.int64), minlength=n_groups) > 0
        return out, has_value

    def _merge_partials(
        self,
        acc: dict[Any, dict[str, _AggState]],
        key_values: dict[Any, tuple[Any, ...]],
        normalized_keys: list[Any],
        display_keys: list[tuple[Any, ...]],
        partials: dict[str, Any],
        specs: list[_AggSpec],
    ) -> None:
        for i, norm_key in enumerate(normalized_keys):
            states = acc.setdefault(norm_key, {})
            key_values.setdefault(norm_key, display_keys[i])
            for spec in specs:
                state = states.setdefault(spec.output_col, _AggState(spec.op))
                partial = partials[spec.output_col]
                if spec.op in {"size", "count"}:
                    state.value = (0 if state.value is None else state.value) + int(partial[i])
                elif spec.op == "sum":
                    sums, counts = partial
                    if counts[i] > 0:
                        state.value = (0 if state.value is None else state.value) + _python_scalar(sums[i])
                        state.count += int(counts[i])
                elif spec.op == "mean":
                    sums, counts = partial
                    if counts[i] > 0:
                        state.value = (0.0 if state.value is None else state.value) + float(sums[i])
                        state.count += int(counts[i])
                elif spec.op in {"min", "max"}:
                    values, has_value = partial
                    if has_value[i]:
                        value = _python_scalar(values[i])
                        if (
                            state.count == 0
                            or (spec.op == "min" and value < state.value)
                            or (spec.op == "max" and value > state.value)
                        ):
                            state.value = value
                        state.count += 1

    def _final_rows(
        self,
        acc: dict[Any, dict[str, _AggState]],
        key_values: dict[Any, tuple[Any, ...]],
        specs: list[_AggSpec],
    ) -> list[dict[str, Any]]:
        keys = list(acc)
        if self.sort:
            keys.sort(key=lambda k: tuple(_sortable_key_part(v) for v in key_values[k]))

        rows = []
        for norm_key in keys:
            row = dict(zip(self.keys, key_values[norm_key], strict=True))
            states = acc[norm_key]
            for spec in specs:
                state = states[spec.output_col]
                if spec.op == "mean":
                    row[spec.output_col] = math.nan if state.count == 0 else state.value / state.count
                elif spec.op in {"sum", "min", "max"} and state.count == 0:
                    row[spec.output_col] = _null_output_value(self._result_spec_for_agg(spec))
                else:
                    row[spec.output_col] = 0 if state.value is None else state.value
            rows.append(row)
        return rows

    def _build_result(self, rows: list[dict[str, Any]], specs: list[_AggSpec]):
        from blosc2.ctable import CTable

        columns = self.keys + [spec.output_col for spec in specs]
        schema_specs = {name: self._result_spec_for_key(name) for name in self.keys}
        for spec in specs:
            schema_specs[spec.output_col] = self._result_spec_for_agg(spec)

        fields = []
        for name in columns:
            fields.append((name, _python_type_for_spec(schema_specs[name]), b2_field(schema_specs[name])))
        row_type = dataclasses.make_dataclass("CTableGroupByRow", fields)
        data = {name: [row[name] for row in rows] for name in columns}
        return CTable(row_type, new_data=data, expected_size=max(len(rows), 1), validate=False)

    def _validate_output_names(self, specs: list[_AggSpec]) -> None:
        names = self.keys + [s.output_col for s in specs]
        bad = [name for name in names if not _IDENTIFIER_RE.match(name)]
        if bad:
            raise NotImplementedError(
                "Phase-1 group_by() result columns must be valid Python identifiers; "
                f"unsupported names: {bad!r}"
            )
        if len(names) != len(set(names)):
            raise ValueError("Group-by result column names would not be unique")

    def _result_spec_for_key(self, name: str) -> SchemaSpec:
        return copy.deepcopy(self.table._schema.columns_by_name[name].spec)

    def _result_spec_for_agg(self, spec: _AggSpec) -> SchemaSpec:
        if spec.op in {"size", "count"}:
            return int64()
        if spec.op == "mean":
            return float64()
        assert spec.input_col is not None
        input_spec = self.table._schema.columns_by_name[spec.input_col].spec
        dtype = getattr(input_spec, "dtype", None)
        if spec.op == "sum":
            if dtype is not None and dtype.kind in "iu":
                return int64()
            if dtype is not None and dtype.kind == "b":
                return int64()
            if dtype is not None and dtype.kind == "f":
                return float64()
        return copy.deepcopy(input_spec)

    def _null_mask(self, name: str, values: np.ndarray, *, is_key: bool) -> np.ndarray:
        col_info = self.table._schema.columns_by_name[name]
        spec = col_info.spec
        if isinstance(spec, DictionarySpec):
            mask = values == np.int32(spec.null_code)
            return mask if is_key or getattr(spec, "nullable", False) else np.zeros(len(values), dtype=bool)
        null_value = getattr(spec, "null_value", None)
        mask = np.zeros(len(values), dtype=bool)
        # For keys, treat all NaNs as missing so dropna behaves predictably.
        # For values, only nullable NaN sentinels are skipped.
        if values.dtype.kind == "f" and (
            is_key or (isinstance(null_value, float) and math.isnan(null_value))
        ):
            mask |= np.isnan(values)
        if null_value is not None and not (isinstance(null_value, float) and math.isnan(null_value)):
            mask |= values == null_value
        return mask


def _normalize_key_part(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return _NAN_KEY
    return value


def _sortable_key_part(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, float) and math.isnan(value):
        return (0, "")
    return (1, value)


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _python_type_for_spec(spec: SchemaSpec):
    if isinstance(spec, DictionarySpec):
        return str
    if isinstance(spec, b2_bool):
        return bool
    dtype = getattr(spec, "dtype", None)
    if dtype is not None:
        if dtype.kind in "iu":
            return int
        if dtype.kind == "f":
            return float
        if dtype.kind == "b":
            return bool
        if dtype.kind in "US":
            return str if dtype.kind == "U" else bytes
    return getattr(spec, "python_type", object)


def _max_identity(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        return np.iinfo(dtype).max
    if dtype.kind == "f":
        return np.inf
    if dtype.kind in "mM":
        return np.iinfo(np.int64).max
    return None


def _min_identity(dtype: np.dtype):
    dtype = np.dtype(dtype)
    if dtype.kind in "iu":
        return np.iinfo(dtype).min
    if dtype.kind == "f":
        return -np.inf
    if dtype.kind in "mM":
        return np.iinfo(np.int64).min
    return None


def _null_output_value(spec: SchemaSpec):
    dtype = getattr(spec, "dtype", None)
    null_value = getattr(spec, "null_value", None)
    if null_value is not None:
        return null_value
    if dtype is not None and dtype.kind == "f":
        return math.nan
    if dtype is not None and dtype.kind in "iu":
        return 0
    if dtype is not None and dtype.kind == "b":
        return False
    if dtype is not None and dtype.kind == "U":
        return ""
    if dtype is not None and dtype.kind == "S":
        return b""
    return None
