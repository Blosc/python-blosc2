#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for sentinel-based null propagation through Column expressions
(arithmetic and comparisons), per Gap C2/C2b of plans/enhancing-ctable.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

NULL_I64 = np.iinfo(np.int64).min


@dataclass
class IntRow:
    id: int = blosc2.field(blosc2.int64())
    score: int = blosc2.field(blosc2.int64(null_value=NULL_I64))
    other: int = blosc2.field(blosc2.int64(null_value=NULL_I64))


@dataclass
class TsRow:
    id: int = blosc2.field(blosc2.int64())
    ts: int = blosc2.field(blosc2.timestamp(null_value=NULL_I64))


# ===========================================================================
# Comparisons: nulls never satisfy any comparison (SQL WHERE semantics)
# ===========================================================================


def test_lt_excludes_null_rows():
    """Headline bug fix: before C2b, INT64_MIN < 0 was True, wrongly
    including the null row in a less-than filter."""
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0), (4, 30, 0)])
    assert t[t.score < 0]["id"][:].tolist() == [3]


def test_gt_excludes_null_rows():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0), (4, 30, 0)])
    assert t[t.score > 0]["id"][:].tolist() == [1, 4]


def test_eq_sentinel_literal_does_not_match_null():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0)])
    assert t[t.score == NULL_I64]["id"][:].tolist() == []


def test_ne_sentinel_literal_does_not_match_null_either():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0)])
    # A null never satisfies `!=` either — it isn't "not equal", it's unknown.
    assert t[t.score != NULL_I64]["id"][:].tolist() == [1]


def test_is_null_still_finds_nulls():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0)])
    assert list(t.score.is_null()) == [False, True]


def test_comparison_between_two_nullable_columns_excludes_either_null():
    t = CTable(
        IntRow,
        new_data=[
            (1, 10, 5),  # 10 > 5 -> True
            (2, NULL_I64, 5),  # score null -> excluded
            (3, 10, NULL_I64),  # other null -> excluded
            (4, 3, 5),  # 3 > 5 -> False
        ],
    )
    assert t[t.score > t.other]["id"][:].tolist() == [1]


def test_ge_le_also_exclude_nulls():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0)])
    assert t[t.score >= -20]["id"][:].tolist() == [1, 3]
    assert t[t.score <= -20]["id"][:].tolist() == [3]


# ===========================================================================
# Arithmetic: null propagates, output promoted to float64/NaN
# ===========================================================================


def test_arith_propagates_null_for_nullable_int():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0)])
    result = (t.score + 1).compute()[:3]
    assert not np.isnan(result[0])
    assert np.isnan(result[1])
    assert not np.isnan(result[2])
    np.testing.assert_array_equal(result[[0, 2]], [11.0, -19.0])


def test_arith_propagates_null_for_timestamp_column():
    t = CTable(TsRow, new_data=[(1, 1000), (2, NULL_I64), (3, 2000)])
    result = (t.ts + 1).compute()[:3]
    assert np.isnan(result[1])
    np.testing.assert_array_equal(result[[0, 2]], [1001.0, 2001.0])


def test_arith_scalar_operand():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0)])
    result = (t.score * 3).compute()[:2]
    assert result[0] == 30.0
    assert np.isnan(result[1])


def test_arith_mixed_nullable_and_non_nullable_column():
    """`id` has no null_value; `score` does. Nullness still propagates."""
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0)])
    result = (t.id + t.score).compute()[:2]
    assert result[0] == 11.0
    assert np.isnan(result[1])


def test_chained_arithmetic_does_not_double_wrap():
    """The rewrite applies once, at the first nullable-operand boundary;
    NaN then propagates through ordinary float arithmetic downstream."""
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0)])
    result = ((t.score + 1) * 2).compute()[:3]
    assert np.isnan(result[1])
    np.testing.assert_array_equal(result[[0, 2]], [22.0, -38.0])


def test_non_nullable_column_arithmetic_unchanged():
    """Zero-overhead guarantee: a non-nullable column's arithmetic result is
    the raw expression, not routed through a null-check rewrite."""
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, 20, 0)])
    result = (t.id + 1).compute()[:2]
    np.testing.assert_array_equal(result, [2, 3])
    assert result.dtype == np.int64  # no float promotion when nothing is nullable


def test_non_nullable_column_comparison_unchanged():
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, 20, 0)])
    assert t[t.id > 1]["id"][:].tolist() == [2]


# ===========================================================================
# Reductions on a *derived* (unregistered) expression: documented limitation.
# ===========================================================================


def test_reduction_on_derived_expression_is_nan_poisoned_not_null_skipping():
    """`t.score.sum()` (a real Column) skips nulls; `(t.score + 1)` is a
    plain LazyExpr with no memory of nullability, so its own `.sum()` follows
    ordinary NumPy semantics (NaN poisons the reduction) rather than pandas'
    skip-null default. Materialize and mask with NumPy for a skip-null total.
    """
    t = CTable(IntRow, new_data=[(1, 10, 0), (2, NULL_I64, 0), (3, -20, 0)])
    assert t.score.sum() == -10  # Column.sum() already skips nulls today
    assert np.isnan((t.score + 1).sum())  # derived expression: NaN poisons it
    # Workaround: mask with notnull() on materialized values, then reduce with NumPy.
    values = t.score[:]
    total = (values[t.score.notnull()] + 1).sum()
    assert total == pytest.approx(-8.0)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
