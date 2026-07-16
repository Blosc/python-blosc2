#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for the unbound column expression (col()) and CTable.assign()."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import col
from blosc2.ctable import ColExpr


@dataclass
class Ledger:
    revenue: float = blosc2.field(blosc2.float64())
    cost: float = blosc2.field(blosc2.float64())


@dataclass
class Nullable:
    x: float = blosc2.field(blosc2.float64(null_value=float("nan")))


def _make_ledger(n: int = 5) -> blosc2.CTable:
    data = [(float(10 * (i + 1)), float(i + 1)) for i in range(n)]
    return blosc2.CTable(Ledger, data)


def test_assign_basic():
    t = _make_ledger()
    t2 = t.assign(profit=col("revenue") - col("cost"))
    assert t2.col_names == ["revenue", "cost", "profit"]
    np.testing.assert_allclose(t2.profit[:], t.revenue[:] - t.cost[:])
    # original table is untouched
    assert t.col_names == ["revenue", "cost"]
    assert "profit" not in t._computed_cols


def test_assign_chain_end_to_end():
    t = _make_ledger()
    result = (
        t.assign(profit=col("revenue") - col("cost"))[col("profit") > 0]
        .sort_by("profit", ascending=False)
        .head(10)
    )
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"revenue": t.revenue[:], "cost": t.cost[:]})
    expected = (
        df.assign(profit=df.revenue - df.cost)
        .query("profit > 0")
        .sort_values("profit", ascending=False)
        .head(10)
    )
    np.testing.assert_allclose(result.profit[:], expected["profit"].to_numpy())


def test_colexpr_filter_matches_bound_form():
    t = _make_ledger()
    view_colexpr = t[col("revenue") > 25]
    view_bound = t[t.revenue > 25]
    np.testing.assert_array_equal(view_colexpr.revenue[:], view_bound.revenue[:])


def test_assign_null_propagation():
    t = blosc2.CTable(Nullable, [(1.0,), (float("nan"),), (3.0,)])
    t2 = t.assign(y=col("x") + 1)
    expected = (t.x + 1)[: t.nrows]  # bound form, already tested elsewhere
    np.testing.assert_array_equal(np.isnan(t2.y[:]), np.isnan(expected))
    np.testing.assert_allclose(t2.y[:][~np.isnan(t2.y[:])], expected[~np.isnan(expected)])

    filtered_colexpr = t[col("x") < 0]
    filtered_bound = t[t.x < 0]
    assert filtered_colexpr.nrows == filtered_bound.nrows == 0


def test_assign_reflected_scalar():
    t = _make_ledger()
    t2 = t.assign(y=100 - col("revenue"))
    np.testing.assert_allclose(t2.y[:], 100 - t.revenue[:])


def test_col_unknown_name_fails_at_bind_time():
    expr = col("nope")  # construction does not raise
    assert isinstance(expr, ColExpr)
    t = _make_ledger()
    with pytest.raises(ValueError):
        t.assign(z=expr)


def test_col_method_call_raises_clear_error():
    with pytest.raises(AttributeError, match="not supported on an unbound column expression"):
        col("x").sum()


def test_colexpr_reused_across_tables():
    t1 = _make_ledger(3)
    t2 = _make_ledger(4)
    expr = col("revenue") + 1
    r1 = t1.assign(y=expr)
    r2 = t2.assign(y=expr)
    np.testing.assert_allclose(r1.y[:], t1.revenue[:] + 1)
    np.testing.assert_allclose(r2.y[:], t2.revenue[:] + 1)


def test_assign_on_view():
    t = _make_ledger()
    view = t[t.revenue > 20]
    assigned = view.assign(z=col("revenue") * 2)
    np.testing.assert_allclose(assigned.z[:], view.revenue[:] * 2)


def test_assign_result_is_read_only_view():
    t = _make_ledger()
    t2 = t.assign(profit=col("revenue") - col("cost"))
    assert t2._cols is t._cols
    with pytest.raises(ValueError):
        t2["revenue"][:] = np.zeros(t.nrows)


def test_assign_duplicate_name_raises():
    t = _make_ledger()
    with pytest.raises(ValueError):
        t.assign(revenue=col("cost"))
