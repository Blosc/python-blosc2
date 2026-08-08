#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Expressions and reductions over mask-storage columns (Phase 5).

The expression layer was already storage-agnostic — it consumes an opaque
boolean null predicate — so most of this is a differential check that mask and
sentinel columns agree wherever they are supposed to, and diverge only where
decision 6 says they must (NaN is a value under a mask, a null under a
sentinel).

The reductions needed real work: ``argmin``/``argmax`` and the ndarray
reduction path both keyed off ``null_value is not None``, so under mask storage
they silently reduced over the *fill*.  For an int column the fill is ``0``,
which is a plausible-looking minimum — the kind of wrong answer nobody notices.

Two things this file pins as **deliberately not done**, because measurement
showed the plan's premises for them were wrong:

* the summary-index ``min``/``max`` shortcut stays disabled for mask columns;
* ndarray columns still get no null predicate in the lazy layer.

Both are explained where they are asserted.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2


def annotation_for(spec):
    if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp)):
        return object
    return spec.python_type


def one_col(spec, values, capacity=32):
    Row = dataclasses.make_dataclass("R", [("v", annotation_for(spec), blosc2.field(spec))])
    t = blosc2.CTable(Row, expected_size=capacity)
    t.extend([(v,) for v in values])
    return t


def mask_col(values, factory=blosc2.int64, **kw):
    return one_col(factory(null_storage="mask", **kw), values)


# ---------------------------------------------------------------------------
# Reductions skip nulls rather than reducing over the fill
# ---------------------------------------------------------------------------


def test_min_ignores_the_zero_fill():
    """The int fill is 0 — a plausible-looking minimum for positive data."""
    assert mask_col([5, None, 1, 9])["v"].min() == 1


def test_max_ignores_the_zero_fill():
    """...and a plausible-looking maximum for negative data."""
    assert mask_col([-5, None, -1])["v"].max() == -1


def test_argmin_points_at_a_real_row():
    assert mask_col([5, None, 1, 9])["v"].argmin() == 2


def test_argmax_points_at_a_real_row():
    assert mask_col([-5, None, -1])["v"].argmax() == 2


def test_argmin_would_have_picked_the_fill():
    """Regression: the fill sits at row 1 and is smaller than every real value."""
    col = mask_col([5, None, 1, 9])["v"]
    assert col[:][1] == 0  # the fill really is there in the values
    assert col.argmin() == 2  # ...and is not what argmin reports


@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_arg_reductions_raise_on_an_all_null_column(op):
    col = mask_col([None, None])["v"]
    with pytest.raises(ValueError, match="all values are null"):
        getattr(col, op)()


def test_sum_and_mean_skip_nulls():
    col = mask_col([2.0, None, 4.0], factory=blosc2.float64)["v"]
    assert col.sum() == 6.0
    assert col.mean() == 3.0


def test_std_skips_nulls():
    col = mask_col([1.0, None, 3.0], factory=blosc2.float64)["v"]
    assert col.std() == pytest.approx(1.0)


def test_reductions_ignore_capacity_padding():
    """Padding is fill-valued too, and must not reach a reduction either."""
    col = one_col(blosc2.int64(null_storage="mask"), [5, None, 1], capacity=64)["v"]
    assert col.min() == 1
    assert col.max() == 5
    assert col.sum() == 6


# ---------------------------------------------------------------------------
# ndarray columns: the gain is in the NumPy reduction paths
# ---------------------------------------------------------------------------


def ndarray_col(items, dtype=blosc2.int64):
    spec = blosc2.ndarray((3,), dtype=dtype(), null_storage="mask")
    return one_col(spec, items)["v"]


def test_ndarray_min_skips_null_rows():
    col = ndarray_col([np.array([1, 2, 3]), None, np.array([7, 8, 9])])
    assert col.min() == 1  # not 0, the fill item


def test_ndarray_max_skips_null_rows():
    col = ndarray_col([np.array([-1, -2, -3]), None, np.array([-7, -8, -9])])
    assert col.max() == -1  # not 0, the fill item


def test_ndarray_sum_skips_null_rows():
    col = ndarray_col([np.array([1, 2, 3]), None, np.array([7, 8, 9])])
    assert col.sum() == 30


def test_ndarray_mean_divides_by_live_non_null_elements():
    col = ndarray_col([np.array([1, 2, 3]), None, np.array([7, 8, 9])])
    assert col.mean() == pytest.approx(30 / 6)


def test_ndarray_reduction_where_composes_with_nulls():
    spec = blosc2.ndarray((2,), dtype=blosc2.int64(), null_storage="mask")
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("e", object, blosc2.field(spec)),
            ("k", int, blosc2.field(blosc2.int64())),
        ],
    )
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(np.array([1, 1]), 0), (None, 1), (np.array([5, 5]), 1)])
    assert t["e"].sum(where="k == 1") == 10


# ---------------------------------------------------------------------------
# Propagation through arithmetic and comparisons
# ---------------------------------------------------------------------------


def test_arithmetic_marks_null_rows_nan():
    expr = mask_col([1, None, 3])["v"] * 2
    values = np.asarray(expr[:3])
    assert values[0] == 2.0
    assert np.isnan(values[1])
    assert values[2] == 6.0


def test_arithmetic_result_reduces_without_nan_poisoning():
    assert (mask_col([1, None, 3])["v"] * 2).sum() == 8.0


def test_comparison_gives_null_rows_false():
    """SQL WHERE semantics: a null satisfies no comparison."""
    col = mask_col([1, None, 3])["v"]
    assert (col > 0)[:3].tolist() == [True, False, True]


def test_where_over_a_mask_column_drops_nulls():
    t = mask_col([1, None, 3])
    assert t.where(t["v"] > 0)["v"][:].tolist() == [1, 3]


def test_not_equal_does_not_leak_nulls():
    """IEEE says nan != x is True; SQL says a null satisfies nothing."""
    col = mask_col([1, None, 3])["v"]
    assert (col != 1)[:3].tolist() == [False, False, True]


def test_two_mask_columns_combine_their_nulls():
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("a", int, blosc2.field(blosc2.int64(null_storage="mask"))),
            ("b", int, blosc2.field(blosc2.int64(null_storage="mask"))),
        ],
    )
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(1, 10), (None, 20), (3, None), (4, 40)])
    values = np.asarray((t["a"] + t["b"])[:4])
    assert values[0] == 11
    assert np.isnan(values[1])
    assert np.isnan(values[2])
    assert values[3] == 44


def test_a_null_free_mask_column_costs_no_operand():
    """No sidecar means no predicate, so arithmetic stays a plain LazyExpr."""
    col = mask_col([1, 2, 3])["v"]
    assert col._raw_null_pred() is None
    assert isinstance(col * 2, blosc2.LazyExpr)


# ---------------------------------------------------------------------------
# Differential: mask and sentinel agree, except where decision 6 says not to
# ---------------------------------------------------------------------------

AGREEING_CASES = [
    ("min", lambda c: c.min()),
    ("max", lambda c: c.max()),
    ("sum", lambda c: c.sum()),
    ("mean", lambda c: c.mean()),
    ("argmin", lambda c: c.argmin()),
    ("argmax", lambda c: c.argmax()),
    ("null_count", lambda c: c.null_count()),
    ("is_null", lambda c: c.is_null().tolist()),
    ("unique", lambda c: c.unique().tolist()),
    ("gt", lambda c: (c > 2)[:4].tolist()),
]


@pytest.mark.parametrize(("label", "op"), AGREEING_CASES)
def test_mask_and_sentinel_agree_on_integer_data(label, op):
    """Same logical data, two storages, one answer."""
    values = [5, None, 1, 9]
    masked = mask_col(values)["v"]
    sentinel = one_col(blosc2.int64(null_value=-999), [-999 if v is None else v for v in values])["v"]
    assert op(masked) == op(sentinel)


def test_mask_and_sentinel_diverge_on_nan_by_design():
    """decision 6: NaN is a value under a mask, the null itself under a sentinel."""
    values = [1.0, float("nan"), 3.0]
    masked = one_col(blosc2.float64(null_storage="mask"), values)["v"]
    sentinel = one_col(blosc2.float64(null_value=float("nan")), values)["v"]

    assert masked.null_count() == 0
    assert sentinel.null_count() == 1
    # The NaN is data for the mask column, so it poisons the reduction the way
    # NumPy does; for the sentinel column it is the null and is skipped.
    assert np.isnan(masked.sum())
    assert sentinel.sum() == 4.0


# ---------------------------------------------------------------------------
# Nullable bool needs no predicate rewrite under mask storage
# ---------------------------------------------------------------------------


def test_mask_bool_is_not_treated_as_a_sentinel_bool():
    col = mask_col([True, None, False], factory=blosc2.bool)["v"]
    assert col._is_nullable_bool is False


def test_sentinel_bool_is_still_treated_as_one():
    col = one_col(blosc2.bool(nullable=True, null_value=255), [True, 255, False])["v"]
    assert col._is_nullable_bool is True


def test_mask_bool_filters_directly():
    t = mask_col([True, None, False, True], factory=blosc2.bool)
    assert t.where(t["v"])["v"][:].tolist() == [True, True]


# ---------------------------------------------------------------------------
# Deliberately not done — with the measurement that says why
# ---------------------------------------------------------------------------


def test_summary_minmax_shortcut_stays_disabled_for_mask_columns(tmp_path):
    """Enabling it would make min() answer differently depending on the index.

    A mask float column's fill is NaN, which the summary builder drops — but so
    is a *genuine* NaN, which decision 6 makes a value.  The scan therefore
    poisons to NaN while the summaries would report a real extremum.
    """
    spec = blosc2.float64(null_storage="mask")
    Row = dataclasses.make_dataclass("R", [("v", float, blosc2.field(spec))])
    path = str(tmp_path / "s.b2d")
    t = blosc2.CTable(Row, expected_size=5, urlpath=path, mode="w")
    t.extend([(1.0,), (float("nan"),), (5.0,), (None,), (3.0,)])
    t.close()

    reopened = blosc2.CTable.open(path, mode="a")
    try:
        assert reopened["v"]._summary_minmax_source() is None
        # The scanned answer, which the shortcut would have contradicted.
        assert np.isnan(reopened["v"].min())
    finally:
        reopened.close()


def test_sentinel_nan_float_keeps_its_summary_shortcut(tmp_path):
    """The contrast: there NaN *is* the null, so dropping it is exactly right."""
    spec = blosc2.float64(null_value=float("nan"))
    Row = dataclasses.make_dataclass("R", [("v", float, blosc2.field(spec))])
    path = str(tmp_path / "n.b2d")
    t = blosc2.CTable(Row, expected_size=4, urlpath=path, mode="w")
    t.extend([(1.0,), (float("nan"),), (5.0,), (3.0,)])
    t.close()

    reopened = blosc2.CTable.open(path, mode="a")
    try:
        assert reopened["v"]._summary_minmax_source() is not None
        assert reopened["v"].min() == 1.0
    finally:
        reopened.close()


def test_ndarray_columns_still_get_no_lazy_null_predicate():
    """Not an oversight: see NullChannel._mask_pred for the two blockers."""
    col = ndarray_col([np.array([1, 2, 3]), None])
    assert col._raw_null_pred() is None
    assert col._nulls.valid_pred() is None
    # ...and the reduction paths, which are row-level NumPy, cover it instead.
    assert col.min() == 1
