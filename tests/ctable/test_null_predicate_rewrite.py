#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""String predicates over nullable columns follow SQL ``WHERE`` semantics.

A null operand satisfies no comparison.  The operator form has always done
this (``Column._null_aware_compare``); the string form gets there through
``CTable._rewrite_null_predicates``, which conjoins a validity guard onto each
comparison leaf that reads a nullable column.

Per *leaf*, not once over the whole expression: a global ``result & valid_a``
would drop a row that is null in ``a`` but matches the other branch of
``(a > 10) | (b == 0)``, which SQL says qualifies.

Under a negation a guard is not enough in either position, and the rewrite
switches to three-valued logic there; ``tests/ctable/test_kleene_logic.py``
covers that side, and the two negation tests here pin the string form against
the operator one.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2 import CTable
from blosc2.ctable_nulls import rewrite_null_predicates, sentinel_guard_expr

# ---------------------------------------------------------------------------
# The pure string -> string transform
# ---------------------------------------------------------------------------

# name -> (guard text, sentinel value).  999 satisfies "> 10", so every leaf
# below genuinely needs its guard.
GUARDS = {"a": ("(a != 999)", 999), "trip.lon": ("(trip.lon != 999)", 999)}


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("a > 10", "(a > 10) & (a != 999)"),
        # Per leaf, so the OR branch that does not read `a` stays reachable.
        ("(a > 10) | (b == 0)", "(a > 10) & (a != 999) | (b == 0)"),
        ("a + b > 10", "(a + b > 10) & (a != 999)"),
        # Comparing against the sentinel itself must match nothing.
        ("a == 999", "(a == 999) & (a != 999)"),
        # Nested leaves keep their dotted name; the nested rewrite runs later.
        ("trip.lon > 10", "(trip.lon > 10) & (trip.lon != 999)"),
        # and/or are normalized to &/| so one precedence rule governs the result.
        ("a > 10 and b < 3", "(a > 10) & (a != 999) & (b < 3)"),
        ("a > 10 or b < 3", "(a > 10) & (a != 999) | (b < 3)"),
    ],
)
def test_rewrite_shapes(expr, expected):
    assert rewrite_null_predicates(expr, GUARDS) == expected


def test_rewrite_guards_outside_a_negation():
    """``not (a > 10)`` must be False for a null ``a``, not True.

    Guarding inside -- ``~((a > 10) & valid)`` -- would yield True.
    """
    out = rewrite_null_predicates("~(a > 10)", GUARDS)
    assert out == "~(a > 10) & (a != 999)"
    out = rewrite_null_predicates("not (a > 10)", GUARDS)
    assert out == "(not a > 10) & (a != 999)"


def test_rewrite_skips_leaves_the_sentinel_cannot_satisfy():
    """A guard that cannot change the answer is not emitted.

    Beyond tidiness: an unguarded single-predicate expression is the shape the
    index planner recognizes, so a needless guard would push a query that is
    correct today off its index onto a full scan.
    """
    guards = {"a": ("(a != -1)", -1)}
    assert rewrite_null_predicates("a > 10", guards) is None
    assert rewrite_null_predicates("a < 10", guards) == "(a < 10) & (a != -1)"
    # ...but never inside a negation: a sentinel that fails `a > 10` passes
    # `not (a > 10)`.
    assert rewrite_null_predicates("~(a > 10)", guards) == "~(a > 10) & (a != -1)"


def test_rewrite_nan_sentinel_only_guards_not_equal():
    """Every comparison with NaN is False -- except ``!=``, which is True."""
    guards = {"a": ("(a == a)", float("nan"))}
    assert rewrite_null_predicates("a > 10", guards) is None
    assert rewrite_null_predicates("a == 10", guards) is None
    assert rewrite_null_predicates("a != 10", guards) == "(a != 10) & (a == a)"


def test_rewrite_returns_none_when_nothing_applies():
    assert rewrite_null_predicates("b > 0", GUARDS) is None
    assert rewrite_null_predicates("a > 10", {}) is None
    assert rewrite_null_predicates("a > > 10", GUARDS) is None  # unparseable


@pytest.mark.parametrize(
    ("null_value", "expected"),
    [
        (-1, "(x != -1)"),
        (255, "(x != 255)"),
        (0.5, "(x != 0.5)"),
        ("", "(x != '')"),
        (b"", "(x != b'')"),
        (np.int64(-1), "(x != -1)"),
    ],
)
def test_sentinel_guard_expr(null_value, expected):
    assert sentinel_guard_expr("x", null_value) == expected


def test_sentinel_guard_expr_nan_compares_to_itself():
    """NaN is the only value unequal to itself, so this needs no isnan()."""
    assert sentinel_guard_expr("x", float("nan")) == "(x == x)"


# ---------------------------------------------------------------------------
# End-to-end SQL semantics
# ---------------------------------------------------------------------------


def _table(spec, annotation, values, sentinel):
    Row = dataclasses.make_dataclass(
        "Row",
        [("a", annotation, blosc2.field(spec)), ("b", int, blosc2.field(blosc2.int64()))],
    )
    t = CTable(Row)
    t.extend({"a": values, "b": np.arange(len(values))})
    return t


# Each case: sentinel, the four `a` values (index 2 is the null), and a probe
# value.  The int-999 case is the one that is wrong before this change: the
# sentinel satisfies `a > 10`, so the null row leaked into the result.
CASES = [
    ("int-sentinel-below", blosc2.int64(null_value=-1), int, [1, 20, -1, 30], -1),
    ("int-sentinel-above", blosc2.int64(null_value=999), int, [1, 20, 999, 30], 999),
    ("nan", blosc2.float64(null_value=float("nan")), float, [1.0, 20.0, np.nan, 30.0], np.nan),
    ("string", blosc2.string(max_length=8, null_value=""), str, ["aa", "zz", "", "mm"], ""),
]


@pytest.mark.parametrize(("label", "spec", "annotation", "values", "sentinel"), CASES)
def test_string_predicate_matches_sql(label, spec, annotation, values, sentinel):
    t = _table(spec, annotation, values, sentinel)
    arr = np.array(values)
    valid = np.isnan(arr) == False if label == "nan" else arr != sentinel  # noqa: E712
    b = np.arange(len(values))
    threshold = "'mm'" if label == "string" else "10"
    cmp = arr > (np.array("mm") if label == "string" else 10)

    got = sorted(t.where(f"a > {threshold}")["b"][:].tolist())
    assert got == sorted(b[cmp & valid].tolist())

    # OR: the null row must still qualify through the other branch.
    got = sorted(t.where(f"(a > {threshold}) | (b == 2)")["b"][:].tolist())
    assert got == sorted(b[(cmp & valid) | (b == 2)].tolist())

    # Negation: a null operand makes `not (...)` False, not True.
    got = sorted(t.where(f"~(a > {threshold})")["b"][:].tolist())
    assert got == sorted(b[~cmp & valid].tolist())


@pytest.mark.parametrize(("label", "spec", "annotation", "values", "sentinel"), CASES)
def test_string_form_agrees_with_operator_form(label, spec, annotation, values, sentinel):
    """The two ways of spelling a predicate must return the same rows."""
    t = _table(spec, annotation, values, sentinel)
    threshold = "mm" if label == "string" else 10
    quoted = f"'{threshold}'" if label == "string" else threshold

    assert sorted(t.where(f"a > {quoted}")["b"][:].tolist()) == sorted(
        t.where(t["a"] > threshold)["b"][:].tolist()
    )
    assert sorted(t.where(f"(a > {quoted}) | (b == 2)")["b"][:].tolist()) == sorted(
        t.where((t["a"] > threshold) | (t["b"] == 2))["b"][:].tolist()
    )


def test_negation_over_and_corner_agrees_with_sql():
    """The corner the two query forms used to disagree on -- now both exact.

    ``~((a > 10) & (b == 999))`` with ``a`` null and the second term False:
    SQL collapses ``NULL AND FALSE`` to ``FALSE``, so the negation is True and
    the row qualifies.  Getting that right needs three values, because it
    hinges on ``unknown & false`` being *false* rather than unknown -- no
    amount of conjoining a validity guard onto the negation can express it
    (the string form dropped the row until Kleene logic landed).
    """
    t = _table(blosc2.int64(null_value=-1), int, [1, 20, -1, 30], -1)

    assert sorted(t.where("~((a > 10) & (b == 999))")["b"][:].tolist()) == [0, 1, 2, 3]
    assert sorted(t.where(~((t["a"] > 10) & (t["b"] == 999)))["b"][:].tolist()) == [0, 1, 2, 3]


def test_operator_form_negation_drops_nulls():
    """``~(a > 10)`` is unknown for a null ``a``, so the row is not a match.

    Was a strict xfail: the comparison collapsed the null to False at the leaf
    and ``~`` inverted it into a match.  Fixed at the source -- the comparison
    now carries its null predicate through ``~`` (:class:`NullableBoolExpr`).
    """
    t = _table(blosc2.int64(null_value=-1), int, [1, 20, -1, 30], -1)
    assert sorted(t.where(~(t["a"] > 10))["b"][:].tolist()) == [0]
    assert sorted(t.where("~(a > 10)")["b"][:].tolist()) == [0]


def test_comparing_against_the_sentinel_matches_nothing():
    """The sentinel is not a value: SQL has no row whose `a` equals NULL."""
    t = _table(blosc2.int64(null_value=-1), int, [1, 20, -1, 30], -1)
    assert t.where("a == -1").nrows == 0
    assert sorted(t.where("a != -1")["b"][:].tolist()) == [0, 1, 3]


def test_non_nullable_column_is_untouched():
    t = _table(blosc2.int64(null_value=-1), int, [1, 20, -1, 30], -1)
    assert sorted(t.where("b > 1")["b"][:].tolist()) == [2, 3]


def test_dropna_then_predicate_is_consistent():
    t = _table(blosc2.int64(null_value=999), int, [1, 20, 999, 30], 999)
    assert sorted(t.where("a > 10")["b"][:].tolist()) == sorted(t.dropna().where("a > 10")["b"][:].tolist())


# ---------------------------------------------------------------------------
# Index and scan must agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sentinel", [-1.0, 999.0, float("nan")])
@pytest.mark.parametrize(
    "expr", ["a > 90", "a != 50", "(a > 90) | (b < 10)", "(a > 90) & (b < 5000)", "~(a > 90)"]
)
def test_indexed_matches_unindexed(sentinel, expr):
    """An ordered index answers by sorted range and never evaluates the
    predicate -- a NaN sentinel sorts last, so it lands inside every ``>``
    range.  Making the expression null-aware cannot fix that, so the index path
    keeps its own null exclusion; this pins the two paths together.
    """
    Row = dataclasses.make_dataclass(
        "Row",
        [
            ("a", float, blosc2.field(blosc2.float64(null_value=sentinel))),
            ("b", int, blosc2.field(blosc2.int64())),
        ],
    )
    n = 20_000
    rng = np.random.default_rng(0)
    a = rng.integers(0, 100, n).astype(np.float64)
    a[::13] = sentinel
    payload = {"a": a, "b": np.arange(n)}

    plain = CTable(Row)
    plain.extend(payload)
    indexed = CTable(Row)
    indexed.extend(payload)
    indexed.create_index("a", kind=blosc2.IndexKind.FULL)

    expected = np.sort(plain.where(expr)["b"][:])
    assert np.array_equal(np.sort(indexed.where(expr)["b"][:]), expected)

    # ...and both agree with SQL.
    valid = ~np.isnan(a) if np.isnan(sentinel) else a != sentinel
    b = np.arange(n)
    sql = {
        "a > 90": (a > 90) & valid,
        "a != 50": (a != 50) & valid,
        "(a > 90) | (b < 10)": ((a > 90) & valid) | (b < 10),
        "(a > 90) & (b < 5000)": (a > 90) & valid & (b < 5000),
        "~(a > 90)": ~(a > 90) & valid,
    }[expr]
    assert np.array_equal(expected, np.sort(b[sql]))
