#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Three-valued (Kleene) logic for predicates over nullable columns.

A comparison against a null is neither true nor false.  Every earlier phase
collapsed that third value to False at the leaf, which is right for ``WHERE``
and wrong for ``~``: negating a collapsed null turns it into a match.  The
comparison now returns a :class:`blosc2.ctable.NullableBoolExpr` carrying the
unknown rows beside the true ones, and ``&``/``|``/``^``/``~`` combine them by
Kleene's rules.

The load-bearing case is ``unknown & false``, which is **false**, not unknown
-- no value of the missing operand could make the conjunction true.  That is
why "drop every row a null touched" is not an implementation of this, and it
is what ``test_and_or_truth_tables`` pins.

Both query forms are covered, because they get there by different routes: the
operator form through the expression objects, the string form through an AST
rewrite that carries two channels under a negation
(``ctable_nulls._NullPredicateRewriter._channels``).  They are checked against
each other and against a NumPy oracle, over both null storages and with and
without indexes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2.ctable import NullableBoolExpr

# ---------------------------------------------------------------------------
# A Kleene oracle, written the way the truth tables read
# ---------------------------------------------------------------------------

TRUE, FALSE, UNKNOWN = True, False, None
CELLS = (TRUE, FALSE, UNKNOWN)


def k_and(x, y):
    if x is FALSE or y is FALSE:
        return FALSE
    return UNKNOWN if UNKNOWN in (x, y) else TRUE


def k_or(x, y):
    if x is TRUE or y is TRUE:
        return TRUE
    return UNKNOWN if UNKNOWN in (x, y) else FALSE


def k_xor(x, y):
    return UNKNOWN if UNKNOWN in (x, y) else (x != y)


def k_not(x):
    return UNKNOWN if x is UNKNOWN else (not x)


def observed(t, pred):
    """The predicate's three-valued result, one cell per live row.

    Read the way a user would: a row is true when ``where()`` keeps it and
    unknown when the predicate says so.  Rows the query dropped without being
    unknown are the false ones.
    """
    n = t.nrows
    unknown = pred.is_null() if isinstance(pred, NullableBoolExpr) else np.zeros(n, dtype=bool)
    true = np.zeros(n, dtype=bool)
    true[np.asarray(t.where(pred)["id"][:], dtype=int)] = True
    return [UNKNOWN if u else bool(v) for v, u in zip(true, unknown, strict=True)]


# ---------------------------------------------------------------------------
# Fixtures: one row per cell of the truth table
# ---------------------------------------------------------------------------

PAIRS = [(x, y) for x in CELLS for y in CELLS]
STORAGES = ("mask", "sentinel")


def nullable(factory, storage, sentinel):
    """A nullable spec plus the value that spells "null" for it.

    A sentinel column has no way to write a null other than its sentinel, so
    the two storages need different cells for the same logical row -- which is
    the whole reason both are parametrized here rather than assumed alike.
    """
    if storage == "mask":
        return factory(nullable=True, null_storage="mask"), None
    return factory(nullable=True, null_value=sentinel), sentinel


def bool_pair_table(storage):
    """Two nullable bool columns holding every ``(x, y)`` combination."""
    spec, null_cell = nullable(blosc2.bool, storage, 255)
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("p", bool, blosc2.field(spec)),
            ("q", bool, blosc2.field(spec)),
        ],
    )
    cell = {TRUE: True, FALSE: False, UNKNOWN: null_cell}
    return blosc2.CTable(Row, new_data=[(i, cell[x], cell[y]) for i, (x, y) in enumerate(PAIRS)])


def int_pair_table(storage):
    """The same combinations as ``a == 1`` / ``b == 1`` over nullable ints."""
    spec, null_cell = nullable(blosc2.int64, storage, -9)
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("a", int, blosc2.field(spec)),
            ("b", int, blosc2.field(spec)),
        ],
    )
    cell = {TRUE: 1, FALSE: 0, UNKNOWN: null_cell}
    return blosc2.CTable(Row, new_data=[(i, cell[x], cell[y]) for i, (x, y) in enumerate(PAIRS)])


# ---------------------------------------------------------------------------
# The truth tables themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize(
    ("op", "expected"),
    [("&", k_and), ("|", k_or), ("^", k_xor)],
)
def test_and_or_truth_tables(storage, op, expected):
    """All nine cells, for both ways of spelling a three-valued operand.

    ``unknown & false`` is the cell that matters: it must come out **false**.
    A row is dropped by ``where()`` either way, so only the unknown channel
    tells the two apart -- which is exactly what makes an enclosing ``~``
    return the right answer.
    """
    t = int_pair_table(storage)
    a, b = (t.a == 1), (t.b == 1)
    pred = {"&": a & b, "|": a | b, "^": a ^ b}[op]
    assert observed(t, pred) == [expected(x, y) for x, y in PAIRS]


@pytest.mark.parametrize("storage", STORAGES)
def test_not_truth_table(storage):
    t = int_pair_table(storage)
    assert observed(t, ~(t.a == 1)) == [k_not(x) for x, _ in PAIRS]


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("op", ["&", "|", "^"])
def test_nullable_bool_columns_are_three_valued_operands(storage, op):
    """A nullable bool column *is* a three-valued predicate -- no comparison
    needed.  Its third value is in band for sentinel storage (the reserved
    ``255``) and in the sidecar for mask storage; both are read here."""
    t = bool_pair_table(storage)
    pred = {"&": t.p & t.q, "|": t.p | t.q, "^": t.p ^ t.q}[op]
    expected = {"&": k_and, "|": k_or, "^": k_xor}[op]
    assert observed(t, pred) == [expected(x, y) for x, y in PAIRS]


@pytest.mark.parametrize("storage", STORAGES)
def test_negating_a_nullable_bool_column_keeps_its_nulls_out(storage):
    t = bool_pair_table(storage)
    assert observed(t, ~t.p) == [k_not(x) for x, _ in PAIRS]


@pytest.mark.parametrize("storage", STORAGES)
def test_double_negation_is_the_identity(storage):
    """Kleene negation is an involution -- ``~~p`` is ``p``, unknown included.
    Two-valued negation was not: it lost the null on the first ``~``."""
    t = int_pair_table(storage)
    p = t.a == 1
    assert observed(t, ~(~p)) == observed(t, p)


@pytest.mark.parametrize("storage", STORAGES)
def test_de_morgan_holds(storage):
    t = int_pair_table(storage)
    a, b = (t.a == 1), (t.b == 1)
    assert observed(t, ~(a & b)) == observed(t, (~a) | (~b))
    assert observed(t, ~(a | b)) == observed(t, (~a) & (~b))


@pytest.mark.parametrize("storage", STORAGES)
def test_excluded_middle_does_not_hold(storage):
    """``p | ~p`` is *not* a tautology when p is unknown -- the property that
    separates three-valued logic from two-valued.  Pinned so that a future
    "simplification" of the operators cannot quietly restore it."""
    t = int_pair_table(storage)
    p = t.a == 1
    assert observed(t, p | ~p) == [k_or(x, k_not(x)) for x, _ in PAIRS]
    assert UNKNOWN in observed(t, p | ~p)


# ---------------------------------------------------------------------------
# Mixing three-valued and two-valued operands
# ---------------------------------------------------------------------------


def test_a_two_valued_operand_combines_without_a_null_channel():
    """A non-nullable column contributes no unknowns, so the rules degrade to
    the two-valued ones -- and the predicate over it is a plain lazy
    expression, not a three-valued one, so nothing is paid for it."""
    t = int_pair_table("mask")
    plain = t.id < 3
    assert not isinstance(plain, NullableBoolExpr)
    assert isinstance(plain, blosc2.LazyExpr)

    a = t.a == 1
    combined = a & plain
    assert observed(t, combined) == [k_and(x, i < 3) for i, (x, _y) in enumerate(PAIRS)]


def test_operand_order_does_not_change_the_answer():
    """``plain & three_valued`` must not silently collapse to two values.

    Python offers the reflected operator to a subclass first, which is why
    :class:`NullableBoolExpr` subclasses :class:`blosc2.LazyExpr`; a
    :class:`blosc2.NDArray` on the left defers explicitly instead
    (``ndarray._defers_boolean_op``).
    """
    t = int_pair_table("mask")
    a = t.a == 1
    plain = t.id < 3
    assert isinstance(plain & a, NullableBoolExpr)
    assert observed(t, plain & a) == observed(t, a & plain)
    assert observed(t, plain | a) == observed(t, a | plain)

    materialized = blosc2.asarray(np.asarray(plain[:]))
    assert isinstance(materialized & a, NullableBoolExpr)
    assert observed(t, materialized & a) == observed(t, a & plain)


def test_a_column_on_the_right_of_a_lazy_operand_stays_three_valued():
    t = bool_pair_table("mask")
    plain = t.id < 3
    assert isinstance(plain & t.p, NullableBoolExpr)
    assert observed(t, plain & t.p) == [k_and(i < 3, x) for i, (x, _y) in enumerate(PAIRS)]


# ---------------------------------------------------------------------------
# Reading the third value
# ---------------------------------------------------------------------------


def test_is_null_notnull_and_null_count():
    t = int_pair_table("mask")
    p = t.a == 1
    unknown = np.array([x is UNKNOWN for x, _ in PAIRS])
    np.testing.assert_array_equal(p.is_null(), unknown)
    np.testing.assert_array_equal(p.notnull(), ~unknown)
    assert p.null_count() == int(unknown.sum())


def test_fillna_resolves_the_unknown_rows():
    """The two readings of an unknown row, both spelled explicitly.

    ``where()`` takes ``fillna(False)`` implicitly, which is SQL; the other
    direction is "keep what I cannot rule out"."""
    t = int_pair_table("mask")
    p = ~(t.a == 1)
    kept_false = sorted(t.where(p.fillna(False))["id"][:].tolist())
    kept_true = sorted(t.where(p.fillna(True))["id"][:].tolist())
    assert kept_false == [i for i, (x, _) in enumerate(PAIRS) if k_not(x) is TRUE]
    assert kept_true == [i for i, (x, _) in enumerate(PAIRS) if k_not(x) is not FALSE]
    assert not isinstance(p.fillna(True), NullableBoolExpr)


def test_the_predicate_is_a_lazy_expression():
    """It does not merely quack like one: consumers type-check, index with and
    plan around this object, and it has to keep working for all of them."""
    t = int_pair_table("mask")
    p = t.a == 1
    assert isinstance(p, blosc2.LazyExpr)
    assert p.dtype == np.bool_
    computed = np.asarray(p.compute()[:])
    live = np.asarray(p[:])[: t.nrows]
    np.testing.assert_array_equal(live, computed[: t.nrows])
    # The values are the collapsed two-valued predicate: unknown reads False.
    np.testing.assert_array_equal(live, [x is TRUE for x, _ in PAIRS])


# ---------------------------------------------------------------------------
# Every V1 kind negates the same way
# ---------------------------------------------------------------------------

KIND_CASES = [
    ("int64", blosc2.int64, int, [1, 20, None, 30], 10, -9),
    ("float64", blosc2.float64, float, [1.0, 20.0, None, 30.0], 10.0, float("nan")),
    (
        "string",
        lambda **kw: blosc2.string(max_length=8, **kw),
        str,
        ["aa", "zz", None, "yy"],
        "mm",
        "ZZZZZZZZ",
    ),
    (
        "bytes",
        lambda **kw: blosc2.bytes(max_length=8, **kw),
        bytes,
        [b"aa", b"zz", None, b"yy"],
        b"mm",
        b"ZZZZZZZZ",
    ),
    ("utf8", blosc2.utf8, str, ["aa", "zz", None, "yy"], "mm", "__BLOSC2_NULL__"),
]


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize(("label", "factory", "annotation", "values", "threshold", "sentinel"), KIND_CASES)
def test_negation_drops_nulls_for_every_kind(
    label, factory, annotation, values, threshold, sentinel, storage
):
    """Whatever the kind, ``~(a > x)`` keeps only rows that are definitely not
    greater -- never the ones with nothing to compare."""
    spec, null_cell = nullable(factory, storage, sentinel)
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("a", annotation, blosc2.field(spec)),
        ],
    )
    cells = [null_cell if v is None else v for v in values]
    t = blosc2.CTable(Row, new_data=list(enumerate(cells)))
    assert sorted(t.where(~(t.a > threshold))["id"][:].tolist()) == [0]
    assert sorted(t.where(t.a <= threshold)["id"][:].tolist()) == [0]


def test_dictionary_inequality_no_longer_returns_nulls():
    """A dictionary null is a reserved *code*, so ``!= 'Uber'`` used to be true
    for it -- the code differs from every value's.  It is unknown instead, and
    ``== None`` stays the way to ask for the nulls."""
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("v", str, blosc2.field(blosc2.dictionary(nullable=True))),
        ],
    )
    t = blosc2.CTable(Row, new_data=[(0, "Uber"), (1, "Lyft"), (2, None), (3, "Via")])
    assert sorted(t.where(t.v != "Uber")["id"][:].tolist()) == [1, 3]
    assert sorted(t.where(t.v == "Uber")["id"][:].tolist()) == [0]
    assert sorted(t.where(~(t.v == "Uber"))["id"][:].tolist()) == [1, 3]
    assert sorted(t.where(t.v == None)["id"][:].tolist()) == [2]  # noqa: E711
    assert (t.v != "Uber").is_null().tolist() == [False, False, True, False]


def test_utf8_negation_drops_nulls():
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("s", str, blosc2.field(blosc2.utf8(nullable=True))),
        ],
    )
    t = blosc2.CTable(Row, new_data=[(0, "aa"), (1, "zz"), (2, None), (3, "")])
    assert sorted(t.where(~(t.s == "zz"))["id"][:].tolist()) == [0, 3]
    assert (t.s == "zz").is_null().tolist() == [False, False, True, False]


# ---------------------------------------------------------------------------
# Differential: string form, operator form, oracle -- indexed and not
# ---------------------------------------------------------------------------


def _oracle_table(storage, n=2000, seed=7):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 100, n)
    b = rng.integers(0, 100, n)
    a_null = rng.random(n) < 0.15
    b_null = rng.random(n) < 0.15
    spec, null_cell = nullable(blosc2.int64, storage, -9)
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("id", int, blosc2.field(blosc2.int64())),
            ("a", int, blosc2.field(spec)),
            ("b", int, blosc2.field(spec)),
        ],
    )
    rows = [
        (i, null_cell if a_null[i] else int(a[i]), null_cell if b_null[i] else int(b[i])) for i in range(n)
    ]
    return blosc2.CTable(Row, new_data=rows), a, b, a_null, b_null


def _kleene_pairs(a, b, a_null, b_null):
    """``(true, unknown)`` NumPy channels for the leaves used below."""

    def leaf(values, nulls, op, x):
        return (op(values, x) & ~nulls, nulls)

    return {
        "a > 50": leaf(a, a_null, np.greater, 50),
        "b < 20": leaf(b, b_null, np.less, 20),
        "a != 50": leaf(a, a_null, np.not_equal, 50),
    }


def _np_and(x, y):
    (ta, ua), (tb, ub) = x, y
    return (ta & tb, (ua & (tb | ub)) | (ub & (ta | ua)))


def _np_or(x, y):
    (ta, ua), (tb, ub) = x, y
    return (ta | tb, (ua & ~tb) | (ub & ~ta))


def _np_not(x):
    t, u = x
    return (~t & ~u, u)


SHAPES = [
    ("a > 50", lambda t: t.a > 50, lambda L: L["a > 50"]),
    ("a != 50", lambda t: t.a != 50, lambda L: L["a != 50"]),
    ("~(a > 50)", lambda t: ~(t.a > 50), lambda L: _np_not(L["a > 50"])),
    ("(a > 50) & (b < 20)", lambda t: (t.a > 50) & (t.b < 20), lambda L: _np_and(L["a > 50"], L["b < 20"])),
    ("(a > 50) | (b < 20)", lambda t: (t.a > 50) | (t.b < 20), lambda L: _np_or(L["a > 50"], L["b < 20"])),
    (
        "~((a > 50) & (b < 20))",
        lambda t: ~((t.a > 50) & (t.b < 20)),
        lambda L: _np_not(_np_and(L["a > 50"], L["b < 20"])),
    ),
    (
        "~((a > 50) | (b < 20))",
        lambda t: ~((t.a > 50) | (t.b < 20)),
        lambda L: _np_not(_np_or(L["a > 50"], L["b < 20"])),
    ),
    (
        "~((a > 50) & ~(b < 20))",
        lambda t: ~((t.a > 50) & ~(t.b < 20)),
        lambda L: _np_not(_np_and(L["a > 50"], _np_not(L["b < 20"]))),
    ),
]


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize(("expr", "build", "oracle"), SHAPES, ids=[s[0] for s in SHAPES])
def test_matches_a_numpy_kleene_oracle(storage, indexed, expr, build, oracle):
    """Four ways to the same answer: the string form, the operator form, and a
    NumPy transcription of the truth tables -- with and without an index.

    An ordered index never evaluates the predicate, so an expression being
    null-aware cannot fix its result; the index paths have their own null
    handling, and this is what keeps the two in step.
    """
    t, a, b, a_null, b_null = _oracle_table(storage)
    if indexed:
        t.create_index("a")
        t.create_index("b")
    want_true, want_unknown = oracle(_kleene_pairs(a, b, a_null, b_null))
    expected = sorted(np.flatnonzero(want_true).tolist())

    assert sorted(t.where(expr)["id"][:].tolist()) == expected
    pred = build(t)
    assert sorted(t.where(pred)["id"][:].tolist()) == expected
    np.testing.assert_array_equal(pred.is_null(), want_unknown)


# ---------------------------------------------------------------------------
# Consumers other than where()
# ---------------------------------------------------------------------------


def test_getitem_filter_matches_where():
    t = int_pair_table("mask")
    p = ~(t.a == 1)
    assert t[p]["id"][:].tolist() == t.where(p)["id"][:].tolist()


def test_aggregate_where_takes_a_three_valued_predicate():
    """``where=`` is a row filter, so it collapses like ``where()`` does: the
    unknown rows are not counted."""
    t = int_pair_table("mask")
    p = ~(t.a == 1)
    kept = t.where(p)
    assert t.id.sum(where=p) == kept.id.sum()
    assert t.id.max(where=p) == kept.id.max()


def test_assign_stores_the_collapsed_predicate():
    """A computed column is an array of values with nowhere to keep a third
    one, so the unknown rows land as False -- the same collapse ``where()``
    makes, stated where it happens.

    Sentinel storage, because a *mask* column's predicate reads its sidecar,
    which is not a stored column of the table -- a computed column cannot
    reference it.  That limitation predates three-valued logic (the collapsed
    predicate carried the same operand) and is unrelated to it.
    """
    t = int_pair_table("sentinel")
    flagged = t.assign(flag=~(t.a == 1))
    assert flagged.flag[:].tolist() == [k_not(x) is TRUE for x, _ in PAIRS]


def test_sorted_view_reports_unknown_rows_in_view_order():
    """``is_null()`` is logical, so it follows the view the predicate was built
    on -- not the physical order the channels live in."""
    t = int_pair_table("mask")
    view = t.sort_by("id", ascending=False)
    p = view.a == 1
    assert p.is_null().tolist() == [x is UNKNOWN for x, _ in reversed(PAIRS)]
