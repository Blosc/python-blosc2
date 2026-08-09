#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""The differential oracle: sentinel and mask storage must answer alike.

CTable can keep a column's nulls in two places -- an in-band **sentinel**, or a
sidecar **validity mask** -- and a second implementation of anything is a
standing invitation to drift.  This suite builds the *same logical data* twice,
once each way, and asserts that every public API gives the same answer.  That
turns "mask is a second implementation" from a permanent liability into a
checked invariant.

Comparison is always **logical**, never physical: a column is read as its
values with ``None`` substituted wherever ``is_null()`` is True (:func:`logical`
below).  What sits underneath a null is the fill for one storage and the
sentinel for the other, and neither is part of the format contract -- asserting
on it would pin down something the design explicitly says may change.

Three differences are **deliberate** and asserted as differences rather than
papered over:

* **NaN is a value under mask storage** and null under a sentinel one
  (decision 6).  This is the entire point of a side channel, so the float cases
  here carry no NaN, and :func:`test_nan_is_a_value_only_under_mask_storage`
  pins the divergence on its own.
* **A nullable bool is physically ``uint8``** under a sentinel, to leave room
  for the reserved ``255``, and plain ``np.bool_`` under a mask.  The values
  still compare equal (``True == 1``), which is what the oracle checks.
* **complex is mask-only**: no complex value is safe to reserve, so there is no
  sentinel column to compare against.

Two more differences are **not** deliberate; they are open bugs, pinned below
as strict xfails so that fixing either one trips this suite.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from utf8_compat import HAVE_UTF8, needs_utf8, utf8_spec

import blosc2

T0 = np.datetime64("2021-03-04T05:06:07", "s")


def secs(n: int) -> np.timedelta64:
    """A timedelta with an explicit unit; a bare int is deprecated in NumPy."""
    return np.timedelta64(n, "s")


#: ``kind -> (spec factory, logical values, sentinel literal)``.
#:
#: Every V1 kind, and the values are chosen to be awkward on purpose: a full
#: range of ``int8``/``uint8``, a genuine ``""`` beside a null in each text
#: kind, and a ``0``/``0.0`` that a fill could be confused with.  No float NaN
#: -- that is decision 6's business and is tested separately.
KINDS: dict = {
    "int64": (lambda **k: blosc2.int64(**k), [5, None, -3, 0, 7], -9),
    "int8": (lambda **k: blosc2.int8(**k), [1, None, -5, 0, 127], -128),
    "uint8": (lambda **k: blosc2.uint8(**k), [1, None, 200, 0, 7], 255),
    "float64": (lambda **k: blosc2.float64(**k), [1.5, None, -2.5, 0.0, 7.25], float("nan")),
    "float32": (lambda **k: blosc2.float32(**k), [1.5, None, -2.5, 0.0, 7.25], float("nan")),
    "bool": (lambda **k: blosc2.bool(**k), [True, None, False, True, False], 255),
    "timestamp": (
        lambda **k: blosc2.timestamp(**k),
        [T0, None, T0 + secs(60), T0 + secs(5), T0 + secs(1)],
        int(np.iinfo(np.int64).min),
    ),
    "string": (lambda **k: blosc2.string(max_length=8, **k), ["a", None, "", "zz", "m"], "ZZZZZZZZ"),
    "bytes": (
        lambda **k: blosc2.bytes(max_length=8, **k),
        [b"a", None, b"", b"zz", b"m"],
        b"ZZZZZZZZ",
    ),
}
if HAVE_UTF8:
    KINDS["utf8"] = (utf8_spec, ["a", None, "", "zz", "m"], "__BLOSC2_NULL__")

ALL_KINDS = sorted(KINDS)
#: Kinds whose values support ordered arithmetic reductions.
NUMERIC_KINDS = [k for k in ALL_KINDS if k.startswith(("int", "uint", "float"))]
#: The row that is null in every table this module builds.
NULL_ROW = 1

#: Kinds whose *fill* and *sentinel* happen to be the same value, so an API
#: that leaks what stands in for a null leaks something indistinguishable
#: either way.  A float column fills with NaN and reserves NaN; a timestamp
#: fills with ``int64.min`` and reserves ``int64.min``.  They are the kinds
#: where the two bugs below are invisible -- not the kinds where they are
#: fixed -- so they still assert the correct behaviour, just without the xfail.
INDISTINGUISHABLE_FILL = ("float32", "float64", "timestamp")


def leaks_its_fill(kind: str) -> bool:
    """Whether a raw read of *kind*'s null slot exposes an ordinary-looking value."""
    return kind not in INDISTINGUISHABLE_FILL


def kinds_xfailing_on_leak(reason: str):
    """``ALL_KINDS``, with the fill-leaking ones marked ``xfail(strict=True)``."""
    return [
        pytest.param(
            kind,
            marks=pytest.mark.xfail(strict=True, reason=reason) if leaks_its_fill(kind) else (),
        )
        for kind in ALL_KINDS
    ]


def annotation_for(spec):
    if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp)):
        return object
    return spec.python_type


def build(kind: str, storage: str, *, capacity: int = 64):
    """The same logical rows, stored *storage*'s way.

    Column ``a`` carries the nulls; ``g`` is a plain non-nullable key so the
    group-by and multi-key sort cases have something to work with.
    """
    factory, values, sentinel = KINDS[kind]
    if storage == "mask":
        spec = factory(nullable=True, null_storage="mask")
        cells = values
    else:
        spec = factory(nullable=True, null_value=sentinel)
        # A sentinel column has no way to spell "null" other than its sentinel.
        cells = [sentinel if v is None else v for v in values]
    row_cls = dataclasses.make_dataclass(
        "EquivRow",
        [
            ("a", annotation_for(spec), blosc2.field(spec)),
            ("g", str, blosc2.field(blosc2.string(max_length=4))),
        ],
    )
    t = blosc2.CTable(row_cls, expected_size=capacity)
    t.extend([(v, f"g{i % 2}") for i, v in enumerate(cells)])
    return t


def both(kind: str, **kwargs):
    """The mask-backed and sentinel-backed tables for *kind*."""
    return build(kind, "mask", **kwargs), build(kind, "sentinel", **kwargs)


def logical(col) -> list:
    """A column as its observable contents: values, with ``None`` where null.

    This is the whole comparison discipline of this suite.  Reading ``col[:]``
    alone would compare the fill against the sentinel and fail everywhere for
    reasons that are not bugs.
    """
    nulls = col.is_null()
    out = []
    for value, is_null in zip(col[:], nulls, strict=True):
        if is_null:
            out.append(None)
        else:
            out.append(value.item() if hasattr(value, "item") else value)
    return out


def logical_rows(t, cols=("a", "g")) -> list[tuple]:
    """Every live row of *t*, each column read through :func:`logical`."""
    columns = [logical(t[c]) for c in cols]
    return [tuple(col[i] for col in columns) for i in range(t.nrows)]


def _null_last(value):
    """Sort key putting ``None`` last and ordering the rest by value.

    By value, not by ``repr``: a nullable bool is ``np.bool_`` under a mask and
    ``uint8`` under a sentinel, so ``False``/``0`` sort together here and
    compare equal afterwards, while their reprs would not.
    """
    return (value is None, 0 if value is None else value)


def assert_same(mask_result, sentinel_result, what: str) -> None:
    """Assert the two storages agree, reporting which API disagreed."""
    assert mask_result == sentinel_result, (
        f"{what}: mask storage gave {mask_result!r}, sentinel storage gave {sentinel_result!r}"
    )


# ---------------------------------------------------------------------------
# The null API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_reads_agree(kind):
    """The baseline: the same rows, and the same rows are null."""
    m, s = both(kind)
    assert_same(logical(m["a"]), logical(s["a"]), "column contents")
    assert_same(logical(m["a"])[NULL_ROW], None, "the null row")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_null_api_agrees(kind):
    m, s = both(kind)
    assert_same(m["a"].is_null().tolist(), s["a"].is_null().tolist(), "is_null")
    assert_same(m["a"].notnull().tolist(), s["a"].notnull().tolist(), "notnull")
    assert_same(m["a"].null_count(), s["a"].null_count(), "null_count")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_to_numpy_masked_agrees(kind):
    """The masked view is the one read that must work for *both* storages."""
    m, s = both(kind)
    got, want = m["a"].to_numpy(masked=True), s["a"].to_numpy(masked=True)
    assert_same(got.mask.tolist(), want.mask.tolist(), "to_numpy(masked=True).mask")
    assert_same(got.compressed().tolist(), want.compressed().tolist(), "to_numpy(masked=True) values")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_dropna_agrees(kind):
    m, s = both(kind)
    assert_same(logical_rows(m.dropna()), logical_rows(s.dropna()), "dropna")
    assert m.dropna().nrows == len(KINDS[kind][1]) - 1


@pytest.mark.parametrize("kind", NUMERIC_KINDS)
def test_fillna_agrees(kind):
    m, s = both(kind)
    assert_same(m["a"].fillna(42).tolist(), s["a"].fillna(42).tolist(), "fillna")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_unique_and_value_counts_agree(kind):
    """Both exclude the null, and neither leaks what sits under it."""
    m, s = both(kind)
    assert_same(sorted(m["a"].unique().tolist()), sorted(s["a"].unique().tolist()), "unique")
    assert_same(
        sorted(m["a"].value_counts().items()),
        sorted(s["a"].value_counts().items()),
        "value_counts",
    )


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("ascending", [True, False])
def test_sort_by_agrees(kind, ascending):
    """Nulls sort last in both directions, whichever storage they live in."""
    m, s = both(kind)
    got = logical(m.sort_by("a", ascending=ascending)["a"])
    want = logical(s.sort_by("a", ascending=ascending)["a"])
    assert_same(got, want, f"sort_by(ascending={ascending})")
    assert got[-1] is None, "nulls sort last"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_multi_key_sort_agrees(kind):
    m, s = both(kind)
    assert_same(logical_rows(m.sort_by(["g", "a"])), logical_rows(s.sort_by(["g", "a"])), "sort_by 2 keys")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_sorted_view_agrees(kind):
    m, s = both(kind)
    assert_same(
        logical(m.sort_by("a", view=True)["a"]),
        logical(s.sort_by("a", view=True)["a"]),
        "sort_by(view=True)",
    )


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", NUMERIC_KINDS)
@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std"])
def test_reductions_agree(kind, op):
    """Every reduction skips the null rather than folding in what stands for it."""
    m, s = both(kind)
    got, want = getattr(m["a"], op)(), getattr(s["a"], op)()
    assert got == pytest.approx(want), f"{op}: {got!r} != {want!r}"


@pytest.mark.parametrize("kind", NUMERIC_KINDS + ["timestamp"])
@pytest.mark.parametrize("op", ["argmin", "argmax"])
def test_arg_reductions_agree(kind, op):
    m, s = both(kind)
    got, want = int(getattr(m["a"], op)()), int(getattr(s["a"], op)())
    assert_same(got, want, op)
    assert got != NULL_ROW, "a null row can never be the extremum"


@pytest.mark.parametrize("kind", ["timestamp", "string", "bytes"])
@pytest.mark.parametrize("op", ["min", "max"])
def test_ordered_non_numeric_reductions_agree(kind, op):
    m, s = both(kind)
    assert_same(getattr(m["a"], op)(), getattr(s["a"], op)(), op)


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS)
@pytest.mark.parametrize("dropna", [True, False])
def test_group_by_key_agrees(kind, dropna):
    """As a *key*: the null forms its own group, keyed None, under both storages.

    A sentinel column stores that group's key as its sentinel and a mask column
    as its fill, so this only holds when the key column is read logically --
    which is the point.  It is also the one place a sentinel column has to
    round-trip its own sentinel back into a null.
    """
    m, s = both(kind)
    got = sorted(logical(m.group_by("a", dropna=dropna).count("g")["a"]), key=_null_last)
    want = sorted(logical(s.group_by("a", dropna=dropna).count("g")["a"]), key=_null_last)
    assert_same(got, want, f"group_by key (dropna={dropna})")
    assert (None in got) is not dropna, "dropna decides whether the null group exists"


@pytest.mark.parametrize("kind", NUMERIC_KINDS)
@pytest.mark.parametrize("op", ["sum", "min", "max", "count"])
def test_group_by_value_agrees(kind, op):
    """As a *value*: the null must not be aggregated as if it were its fill."""
    m, s = both(kind)
    got = getattr(m.group_by("g"), op)("a")
    want = getattr(s.group_by("g"), op)("a")
    assert_same(logical_rows(got, got.col_names), logical_rows(want, want.col_names), f"group_by {op}")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

#: Expression shapes the two storages must answer identically.  Each is written
#: so the *stored* null value would satisfy it under at least one storage if
#: nullity leaked -- which is what makes them worth asserting.
NUMERIC_EXPRESSIONS = [
    "a > 0",
    "a < 3",
    "a == 0",
    "a != 0",
    "(a > 0) & (g == 'g0')",
    "(a > 0) | (g == 'g1')",
    "~(a > 0)",
    "(a < 3) & (a > -100)",
]


@pytest.mark.parametrize("kind", ["int64", "int8", "uint8", "float64"])
@pytest.mark.parametrize("expression", NUMERIC_EXPRESSIONS)
def test_where_agrees(kind, expression):
    m, s = both(kind)
    assert_same(logical_rows(m.where(expression)), logical_rows(s.where(expression)), f"where({expression})")


def build_big(storage: str, *, indexed: bool, n: int = 4000):
    """A table big enough for the planner to actually reach for an index.

    Every seventh row is null, and the values straddle zero so the expressions
    below select a real range rather than everything or nothing.
    """
    values = [None if i % 7 == 0 else (i % 101) - 50 for i in range(n)]
    sentinel = -9999
    if storage == "mask":
        spec = blosc2.int64(nullable=True, null_storage="mask")
        cells = values
    else:
        spec = blosc2.int64(nullable=True, null_value=sentinel)
        cells = [sentinel if v is None else v for v in values]
    row_cls = dataclasses.make_dataclass(
        "BigRow",
        [("a", int, blosc2.field(spec)), ("g", str, blosc2.field(blosc2.string(max_length=4)))],
    )
    t = blosc2.CTable(row_cls, expected_size=n + 16, create_summary_index=False)
    t.extend([(v, f"g{i % 2}") for i, v in enumerate(cells)])
    if indexed:
        t.create_index("a", kind="summary")
        assert "a" in t._get_index_catalog(), "the index this test is about was not built"
    else:
        assert "a" not in t._get_index_catalog(), "this table was meant to have no index"
    return t


@pytest.mark.parametrize("expression", NUMERIC_EXPRESSIONS)
def test_indexed_where_agrees(expression):
    """The same answers with an index in play.

    An ordered index answers by slicing the sorted column rather than by
    evaluating the predicate, so this is the path where a leaked null shows up
    as a *different* result from the identical unindexed query -- the failure
    the plan's Addendum 2 found leaking every null through a float index.
    """
    m = build_big("mask", indexed=True)
    s = build_big("sentinel", indexed=True)
    assert_same(
        logical_rows(m.where(expression)), logical_rows(s.where(expression)), f"indexed {expression}"
    )


@pytest.mark.parametrize("storage", ["mask", "sentinel"])
@pytest.mark.parametrize("expression", NUMERIC_EXPRESSIONS)
def test_an_index_does_not_change_the_answer(storage, expression):
    """Each storage must agree with *itself*, indexed versus not.

    Cross-storage agreement alone would not catch an index that leaks nulls
    into both storages the same way, so this is the other half of the oracle:
    the scan is the reference, and the index has to match it.
    """
    indexed = build_big(storage, indexed=True)
    scanned = build_big(storage, indexed=False)
    assert_same(
        logical_rows(indexed.where(expression)),
        logical_rows(scanned.where(expression)),
        f"{storage}: indexed vs scanned {expression}",
    )


@needs_utf8
@pytest.mark.parametrize("expression", ["a == 'a'", "a != 'a'", "(a == 'a') | (g == 'g1')"])
def test_text_where_agrees(expression):
    m, s = both("utf8")
    assert_same(logical_rows(m.where(expression)), logical_rows(s.where(expression)), f"where({expression})")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_a_null_never_satisfies_a_predicate(kind):
    """SQL WHERE semantics, which is what both storages promise (decision 8)."""
    for t in both(kind):
        for row in logical_rows(t.where("a == a")):
            assert row[0] is not None


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_to_arrow_agrees(kind):
    """Arrow has a validity bitmap of its own, so this is the exact comparison."""
    m, s = both(kind)
    got, want = m.to_arrow()["a"], s.to_arrow()["a"]
    assert_same(got.null_count, want.null_count, "to_arrow null_count")
    assert_same(got.to_pylist(), want.to_pylist(), "to_arrow values")
    assert got.to_pylist()[NULL_ROW] is None


# ---------------------------------------------------------------------------
# Deliberate divergences
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["float64", "float32"])
def test_nan_is_a_value_only_under_mask_storage(kind):
    """Decision 6, and the reason the float rows above carry no NaN.

    A sentinel float column spells its null ``NaN``, so it cannot also hold one
    as data; a mask column follows Arrow and keeps NaN an ordinary value.  This
    is the divergence the side channel exists to create, so it is asserted
    rather than tolerated.
    """
    factory = KINDS[kind][0]
    nan = float("nan")

    mask_t = build_one(factory(nullable=True, null_storage="mask"), [1.0, nan, 3.0])
    sent_t = build_one(factory(nullable=True, null_value=nan), [1.0, nan, 3.0])

    assert mask_t["a"].is_null().tolist() == [False, False, False], "NaN is a value under a mask"
    assert sent_t["a"].is_null().tolist() == [False, True, False], "NaN is the null under a sentinel"
    assert mask_t["a"].null_count() == 0
    assert sent_t["a"].null_count() == 1
    # And it follows through to the reductions, which is where a user meets it.
    assert np.isnan(mask_t["a"].sum())
    assert sent_t["a"].sum() == 4.0


def build_one(spec, values):
    """A one-column table of *values*, with no null coercion of any kind."""
    row_cls = dataclasses.make_dataclass("NanRow", [("a", annotation_for(spec), blosc2.field(spec))])
    t = blosc2.CTable(row_cls, expected_size=16)
    t.extend([(v,) for v in values])
    return t


def test_nullable_bool_is_uint8_only_under_a_sentinel():
    """The reserved 255 needs room; a mask column has np.bool_ and needs none."""
    m, s = both("bool")
    assert m["a"].dtype == np.dtype(np.bool_)
    assert s["a"].dtype == np.dtype(np.uint8)
    # The physical difference is invisible to the logical read, which is why
    # every other bool case in this suite compares equal.
    assert_same(logical(m["a"]), logical(s["a"]), "bool contents")


def test_complex_has_no_sentinel_to_compare_against():
    """complex is mask-only: no complex value is safe to reserve."""
    with pytest.raises((TypeError, ValueError)):
        blosc2.complex128(nullable=True, null_value=0j)
    assert blosc2.complex128(nullable=True).null_storage == "mask"


# ---------------------------------------------------------------------------
# Divergences that are bugs
# ---------------------------------------------------------------------------


ISIN_LEAK = (
    "isin() reads col[:] and tests membership on the raw values, so it matches whatever "
    "stands in for a null -- the fill under mask storage, the sentinel under a sentinel "
    "one. A null row should match nothing."
)
TO_PANDAS_LEAK = (
    "to_pandas() writes the raw values, so a null arrives as the fill under mask storage "
    "and as the sentinel under a sentinel one. Neither is NA, and to_arrow() on the same "
    "data is already exact."
)


@pytest.mark.parametrize("kind", kinds_xfailing_on_leak(ISIN_LEAK))
def test_isin_agrees(kind):
    m, s = both(kind)
    # Probe with each storage's own stand-in for a null: neither should match.
    mask_stand_in = m["a"][:][NULL_ROW]
    sentinel_stand_in = s["a"][:][NULL_ROW]
    for probe in (mask_stand_in, sentinel_stand_in):
        got, want = m["a"].isin([probe]).tolist(), s["a"].isin([probe]).tolist()
        assert_same(got, want, f"isin([{probe!r}])")
        assert not got[NULL_ROW], "a null row is not a member of anything"


@pytest.mark.parametrize("kind", kinds_xfailing_on_leak(TO_PANDAS_LEAK))
def test_to_pandas_agrees(kind):
    pd = pytest.importorskip("pandas")
    m, s = both(kind)
    got, want = m.to_pandas()["a"], s.to_pandas()["a"]
    assert_same(got.isna().tolist(), want.isna().tolist(), "to_pandas isna")
    assert got.isna()[NULL_ROW], "the null row should read as NA"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
