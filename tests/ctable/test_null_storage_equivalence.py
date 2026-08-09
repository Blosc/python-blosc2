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

Everything else agrees, and there are no known unintentional divergences left:
the ``isin`` and ``to_pandas`` leaks this suite was first written to pin are
both fixed, and their tests are now plain assertions.  The one remaining
inexactness is not a divergence but a limit of the destination format --
pandas has no float dtype separating NaN from missing, so a mask float column
cannot round-trip through it (:func:`test_a_mask_float_cannot_round_trip_through_pandas`).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from utf8_compat import HAVE_UTF8, needs_utf8, utf8_spec

import blosc2
from blosc2.ctable_nulls import is_na_marker

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


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_isin_agrees(kind):
    """A null matches nothing -- not even the value that stands in for it."""
    m, s = both(kind)
    # Probe with each storage's own stand-in for a null.  Neither should match,
    # because neither stand-in is the row's value: the fill is not part of the
    # format contract, and the sentinel is reserved.
    for probe in (m["a"][:][NULL_ROW], s["a"][:][NULL_ROW]):
        got, want = m["a"].isin([probe]).tolist(), s["a"].isin([probe]).tolist()
        assert_same(got, want, f"isin([{probe!r}])")
        if is_na_marker(probe):
            # A timestamp's fill decodes to NaT, which *is* a way of spelling
            # "missing", so probing with it is asking for the nulls.
            assert got[NULL_ROW], f"{probe!r} is a null marker, so it selects nulls"
        else:
            assert not got[NULL_ROW], "a null row is not a member of anything"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_isin_none_selects_the_nulls(kind):
    """``None`` is how you ask for them, and it agrees with is_null exactly."""
    for t in both(kind):
        assert_same(t["a"].isin([None]).tolist(), t["a"].is_null().tolist(), "isin([None]) vs is_null")


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_isin_none_beside_a_real_value(kind):
    """Asking for a value *and* the nulls gets both, and nothing else."""
    m, s = both(kind)
    rows = logical(m["a"])
    present = rows[0]
    want = [v is None or v == present for v in rows]
    assert m["a"].isin([present, None]).tolist() == want
    assert s["a"].isin([present, None]).tolist() == want


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_to_pandas_agrees(kind):
    """A null reaches pandas as NA, not as whatever stands in for it."""
    pytest.importorskip("pandas")
    m, s = both(kind)
    got, want = m.to_pandas()["a"], s.to_pandas()["a"]
    assert_same(got.isna().tolist(), want.isna().tolist(), "to_pandas isna")
    assert_same(got.isna().tolist(), m["a"].is_null().tolist(), "to_pandas isna vs is_null")
    assert got.isna()[NULL_ROW], "the null row should read as NA"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_to_pandas_keeps_the_values_it_does_have(kind):
    """Expressing the nulls must not disturb the rows that are not null."""
    pytest.importorskip("pandas")
    for t in both(kind):
        series = t.to_pandas()["a"]
        for i, value in enumerate(logical(t["a"])):
            if value is None:
                continue
            got = series[i]
            got = got.item() if hasattr(got, "item") else got
            assert got == value, f"row {i}: {got!r} != {value!r}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])


# ---------------------------------------------------------------------------
# The pandas round trip
# ---------------------------------------------------------------------------


def row_cls_for(kind: str, storage: str):
    """A one-column dataclass matching what :func:`build` produces."""
    factory, _values, sentinel = KINDS[kind]
    spec = (
        factory(nullable=True, null_storage="mask")
        if storage == "mask"
        else factory(nullable=True, null_value=sentinel)
    )
    return dataclasses.make_dataclass("RoundTripRow", [("a", annotation_for(spec), blosc2.field(spec))])


#: Float is excluded: pandas spells missing as NaN in every float dtype it
#: has, and a mask float column keeps NaN a value (decision 6), so the two
#: cannot survive the trip.  :func:`test_a_mask_float_cannot_round_trip_through_pandas`
#: pins that limitation rather than leaving it implicit.
ROUND_TRIP_KINDS = [k for k in ALL_KINDS if not k.startswith("float")]


@pytest.mark.parametrize("kind", ROUND_TRIP_KINDS)
@pytest.mark.parametrize("storage", ["mask", "sentinel"])
def test_pandas_round_trip_keeps_the_nulls(kind, storage):
    """to_pandas must emit something from_pandas can read back unchanged."""
    pytest.importorskip("pandas")
    t = build(kind, storage)
    back = blosc2.CTable.from_pandas(t.to_pandas()[["a"]], row_cls_for(kind, storage))
    assert_same(logical(back["a"]), logical(t["a"]), f"{storage} round trip")
    assert logical(back["a"])[NULL_ROW] is None


@pytest.mark.parametrize("storage", ["mask", "sentinel"])
def test_pandas_round_trip_of_a_null_free_column(storage):
    """The common case must not be disturbed by any of the above."""
    pytest.importorskip("pandas")
    spec = (
        blosc2.int64(nullable=True, null_storage="mask")
        if storage == "mask"
        else blosc2.int64(nullable=True, null_value=-9)
    )
    row_cls = dataclasses.make_dataclass("PlainRow", [("a", int, blosc2.field(spec))])
    t = blosc2.CTable(row_cls, expected_size=16)
    t.extend([(1,), (2,), (3,)])
    df = t.to_pandas()
    # No null, so nothing widens: the dtype is what it has always been here.
    assert df["a"].dtype == np.dtype(np.int64)
    back = blosc2.CTable.from_pandas(df, row_cls)
    assert logical(back["a"]) == [1, 2, 3]


@pytest.mark.parametrize("kind", ["float32", "float64"])
def test_a_mask_float_cannot_round_trip_through_pandas(kind):
    """The one stated limitation, pinned so it stays a known quantity.

    pandas has no float dtype that distinguishes NaN from missing -- even
    ``Float64`` folds a NaN into ``NA`` -- so a mask float column's null comes
    back as a NaN *value*.  A sentinel float column is lossless precisely
    because NaN is what it means by null in the first place.
    """
    pytest.importorskip("pandas")
    mask_back = blosc2.CTable.from_pandas(build(kind, "mask").to_pandas()[["a"]], row_cls_for(kind, "mask"))
    assert mask_back["a"].null_count() == 0, "the null came back as a NaN value"
    assert np.isnan(mask_back["a"][:][NULL_ROW])

    sentinel_back = blosc2.CTable.from_pandas(
        build(kind, "sentinel").to_pandas()[["a"]], row_cls_for(kind, "sentinel")
    )
    assert sentinel_back["a"].null_count() == 1
    # to_arrow is the export that keeps the distinction for both storages.
    assert build(kind, "mask").to_arrow()["a"].null_count == 1
