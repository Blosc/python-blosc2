#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Null-aware column indexes (Phase 10 of plans/mask-based-nulls.md).

An index summarises the column's *physical* array, so until now every nullable
column looked to it like a column of data: a sentinel entered the per-segment
extrema, and so did a mask column's fill.  That is why ``min()``/``max()``
declined the summary shortcut for any nullable column except the one case where
the null happens to be a NaN the summary builder already dropped.

Phase 10 hands the builder the column's validity channel.  The extrema then
cover only rows that carry a value, a segment with no such row is flagged
``FLAG_ALL_NULL``, and the descriptor records ``null_aware`` so that indexes
built before this keep the old bail instead of being trusted.  The payoff lands
on *both* storages: a mask column and an ``INT64_MIN``-sentinel one are equally
unreadable to a summary that does not know about nulls.

The second half is ``where()``: an OR over a nullable indexed column used to
fall back to a full scan, because the only null filtering available was global
and a global filter drops a row that is null in one branch but matches the
other.  The segment path never needed that filter -- it *evaluates* the
predicate, which has been null-aware per leaf since Phase 1 -- so it now serves
OR directly.  The ordered-range paths still bail, for the reason recorded in
§Expression layer of the plan: an index that answers by slicing the sorted
column never evaluates anything.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2
from blosc2.indexing import (
    FLAG_ALL_NAN,
    FLAG_ALL_NULL,
    FLAG_HAS_NAN,
    _compute_segment_summaries,
)


def one_col(values, spec, urlpath, capacity=None, name="a"):
    Row = dataclasses.make_dataclass("R", [(name, spec.python_type, blosc2.field(spec))])
    t = blosc2.CTable(Row, expected_size=max(len(values), capacity or 0), urlpath=str(urlpath), mode="w")
    t.extend([(v,) for v in values])
    return t


# ---------------------------------------------------------------------------
# The summaries themselves
# ---------------------------------------------------------------------------


def test_summaries_take_their_extrema_over_the_valid_rows_only():
    """The whole phase in one assertion: -1 is a null, not the minimum."""
    values = np.array([-1, 7, 3, -1, 9, 5], dtype=np.int64)
    valid = np.array([False, True, True, False, True, True])
    naive = _compute_segment_summaries(values, values.dtype, 3)
    aware = _compute_segment_summaries(values, values.dtype, 3, valid)
    assert naive["min"].tolist() == [-1, -1]
    assert aware["min"].tolist() == [3, 5]
    assert aware["max"].tolist() == [7, 9]


def test_an_all_null_segment_is_flagged_rather_than_summarised():
    """FLAG_ALL_NAN rides along with FLAG_ALL_NULL so that a reader which knows
    only the NaN flags still skips the segment instead of trusting its zeros."""
    values = np.array([1, 2, 8, 9], dtype=np.int64)
    valid = np.array([True, True, False, False])
    summaries = _compute_segment_summaries(values, values.dtype, 2, valid)
    assert not summaries["flags"][0] & FLAG_ALL_NULL
    assert summaries["flags"][1] & FLAG_ALL_NULL
    assert summaries["flags"][1] & FLAG_ALL_NAN
    # The placeholder extrema must not look like data.
    assert summaries["min"][1] == 0
    assert summaries["max"][1] == 0


def test_has_nan_marks_a_real_nan_and_not_a_nan_fill():
    """This flag is what keeps min() honest on a mask float column.

    A mask column's fill *is* NaN, so a flag raised over all rows would mark
    every segment holding a null.  Taken over the valid rows it marks exactly
    the segments where NaN is data -- the case decision 6 says must poison
    min() to NaN, and so must not be answered from the summaries.
    """
    values = np.array([1.0, np.nan, 5.0, np.nan], dtype=np.float64)
    valid = np.array([True, False, True, True])  # index 1 is a null, index 3 a value
    summaries = _compute_segment_summaries(values, values.dtype, 2, valid)
    assert not summaries["flags"][0] & FLAG_HAS_NAN  # the fill
    assert summaries["flags"][1] & FLAG_HAS_NAN  # the value
    assert summaries["min"][0] == 1.0


def test_string_segments_skip_their_nulls_too():
    values = np.array(["", "pear", "", "apple"], dtype="U8")
    valid = np.array([False, True, False, True])
    summaries = _compute_segment_summaries(values, values.dtype, 4, valid)
    assert summaries["min"][0] == "apple"
    assert summaries["max"][0] == "pear"


# ---------------------------------------------------------------------------
# The descriptor claim
# ---------------------------------------------------------------------------

STORAGES = [
    pytest.param(blosc2.int64(null_storage="mask"), False, id="mask"),
    pytest.param(blosc2.int64(nullable=True, null_value=-1), True, id="sentinel"),
]


@pytest.mark.parametrize(("spec", "write_sentinel"), STORAGES)
def test_a_nullable_column_records_null_aware(spec, write_sentinel, tmp_path):
    vals = [None if i % 7 == 0 else i for i in range(500)]
    if write_sentinel:
        vals = [-1 if v is None else v for v in vals]
    t = one_col(vals, spec, tmp_path / "d.b2t")
    t.create_index("a", kind="summary")
    assert t._get_index_catalog()["a"]["null_aware"] is True


def test_a_null_free_mask_column_is_null_aware_without_a_sidecar(tmp_path):
    """Nothing to exclude is not the same as nothing known: the summaries of a
    column that has never held a null already agree with a null-skipping scan,
    so it must not pay the bail."""
    t = one_col(list(range(500)), blosc2.int64(null_storage="mask"), tmp_path / "n.b2t")
    assert t._null_mask("a") is None
    t.create_index("a", kind="summary")
    assert t._get_index_catalog()["a"]["null_aware"] is True
    assert t["a"]._index_summary_minmax("min") == 0


def test_an_index_built_before_phase_10_keeps_the_bail(tmp_path):
    """The staleness rule for this phase.  An older index carries no
    ``null_aware`` key and its extrema cover the sentinel, so it must be read as
    unusable rather than as False-meaning-anything-else."""
    vals = [-1 if i % 7 == 0 else i for i in range(500)]
    t = one_col(vals, blosc2.int64(nullable=True, null_value=-1), tmp_path / "old.b2t")
    t.create_index("a", kind="summary")
    assert t["a"]._summary_minmax_source() is not None

    del t._get_index_catalog()["a"]["null_aware"]  # simulate the older build
    assert t["a"]._summary_minmax_source() is None
    assert t["a"].min() == 1  # the scan still answers, and answers correctly

    t.rebuild_index("a")
    assert t["a"]._summary_minmax_source() is not None


# ---------------------------------------------------------------------------
# min() / max() through the summaries
# ---------------------------------------------------------------------------

MINMAX_KINDS = [
    pytest.param("int64", blosc2.int64(null_storage="mask"), lambda i: i * 3 % 977, id="int64"),
    pytest.param(
        "float64", blosc2.float64(null_storage="mask"), lambda i: (i * 7 % 991) / 3.0, id="float64"
    ),
    pytest.param("uint16", blosc2.uint16(null_storage="mask"), lambda i: i * 5 % 60_000, id="uint16"),
    pytest.param("bool", blosc2.bool(null_storage="mask"), lambda i: bool(i % 3), id="bool"),
    pytest.param(
        "string", blosc2.string(max_length=8, null_storage="mask"), lambda i: f"s{i % 977:05d}", id="string"
    ),
]


@pytest.mark.parametrize(("label", "spec", "value_of"), MINMAX_KINDS)
def test_the_shortcut_agrees_with_the_scan_for_every_kind(label, spec, value_of, tmp_path):
    n = 20_000
    vals = [None if i % 11 == 0 else value_of(i) for i in range(n)]
    live = [v for v in vals if v is not None]
    t = one_col(vals, spec, tmp_path / f"{label}.b2t")
    t.create_index("a", kind="summary")

    assert t["a"]._index_summary_minmax("min") is not NotImplemented
    assert t["a"].min() == min(live)
    assert t["a"].max() == max(live)


def test_a_full_range_int8_reports_its_true_extrema(tmp_path):
    """The case no sentinel column can express, so nothing else could test it:
    -128 and 127 are both data and there is no spare value to mean null."""
    n = 5_000
    vals = [None if i % 9 == 0 else (i % 256) - 128 for i in range(n)]
    live = [v for v in vals if v is not None]
    t = one_col(vals, blosc2.int8(null_storage="mask"), tmp_path / "i8.b2t")
    t.create_index("a", kind="summary")
    assert t["a"]._index_summary_minmax("min") == -128
    assert t["a"].min() == min(live) == -128
    assert t["a"].max() == max(live) == 127


def test_the_sentinel_storage_gets_the_same_answer(tmp_path):
    """Phase 10 is not a mask feature: an INT64_MIN sentinel is exactly as
    invisible to a summary that does not consult a validity channel."""
    imin = np.iinfo(np.int64).min
    n = 20_000
    vals = [imin if i % 11 == 0 else (i * 3 % 977) for i in range(n)]
    live = [v for v in vals if v != imin]
    t = one_col(vals, blosc2.int64(nullable=True, null_value=imin), tmp_path / "s.b2t")
    t.create_index("a", kind="summary")
    assert t["a"]._index_summary_minmax("min") == min(live)
    assert t["a"].max() == max(live)


def test_nulls_in_the_straddling_block_are_skipped_by_hand(tmp_path):
    """The one block the summaries cannot cover is rescanned, and that rescan
    has to apply the same rule they did."""
    n = 100_003  # big enough for several whole blocks, and not a multiple of one
    vals = list(range(10, n + 10))
    t = one_col(vals, blosc2.int64(null_storage="mask"), tmp_path / "tail.b2t")
    t.create_index("a", kind="summary")
    segment_len = t["a"]._summary_minmax_source()[-1]
    tail_start = (n // segment_len) * segment_len
    assert tail_start < n, "the fixture needs a partial trailing block"

    # Put both a new minimum and a null in the rescanned tail.
    t["a"][tail_start] = 0
    t["a"][tail_start + 1] = None
    t.rebuild_index("a")
    assert t["a"]._index_summary_minmax("min") == 0
    assert t["a"].min() == 0
    assert t["a"].null_count() == 1


def test_an_all_null_column_falls_back_to_the_scan(tmp_path):
    """Every segment is FLAG_ALL_NULL, so there is nothing left to reduce and
    the shortcut must decline rather than answer with the placeholder zeros."""
    t = one_col([None] * 2_000, blosc2.int64(null_storage="mask"), tmp_path / "allnull.b2t")
    t.create_index("a", kind="summary")
    assert t["a"]._index_summary_minmax("min") is NotImplemented
    with pytest.raises(ValueError):
        t["a"].min()


def test_a_null_written_after_the_build_takes_the_shortcut_away(tmp_path):
    """A mask column with no sidecar claims null_aware, so the claim has to stop
    holding the moment a null appears.  It does, through the ordinary staleness
    rule: the write invalidates the index."""
    t = one_col(list(range(5_000)), blosc2.int64(null_storage="mask"), tmp_path / "later.b2t")
    t.create_index("a", kind="summary")
    assert t["a"]._index_summary_minmax("min") == 0

    t["a"][0] = None
    assert t._get_index_catalog()["a"]["stale"] is True
    assert t["a"]._index_summary_minmax("min") is NotImplemented
    assert t["a"].min() == 1

    t.rebuild_index("a")
    assert t["a"]._index_summary_minmax("min") == 1


# ---------------------------------------------------------------------------
# where(): OR over a nullable indexed column
# ---------------------------------------------------------------------------


def or_table(tmp_path, name, spec, sentinel=None):
    """Two int columns whose large values are clustered, so a summary index has
    something to prune and the planner prefers it to a scan.

    The row count is load-bearing and cannot be trimmed much: the planner's
    cost model only prefers the index once the scan it would replace is big
    enough, and measured here the switch happens between 200k and 400k rows.
    Two million keeps a comfortable margin.  It is not expensive despite the
    size -- the values are deliberately repetitive, so the whole table plus
    both indexes come to about 90 KB on disk.
    """
    n = 2_000_000
    idx = np.arange(n)
    a = np.where((idx >= 1000) & (idx < 1100), 5000 + idx, idx % 100).astype(np.int64)
    b = np.where((idx >= 1_500_000) & (idx < 1_500_100), 7000 + idx, idx % 100).astype(np.int64)
    nulls = (idx % 31) == 0

    Row = dataclasses.make_dataclass(
        "OrRow", [("a", int, blosc2.field(spec)), ("b", int, blosc2.field(blosc2.int64()))]
    )
    t = blosc2.CTable(Row, expected_size=n, urlpath=str(tmp_path / name), mode="w")
    if sentinel is None:
        t.extend({"a": np.ma.MaskedArray(a, mask=nulls), "b": b})
    else:
        col = a.copy()
        col[nulls] = sentinel
        t.extend({"a": col, "b": b})
    t.create_index("a", kind="summary")
    t.create_index("b", kind="summary")
    expected = int(np.count_nonzero(((a > 4000) & ~nulls) | (b > 6000)))
    return t, expected


@pytest.mark.parametrize(
    ("label", "spec", "sentinel"),
    [
        ("mask", blosc2.int64(null_storage="mask"), None),
        ("sentinel", blosc2.int64(nullable=True, null_value=-1), -1),
    ],
)
def test_indexed_or_over_a_nullable_column_uses_its_index(label, spec, sentinel, tmp_path, monkeypatch):
    """It used to bail to a full scan; the segment path evaluates the
    (null-aware) predicate, so it is exact without any post-filter."""
    from blosc2.ctable_indexing import _CTableIndexingMixin

    t, expected = or_table(tmp_path, f"or-{label}.b2t", spec, sentinel)

    used = []
    original = _CTableIndexingMixin._try_index_where

    def spy(self, expr):
        result = original(self, expr)
        used.append(result is not None)
        return result

    monkeypatch.setattr(_CTableIndexingMixin, "_try_index_where", spy)

    got = len(t.where("(a > 4000) | (b > 6000)"))
    assert got == expected
    assert used == [True], "the OR should now be answered from the index"


def test_a_row_null_in_one_branch_still_matches_the_other(tmp_path):
    """What a global null post-filter would have got wrong, at small scale."""
    rng = np.random.default_rng(6)
    n = 2000
    a = [None if i % 31 == 0 else int(rng.integers(0, 1000)) for i in range(n)]
    b = [int(rng.integers(0, 1000)) for _ in range(n)]
    Row = dataclasses.make_dataclass(
        "R",
        [
            ("a", int, blosc2.field(blosc2.int64(null_storage="mask"))),
            ("b", int, blosc2.field(blosc2.int64())),
        ],
    )
    t = blosc2.CTable(Row, expected_size=n, urlpath=str(tmp_path / "small.b2t"), mode="w")
    t.extend(list(zip(a, b, strict=True)))
    t.create_index("a", kind="full")
    t.create_index("b", kind="full")

    expected = sum(((av is not None and av > 800) or bv < 200) for av, bv in zip(a, b, strict=True))
    assert len(t.where("(a > 800) | (b < 200)")) == expected


def test_a_pipe_inside_a_string_literal_is_not_an_or():
    """The OR test decides whether a nullable column keeps its index, so it
    parses rather than searching for a character that string data may contain."""
    from blosc2.ctable_indexing import _expression_has_or

    assert _expression_has_or("(a > 1) | (b < 2)")
    assert _expression_has_or("a > 1 or b < 2")
    assert not _expression_has_or("name == 'a|b'")
    assert not _expression_has_or("(a > 1) & (b < 2)")
