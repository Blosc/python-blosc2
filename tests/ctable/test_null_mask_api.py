#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Reading and writing mask-storage nullable columns (Phase 4).

Phase 3 gave the sidecar a place to live; this is where it becomes usable.
``None`` is now the canonical way to write a null into a fixed-width scalar
column -- which could not accept one at all before -- and the null API reads
validity from the sidecar rather than inferring it from the values.

Two consequences are load-bearing and tested here:

* **NaN is a value, not a null** (decision 6).  A mask-backed float column
  follows Arrow: only ``mask=False`` is missing.  This is the whole point of
  moving nullity to a side channel, and it is where mask and sentinel columns
  deliberately diverge.
* **``fillna`` becomes correct when the fill collides with real data**, which
  is impossible to get right under a sentinel.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import blosc2


def annotation_for(spec):
    """An annotation the schema compiler accepts for *spec*.

    ``validate_annotation_matches_spec`` keys off ``spec.python_type`` for
    everything except the two kinds that take ``object``.
    """
    if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp)):
        return object
    return spec.python_type


def row_type(**cols):
    """A dataclass from ``name=spec`` pairs, annotations derived from the specs."""
    return dataclasses.make_dataclass(
        "MaskRow", [(n, annotation_for(spec), blosc2.field(spec)) for n, spec in cols.items()]
    )


def table(rows, capacity=32, **cols):
    t = blosc2.CTable(row_type(**cols), expected_size=capacity)
    if rows:
        t.extend(rows)
    return t


def simple(values, spec=None, capacity=32):
    """A one-column table of *values*, nulls written as ``None``."""
    spec = blosc2.int64(null_storage="mask") if spec is None else spec
    return table([(v,) for v in values], capacity=capacity, a=spec)


# ---------------------------------------------------------------------------
# None is accepted, and nothing else is mistaken for it
# ---------------------------------------------------------------------------

WRITEABLE_SPECS = [
    ("int64", blosc2.int64(null_storage="mask"), 7),
    ("int8", blosc2.int8(null_storage="mask"), -128),
    ("uint8", blosc2.uint8(null_storage="mask"), 255),
    ("float64", blosc2.float64(null_storage="mask"), 1.5),
    ("complex128", blosc2.complex128(null_storage="mask"), 1 + 2j),
    ("bool", blosc2.bool(null_storage="mask"), True),
    ("string", blosc2.string(max_length=4, null_storage="mask"), "abcd"),
    ("bytes", blosc2.bytes(max_length=4, null_storage="mask"), b"abcd"),
    ("utf8", blosc2.utf8(null_storage="mask"), "hello"),
]


@pytest.mark.parametrize(("label", "spec", "value"), WRITEABLE_SPECS)
def test_extend_accepts_none(label, spec, value):
    t = simple([value, None, value], spec=spec)
    assert t["a"].is_null().tolist() == [False, True, False]
    assert t["a"].null_count() == 1


@pytest.mark.parametrize(("label", "spec", "value"), WRITEABLE_SPECS)
def test_append_accepts_none(label, spec, value):
    t = simple([value], spec=spec)
    t.append((None,))
    assert t["a"].is_null().tolist() == [False, True]


@pytest.mark.parametrize(("label", "spec", "value"), WRITEABLE_SPECS)
def test_setitem_accepts_none(label, spec, value):
    t = simple([value, value], spec=spec)
    t["a"][1] = None
    assert t["a"].is_null().tolist() == [False, True]


@pytest.mark.parametrize(("label", "spec", "value"), WRITEABLE_SPECS)
def test_assign_accepts_none(label, spec, value):
    t = simple([value, value], spec=spec)
    t["a"].assign([None, value])
    assert t["a"].is_null().tolist() == [True, False]


def test_nullable_bool_is_a_real_bool():
    """The case that motivated the whole design: no uint8, no reserved 255."""
    t = simple([True, None, False], spec=blosc2.bool(null_storage="mask"))
    assert t["a"].dtype == np.dtype(np.bool_)
    assert t["a"][:].tolist() == [True, False, False]
    assert t["a"].is_null().tolist() == [False, True, False]


def test_int8_keeps_its_full_range_alongside_nulls():
    values = list(range(-128, 128))
    t = simple([*values, None], spec=blosc2.int8(null_storage="mask"), capacity=300)
    assert t["a"].dtype == np.dtype(np.int8)
    assert t["a"][:-1].tolist() == values
    assert t["a"].is_null()[-1]
    assert t["a"].null_count() == 1


def test_string_keeps_its_declared_width():
    """No sentinel means no max_length widening to fit one."""
    t = simple(["abcd", None], spec=blosc2.string(max_length=4, null_storage="mask"))
    assert t["a"].dtype == np.dtype("U4")


def test_utf8_accepts_text_no_sentinel_could_survive():
    tricky = ["", "\x00", "__BLOSC2_NULL__", "🎉x"]
    t = simple([*tricky, None], spec=blosc2.utf8(null_storage="mask"))
    assert list(t["a"][:-1]) == tricky
    assert t["a"].is_null().tolist() == [False] * 4 + [True]


def test_timestamp_accepts_none():
    spec = blosc2.timestamp(null_storage="mask")
    t = simple(["2020-01-01", None, "2020-01-03"], spec=spec)
    assert t["a"].is_null().tolist() == [False, True, False]
    assert np.isnat(t["a"][:][1])


def test_ndarray_column_accepts_none():
    spec = blosc2.ndarray((3,), dtype=blosc2.float32(), null_storage="mask")
    item = np.ones(3, dtype=np.float32)
    t = simple([item, None, item * 2], spec=spec)
    assert t["a"].is_null().tolist() == [False, True, False]
    assert t["a"][:][0].tolist() == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Decision 6: NaN is a value
# ---------------------------------------------------------------------------


def test_nan_is_a_value_not_a_null():
    t = simple([1.0, float("nan"), None], spec=blosc2.float64(null_storage="mask"))
    assert t["a"].is_null().tolist() == [False, False, True]
    assert t["a"].null_count() == 1
    assert np.isnan(t["a"][:][1])


def test_sentinel_float_still_treats_nan_as_null():
    """The two storages diverge here on purpose, and both are documented."""
    t = simple([1.0, float("nan")], spec=blosc2.float64(null_value=float("nan")))
    assert t["a"].is_null().tolist() == [False, True]


def test_signed_zero_and_inf_are_values():
    values = [0.0, -0.0, float("inf"), float("-inf")]
    t = simple([*values, None], spec=blosc2.float64(null_storage="mask"))
    assert t["a"].is_null().tolist() == [False] * 4 + [True]
    assert np.array_equal(t["a"][:-1], np.array(values), equal_nan=True)


def test_int64_min_is_a_value_in_a_timestamp_column():
    spec = blosc2.timestamp(null_storage="mask")
    t = simple([np.datetime64(np.iinfo(np.int64).min + 1, "us"), None], spec=spec)
    assert t["a"].is_null().tolist() == [False, True]


# ---------------------------------------------------------------------------
# Input forms that carry their own validity
# ---------------------------------------------------------------------------


def test_masked_array_input_supplies_validity_verbatim():
    spec = blosc2.float64(null_storage="mask")
    data = np.ma.MaskedArray([1.0, 2.0, np.nan], mask=[False, True, False])
    t = blosc2.CTable(row_type(a=spec), expected_size=8)
    t.extend({"a": data})
    # The NaN is *not* masked, so it stays a value; only index 1 is null.
    assert t["a"].is_null().tolist() == [False, True, False]


def test_numpy_float_array_input_has_no_nulls():
    spec = blosc2.float64(null_storage="mask")
    t = blosc2.CTable(row_type(a=spec), expected_size=8)
    t.extend({"a": np.array([1.0, np.nan, 3.0])})
    assert t["a"].null_count() == 0
    assert t["a"].is_null().tolist() == [False, False, False]


def test_nat_reads_as_null_in_a_timestamp_column():
    spec = blosc2.timestamp(null_storage="mask")
    t = simple([np.datetime64("2020-01-01"), np.datetime64("NaT")], spec=spec)
    assert t["a"].is_null().tolist() == [False, True]


# ---------------------------------------------------------------------------
# Decision 9 through the public API
# ---------------------------------------------------------------------------


def test_a_null_free_batch_writes_no_sidecar():
    t = simple([1, 2, 3])
    assert t._null_mask("a") is None
    assert t["a"].null_count() == 0
    assert t["a"].is_null().tolist() == [False, False, False]


def test_the_sidecar_appears_on_the_first_null():
    t = simple([1, 2, 3])
    assert t._null_mask("a") is None
    t.append((None,))
    assert t._null_mask("a") is not None


def test_marking_a_row_valid_does_not_create_a_sidecar():
    t = simple([1, 2, 3])
    t["a"][0] = 5
    assert t._null_mask("a") is None


def test_info_reports_storage_and_whether_a_sidecar_exists():
    t = simple([1, 2, 3])
    items = dict(t["a"].info_items)
    assert items["nullable"] is True
    assert items["null_storage"] == "mask"
    assert items["null_sidecar"] is False
    t.append((None,))
    assert dict(t["a"].info_items)["null_sidecar"] is True


def test_non_nullable_column_reports_no_storage():
    t = table([(1,)], a=blosc2.int64())
    assert "null_storage" not in dict(t["a"].info_items)


# ---------------------------------------------------------------------------
# Overwrite semantics: a write replaces validity, it does not merge it
# ---------------------------------------------------------------------------


def test_assign_clears_previously_written_nulls():
    t = simple([1, None, 3])
    t["a"].assign([1, 2, 3])
    assert t["a"].null_count() == 0
    assert t["a"][:].tolist() == [1, 2, 3]


def test_setitem_clears_a_null():
    t = simple([1, None, 3])
    t["a"][1] = 2
    assert t["a"].null_count() == 0
    assert t["a"][:].tolist() == [1, 2, 3]


def test_setitem_slice_replaces_validity():
    t = simple([1, None, 3, None])
    t["a"][0:3] = [None, 2, 3]
    assert t["a"].is_null().tolist() == [True, False, False, True]


def test_setitem_boolean_mask_replaces_validity():
    t = simple([1, None, 3, 4])
    t["a"][np.array([True, True, False, False])] = [None, 2]
    assert t["a"].is_null().tolist() == [True, False, False, False]


def test_setitem_index_list_replaces_validity():
    t = simple([1, None, 3, 4])
    t["a"][[1, 3]] = [2, None]
    assert t["a"].is_null().tolist() == [False, False, False, True]


# ---------------------------------------------------------------------------
# The null API
# ---------------------------------------------------------------------------


def test_notnull_is_the_complement_of_is_null():
    t = simple([1, None, 3])
    assert (t["a"].notnull() == ~t["a"].is_null()).all()


def test_fillna_is_correct_when_the_fill_collides_with_real_data():
    """Impossible under a sentinel: there, 7 *is* how a null would look."""
    t = simple([7, None, 3])
    assert t["a"].fillna(7).tolist() == [7, 7, 3]
    # ...and the column itself is unchanged, still one null.
    assert t["a"].null_count() == 1


def test_to_numpy_masked_marks_only_the_nulls():
    t = simple([1.0, float("nan"), None], spec=blosc2.float64(null_storage="mask"))
    out = t["a"].to_numpy(masked=True)
    assert isinstance(out, np.ma.MaskedArray)
    assert out.mask.tolist() == [False, False, True]


def test_to_numpy_masked_works_for_sentinel_columns_too():
    """One uniform way to ask for values-plus-validity, whatever the storage."""
    t = simple([1, -1, 3], spec=blosc2.int64(null_value=-1))
    out = t["a"].to_numpy(masked=True)
    assert out.mask.tolist() == [False, True, False]


def test_to_numpy_without_masked_is_a_plain_array():
    t = simple([1, None, 3])
    assert isinstance(t["a"].to_numpy(), np.ndarray)
    assert not isinstance(t["a"].to_numpy(), np.ma.MaskedArray)


def test_unique_and_value_counts_skip_nulls():
    t = simple([1, None, 3, 1])
    assert t["a"].unique().tolist() == [1, 3]
    assert t["a"].value_counts() == {1: 2, 3: 1}


def test_dropna_uses_the_sidecar():
    t = table(
        [(1, 10), (None, 20), (3, 30)],
        a=blosc2.int64(null_storage="mask"),
        b=blosc2.int64(),
    )
    assert t.dropna()["b"][:].tolist() == [10, 30]


def test_reductions_skip_nulls():
    t = simple([2.0, None, 4.0], spec=blosc2.float64(null_storage="mask"))
    assert t["a"].sum() == 6.0
    assert t["a"].mean() == 3.0


def test_null_count_with_deletions():
    t = simple([None, 1, None, 3])
    t.delete(0)
    assert t["a"].null_count() == 1
    assert t["a"].is_null().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Views keep their nulls
# ---------------------------------------------------------------------------


def two_col_table():
    return table(
        [(None if i in (1, 4) else i, 10 - i) for i in range(6)],
        a=blosc2.int64(null_storage="mask"),
        b=blosc2.int64(),
    )


def test_where_view_remaps_nulls():
    t = two_col_table()
    assert t.where("b < 8")["a"].is_null().tolist() == [False, True, False]


def test_slice_view_remaps_nulls():
    assert two_col_table()[1:5]["a"].is_null().tolist() == [True, False, False, True]


def test_reversed_slice_view_remaps_nulls():
    assert two_col_table()[::-1]["a"].is_null().tolist() == [False, True, False, False, True, False]


@pytest.mark.parametrize("view", [True, False])
def test_sort_by_keeps_nulls_with_their_rows(view):
    sorted_t = two_col_table().sort_by("b", view=view)
    assert sorted_t["a"].is_null().tolist() == [False, True, False, False, True, False]
    assert sorted_t["b"][:].tolist() == [5, 6, 7, 8, 9, 10]


def test_sort_by_inplace_keeps_nulls_with_their_rows():
    t = two_col_table()
    t.sort_by("b", inplace=True)
    assert t["a"].is_null().tolist() == [False, True, False, False, True, False]


def test_sorting_a_view_keeps_nulls_with_their_rows():
    t = two_col_table()
    sorted_view = t.where("b > 5").sort_by("b")
    assert sorted_view["a"].is_null().tolist() == [True, False, False, True, False]


def test_take_keeps_nulls_with_their_rows():
    assert two_col_table().take([4, 1, 0])["a"].is_null().tolist() == [True, True, False]


def test_slice_copy_keeps_nulls_with_their_rows():
    assert two_col_table().slice(1, 5)["a"].is_null().tolist() == [True, False, False, True]


@pytest.mark.parametrize("compact", [True, False])
def test_copy_keeps_nulls(compact):
    assert two_col_table().copy(compact=compact)["a"].null_count() == 2


def test_compact_keeps_nulls_with_their_rows():
    t = two_col_table()
    t.delete(0)
    t.compact()
    assert t["a"].is_null().tolist() == [True, False, False, True, False]


def test_a_view_of_a_null_free_column_stays_sidecar_free():
    t = simple([1, 2, 3, 4])
    assert t.take([0, 2])._null_mask("a") is None


def test_gathering_only_valid_rows_needs_no_sidecar():
    """Decision 9 again: the copy has no nulls, so it gets no bytes."""
    t = two_col_table()
    assert t.take([0, 2, 3])._null_mask("a") is None


# ---------------------------------------------------------------------------
# Persistence of nulls written through the API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".b2d", ".b2z"])
def test_nulls_survive_save_and_reopen(tmp_path, suffix):
    path = str(tmp_path / f"t{suffix}")
    two_col_table().save(path)
    reopened = blosc2.CTable.open(path)
    try:
        assert reopened["a"].is_null().tolist() == [False, True, False, False, True, False]
        assert reopened["a"].null_count() == 2
    finally:
        reopened.close()


def test_nulls_survive_a_cframe_round_trip():
    rebuilt = blosc2.ctable_from_cframe(two_col_table().to_cframe())
    assert rebuilt["a"].is_null().tolist() == [False, True, False, False, True, False]


def test_appending_to_a_reopened_table_keeps_earlier_nulls(tmp_path):
    path = str(tmp_path / "grow.b2d")
    two_col_table().save(path)
    reopened = blosc2.CTable.open(path, mode="a")
    try:
        reopened.append((None, 99))
        assert reopened["a"].is_null().tolist() == [
            False, True, False, False, True, False, True,
        ]  # fmt: skip
    finally:
        reopened.close()


# ---------------------------------------------------------------------------
# Sentinel columns are untouched
# ---------------------------------------------------------------------------


def test_sentinel_columns_keep_their_behaviour():
    t = simple([1, -1, 3], spec=blosc2.int64(null_value=-1))
    assert t["a"].null_storage == "sentinel"
    assert t["a"].is_null().tolist() == [False, True, False]
    assert t._null_mask("a") is None


def test_sentinel_bool_is_still_uint8_backed():
    t = simple([True, 255, False], spec=blosc2.bool(nullable=True, null_value=255))
    assert t["a"].dtype == np.dtype(np.uint8)
    assert t["a"].is_null().tolist() == [False, True, False]


def test_channel_is_not_cached_on_its_column():
    """A cached channel would close a Column-CTable reference cycle."""
    col = simple([1, 2])["a"]
    assert col._nulls is not col._nulls
    assert col._nulls.kind == "mask"


def test_a_table_is_freed_without_a_gc_pass():
    """The regression the no-caching rule above exists to prevent."""
    import gc
    import weakref

    def build():
        return weakref.ref(simple([1, None, 3]))

    gc.disable()
    try:
        ref = build()
        assert ref() is None
    finally:
        gc.enable()
