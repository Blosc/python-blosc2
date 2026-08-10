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
import datetime

import numpy as np
import pytest
from utf8_compat import needs_utf8, utf8_spec

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
    pytest.param("utf8", utf8_spec(null_storage="mask"), "hello", marks=needs_utf8),
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


@needs_utf8
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
    # The NaT carries a unit: NumPy 2.5 deprecated the *generic* one, so a bare
    # np.datetime64("NaT") warns in the caller.  Taking the unit from the spec
    # keeps this pinned to the column rather than to a literal.
    spec = blosc2.timestamp(null_storage="mask")
    t = simple([np.datetime64("2020-01-01"), np.datetime64("NaT", spec.unit)], spec=spec)
    assert t["a"].is_null().tolist() == [False, True]


def test_a_nat_of_any_unit_reads_as_null():
    """Detection is np.isnat, so it does not care which unit the caller used."""
    spec = blosc2.timestamp(null_storage="mask")
    for unit in ("s", "ms", "us", "ns"):
        t = simple([np.datetime64("2020-01-01"), np.datetime64("NaT", unit)], spec=spec)
        assert t["a"].is_null().tolist() == [False, True], unit


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
# One value for many rows.  Assigning a scalar to a selection has always
# broadcast it, the way NumPy does, and mask storage must not take that away --
# it is the default now, so ``t.a[i:j] = value`` on a plain nullable column
# goes through here.  ``None`` broadcasts the same way, which is the mask
# spelling of "make these rows null".
# ---------------------------------------------------------------------------


def keys_for(t):
    """The three multi-row key forms, each selecting rows 0 and 1."""
    return [slice(0, 2), np.array([True, True, False]), [0, 1]]


@pytest.mark.parametrize("key_index", [0, 1, 2], ids=["slice", "bool_mask", "index_list"])
def test_setitem_broadcasts_a_scalar_over_every_key_form(key_index):
    t = simple([1, 2, 3])
    t["a"][keys_for(t)[key_index]] = 7
    assert t["a"][:].tolist() == [7, 7, 3]
    assert t["a"].is_null().tolist() == [False, False, False]


@pytest.mark.parametrize("key_index", [0, 1, 2], ids=["slice", "bool_mask", "index_list"])
def test_setitem_broadcasts_none_over_every_key_form(key_index):
    t = simple([1, 2, 3])
    t["a"][keys_for(t)[key_index]] = None
    assert t["a"].is_null().tolist() == [True, True, False]


@pytest.mark.parametrize(("label", "spec", "value"), WRITEABLE_SPECS)
def test_setitem_broadcasts_one_cell_for_every_kind(label, spec, value):
    t = table([(value,), (value,), (value,)], a=spec)
    if label == "utf8":
        # A varlen column has never accepted a broadcast value -- it says so in
        # its own words, and a sentinel utf8 column says exactly the same.
        with pytest.raises(ValueError, match="Length mismatch"):
            t["a"][0:2] = value
    else:
        t["a"][0:2] = value
        assert t["a"].is_null().tolist() == [False, False, False]
    t["a"][0:2] = None
    assert t["a"].is_null().tolist() == [True, True, False]


def test_broadcasting_a_scalar_matches_sentinel_storage():
    """The two storages agree on a broadcast write, value and validity alike."""
    mask = simple([1, 2, 3])
    sentinel = simple([1, 2, 3], spec=blosc2.int64(null_storage="sentinel", null_value=-9))
    mask["a"][0:2] = 7
    sentinel["a"][0:2] = 7
    assert mask["a"][:].tolist() == sentinel["a"][:].tolist()
    assert mask["a"].is_null().tolist() == sentinel["a"].is_null().tolist()


def test_broadcasting_none_clears_a_previously_written_value():
    """A broadcast null replaces validity rather than merging with it."""
    t = simple([1, None, 3])
    t["a"][0:3] = None
    assert t["a"].null_count() == 3
    t["a"][0:3] = 5
    assert t["a"].null_count() == 0
    assert t["a"][:].tolist() == [5, 5, 5]


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


def ordered_pair(storage):
    """A table whose sorted order differs from its physical order.

    The key column is deliberately shuffled, so a reduction that reads nullity
    in one order and values in the other pairs the wrong rows.
    """
    spec = blosc2.float64(null_storage=storage)
    return table(
        list(zip([5, 1, 4, 2, 3], [10.0, None, 30.0, None, 50.0], strict=True)),
        k=blosc2.int64(),
        v=spec,
    )


@pytest.mark.parametrize("storage", ["mask", "sentinel"])
def test_reductions_on_an_ordered_view_pair_the_right_rows(storage):
    """Values arrive physically ordered, so the null flags must too.

    ``iter_chunks`` walks the validity array chunk by chunk and never consults a
    view's ordering, while ``null_mask()`` answers in view order.  Reading the
    second against the first dropped live values and let the fill through as
    data -- ``sum()`` came back ``nan`` from a column whose nulls are not NaN.
    """
    t = ordered_pair(storage)
    view = t.sort_by("k", view=True)
    assert view["v"].sum() == 90.0
    assert sorted(view["v"].unique().tolist()) == [10.0, 30.0, 50.0]
    assert view["v"].mean() == 30.0


@pytest.mark.parametrize("storage", ["mask", "sentinel"])
def test_the_two_storages_reduce_an_ordered_view_alike(storage):
    """The differential that would have caught this: mask must match sentinel."""
    view = ordered_pair(storage).sort_by("k", view=True)
    reference = ordered_pair("sentinel").sort_by("k", view=True)
    assert view["v"].sum() == reference["v"].sum()
    assert sorted(view["v"].unique().tolist()) == sorted(reference["v"].unique().tolist())


def test_reductions_on_a_descending_view_pair_the_right_rows():
    view = ordered_pair("mask").sort_by("k", ascending=False, view=True)
    assert view["v"].sum() == 90.0


def test_reductions_on_an_ordered_view_with_deletions():
    """Deletions and ordering at once: both streams read the same _valid_rows."""
    t = ordered_pair("mask")
    t.delete(0)  # drops the row holding 10.0
    assert t.sort_by("k", view=True)["v"].sum() == 80.0


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


# ---------------------------------------------------------------------------
# Extending from another CTable
# ---------------------------------------------------------------------------
#
# A CTable source is the one input shape whose nulls do not travel with its
# values: under mask storage nullity lives in a sidecar, so copying the raw
# column alone turns every null into its fill -- a plausible-looking 0 or "",
# not an error.  The sentinel and list-of-rows forms below are the controls
# that say what the answer has to be.


def test_extend_from_a_table_carries_the_nulls_over():
    src = simple([1, None, 3])
    dst = simple([])
    dst.extend(src)
    assert dst["a"][:].tolist() == [1, 0, 3]
    assert dst["a"].is_null().tolist() == [False, True, False]
    assert dst["a"].null_count() == 1


def test_extend_from_a_table_agrees_with_the_other_input_shapes():
    """The same rows, spelled three ways, must land identically."""
    from_table = simple([])
    from_table.extend(simple([1, None, 3]))

    from_rows = simple([])
    from_rows.extend([(1,), (None,), (3,)])

    sentinel = table([], a=blosc2.int64(nullable=True, null_value=-9))
    sentinel.extend(table([(1,), (-9,), (3,)], a=blosc2.int64(nullable=True, null_value=-9)))

    assert from_table["a"].is_null().tolist() == from_rows["a"].is_null().tolist()
    assert from_table["a"].is_null().tolist() == sentinel["a"].is_null().tolist()


@needs_utf8
def test_extend_from_a_table_carries_utf8_nulls():
    """The fill is "" here, which is also a legal value -- so it has to be the sidecar."""
    src = table([("x",), (None,), ("",)], s=utf8_spec(null_storage="mask"))
    dst = table([], s=utf8_spec(null_storage="mask"))
    dst.extend(src)
    assert list(dst["s"][:]) == ["x", "", ""]
    assert dst["s"].is_null().tolist() == [False, True, False]


def test_extend_from_a_table_with_deleted_rows_copies_the_live_ones():
    """Live rows scatter through the physical extent, so a leading slice is the wrong rows."""
    src = simple([10, 11, 12, 13])
    src.delete(0)
    dst = simple([])
    dst.extend(src)
    assert dst["a"][:].tolist() == [11, 12, 13]
    assert dst.nrows == 3


def test_extend_from_a_table_with_holes_keeps_values_and_nulls_aligned():
    src = simple([10, None, 12, None, 14])
    src.delete(0)
    dst = simple([])
    dst.extend(src)
    assert dst["a"].is_null().tolist() == src["a"].is_null().tolist()
    assert dst["a"][:].tolist() == src["a"][:].tolist()


def test_extend_from_a_sorted_view_copies_it_in_sorted_order():
    src = simple([3, None, 1, 2])
    dst = simple([])
    dst.extend(src.sort_by("a", view=True))
    # Nulls sort last, in both directions, and the copy has to agree.
    assert dst["a"][:].tolist() == [1, 2, 3, 0]
    assert dst["a"].is_null().tolist() == [False, False, False, True]


def test_extend_from_a_null_free_table_writes_no_sidecar():
    """Decision 9: an absent sidecar is the common state and must stay absent."""
    dst = simple([])
    dst.extend(simple([1, 2, 3]))
    assert dst["a"].is_null().tolist() == [False, False, False]
    assert dst._null_mask("a") is None


# ---------------------------------------------------------------------------
# add_column
# ---------------------------------------------------------------------------
#
# A mask column is the only kind that can say "this row has no value" without
# reserving one, so it is the only way to add a column to a populated table
# and be honest about the rows that predate it.  A sentinel column has to
# spell that with its sentinel, and says so.


def test_add_column_values_accept_none():
    t = simple([1, 2, 3], spec=blosc2.int64())
    t.add_column("b", blosc2.int64(nullable=True), values=[10, None, 30])
    assert t["b"][:].tolist() == [10, 0, 30]
    assert t["b"].is_null().tolist() == [False, True, False]
    assert t["b"].null_count() == 1


def test_add_column_default_none_backfills_nulls():
    """The rows that predate the column have no value, and now can say so."""
    t = simple([1, 2, 3], spec=blosc2.int64())
    t.add_column("b", blosc2.field(blosc2.int64(nullable=True), default=None))
    assert t["b"].is_null().tolist() == [True, True, True]


def test_add_column_default_none_still_applies_to_later_rows():
    t = simple([1, 2], spec=blosc2.int64())
    t.add_column("b", blosc2.field(blosc2.int64(nullable=True), default=None))
    t.append((3, 9))
    t.extend([(4, None)])
    assert t["b"][:].tolist() == [0, 0, 9, 0]
    assert t["b"].is_null().tolist() == [True, True, False, True]


def test_add_column_without_a_null_writes_no_sidecar():
    """Decision 9 again: the lazy sidecar must stay lazy here too."""
    t = simple([1, 2, 3], spec=blosc2.int64())
    t.add_column("b", blosc2.int64(nullable=True), values=[10, 20, 30])
    assert t["b"].is_null().tolist() == [False, False, False]
    assert t._null_mask("b") is None


def test_add_column_scatters_nulls_past_deleted_rows():
    """values= is one entry per *live* row, so validity has to scatter with them."""
    t = simple([0, 1, 2, 3], spec=blosc2.int64())
    t.delete(1)
    t.add_column("b", blosc2.int64(nullable=True), values=[10, None, 30])
    assert t["b"].is_null().tolist() == [False, True, False]
    t.compact()
    assert t["b"].is_null().tolist() == [False, True, False]
    assert t["b"][:].tolist() == [10, 0, 30]


@needs_utf8
def test_add_column_utf8_values_accept_none():
    """The fill is "" here, which a genuine row may also hold."""
    t = simple([1, 2, 3], spec=blosc2.int64())
    t.add_column("s", utf8_spec(null_storage="mask"), values=["p", None, ""])
    assert list(t["s"][:]) == ["p", "", ""]
    assert t["s"].is_null().tolist() == [False, True, False]


def test_add_column_ndarray_values_accept_none():
    t = simple([1, 2], spec=blosc2.int64())
    t.add_column(
        "v",
        blosc2.ndarray((3,), dtype=blosc2.int64(), nullable=True),
        values=[np.array([1, 2, 3]), None],
    )
    assert t["v"][:].tolist() == [[1, 2, 3], [0, 0, 0]]
    assert t["v"].is_null().tolist() == [False, True]


def test_add_column_nulls_survive_a_reopen(tmp_path):
    t = simple([1, 2, 3], spec=blosc2.int64())
    urlpath = str(tmp_path / "added.b2t")
    t.save(urlpath)
    live = blosc2.CTable.open(urlpath, mode="a")
    try:
        live.add_column("b", blosc2.int64(nullable=True), values=[10, None, 30])
    finally:
        live.close()
    reopened = blosc2.CTable.open(urlpath)
    try:
        assert reopened["b"].null_storage == "mask"
        assert reopened["b"].is_null().tolist() == [False, True, False]
    finally:
        reopened.close()


@pytest.mark.parametrize(
    ("spec", "match"),
    [
        (blosc2.int64(), "nullable"),
        (blosc2.int64(nullable=True, null_value=-1), r"null_value \(-1\)"),
    ],
)
def test_add_column_says_why_a_null_will_not_fit(spec, match):
    """Not NumPy's "int() argument must be ...", which names neither the column nor the way out."""
    t = simple([1, 2, 3], spec=blosc2.int64())
    with pytest.raises(TypeError, match=match):
        t.add_column("b", spec, values=[10, None, 30])


# A timestamp column stores int64 in the spec's unit, so add_column has to
# convert through that unit exactly as extend does.  It did not, which put a
# datetime64[s] value in a microsecond column and read it back as 1970.


@pytest.mark.parametrize("nullable", [False, True])
def test_add_column_timestamps_keep_their_unit(nullable):
    when = np.datetime64("2020-01-01T00:00:00", "s")
    t = simple([1, 2], spec=blosc2.int64())
    t.add_column("ts", blosc2.timestamp(nullable=nullable), values=[when, when])
    assert t["ts"][:].tolist() == [when.astype("datetime64[us]").item()] * 2


def test_add_column_timestamp_default_keeps_its_unit():
    when = np.datetime64("2020-01-01T00:00:00", "s")
    t = simple([1, 2], spec=blosc2.int64())
    t.add_column("ts", blosc2.field(blosc2.timestamp(), default=when))
    assert t["ts"][:].tolist() == [when.astype("datetime64[us]").item()] * 2


def test_add_column_timestamp_null_reads_as_nat():
    when = np.datetime64("2020-01-01T00:00:00", "s")
    t = simple([1, 2], spec=blosc2.int64())
    t.add_column("ts", blosc2.timestamp(nullable=True), values=[when, None])
    assert t["ts"].is_null().tolist() == [False, True]
    assert np.isnat(t["ts"][:][1])


# ---------------------------------------------------------------------------
# isin
# ---------------------------------------------------------------------------
#
# Membership is asked of the row's *value*, and a null row has none.  Testing
# the raw values instead matched whatever stands in for a null -- the fill
# here, a sentinel elsewhere -- neither of which the column ever means
# literally.  Cross-storage agreement is pinned in
# test_null_storage_equivalence.py; these are the mask-only corners.


def test_isin_does_not_match_the_fill():
    t = simple([1, None, 0])
    # Rows 1 and 2 both hold a physical 0; only row 2 holds it as a value.
    assert t["a"][:].tolist() == [1, 0, 0]
    assert t["a"].isin([0]).tolist() == [False, False, True]


def test_isin_none_selects_the_nulls():
    t = simple([1, None, 3, None])
    assert t["a"].isin([None]).tolist() == t["a"].is_null().tolist()


def test_isin_none_can_be_combined_with_values():
    t = simple([1, None, 3])
    assert t["a"].isin([3, None]).tolist() == [False, True, True]


def test_isin_pandas_na_spells_the_same_request():
    pd = pytest.importorskip("pandas")
    t = simple([1, None, 3])
    assert t["a"].isin([pd.NA]).tolist() == [False, True, False]


@needs_utf8
def test_isin_does_not_match_the_empty_string_fill():
    """The utf8 fill is "", which a genuine row may hold -- so it has to be the sidecar."""
    t = table([("x",), (None,), ("",)], s=utf8_spec(null_storage="mask"))
    assert list(t["s"][:]) == ["x", "", ""]
    assert t["s"].isin([""]).tolist() == [False, False, True]


def test_isin_on_a_view_uses_the_views_rows():
    t = simple([1, None, 3, None, 5])
    assert t.sort_by("a", view=True)["a"].isin([None]).tolist() == [False, False, False, True, True]
    assert t.take([1, 2])["a"].isin([None]).tolist() == [True, False]


def test_isin_keeps_nan_a_value():
    """Decision 6: only mask=False is missing, so a NaN row is not a null row."""
    t = simple([1.0, float("nan"), None], spec=blosc2.float64(null_storage="mask"))
    assert t["a"].is_null().tolist() == [False, False, True]
    assert t["a"].isin([None]).tolist() == [False, False, True]


def test_isin_on_an_empty_column():
    assert simple([])["a"].isin([1, None]).tolist() == []


# ---------------------------------------------------------------------------
# Writing to a timestamp column, which never worked through __setitem__
# ---------------------------------------------------------------------------


def timestamp_table(storage="mask"):
    spec = blosc2.timestamp(null_storage=storage, nullable=True)
    return table([(datetime.datetime(2020, 1, 1),)] * 4, ts=spec)


@pytest.mark.parametrize("storage", ["mask", "sentinel"])
@pytest.mark.parametrize(
    ("label", "key", "value"),
    [
        ("single", 0, datetime.datetime(2021, 1, 1)),
        ("slice_batch", slice(1, 3), [datetime.datetime(2023, 1, 1), datetime.datetime(2024, 1, 1)]),
        ("broadcast", slice(0, 2), datetime.datetime(2025, 6, 1)),
        ("bool_mask", np.array([False, False, False, True]), datetime.datetime(2026, 1, 1)),
        ("iso_string", 0, "2027-03-04"),
        ("datetime64", 1, np.datetime64("2028-05-06")),
    ],
)
def test_setitem_accepts_datetimes_for_every_key_form(storage, label, key, value):
    """A datetime had to be encoded to the stored int64, and nothing did it.

    ``extend`` gets this from the schema validators, but ``__setitem__`` wrote
    what it was handed straight to the NDArray -- and NumPy has no conversion
    from a Python ``datetime`` to an ``int64`` buffer to fall back on, so every
    key form failed under both storages.
    """
    t = timestamp_table(storage)
    t["ts"][key] = value
    assert t["ts"].null_count() == 0


def test_a_timestamp_batch_may_mix_values_and_nulls():
    t = timestamp_table()
    t["ts"][0:3] = [datetime.datetime(2030, 1, 1), None, datetime.datetime(2031, 1, 1)]
    assert t["ts"].is_null().tolist() == [False, True, False, False]


def test_broadcasting_none_over_a_timestamp_column():
    t = timestamp_table()
    t["ts"][0:2] = None
    assert t["ts"].is_null().tolist() == [True, True, False, False]


# ---------------------------------------------------------------------------
# assign() takes one value per row, and says so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [5, None])
def test_assign_rejects_a_single_value_with_the_documented_error(value):
    """``len()`` of a 0-d array raises TypeError, where assign documents a
    ValueError naming the count -- and points at the spelling that does work."""
    t = simple([1, 2, 3])
    with pytest.raises(ValueError, match=r"requires 3 values.*single value"):
        t["a"].assign(value)


# ---------------------------------------------------------------------------
# extend() from another table has to *translate* nullity, not copy it
# ---------------------------------------------------------------------------

CROSS_STORAGE_KINDS = [
    ("int64", blosc2.int64, int, [1, None, 3]),
    ("int8", blosc2.int8, int, [5, None, -3]),
    ("float64", blosc2.float64, float, [1.0, None, 3.0]),
    ("bool", blosc2.bool, bool, [True, None, False]),
    ("string", lambda **k: blosc2.string(max_length=4, **k), str, ["x", None, "z"]),
    ("bytes", lambda **k: blosc2.bytes(max_length=4, **k), bytes, [b"x", None, b"z"]),
]


def one_col_table(factory, annotation, values, storage):
    """A one-column table holding *values*, with ``None`` meaning null.

    A sentinel column cannot be handed a ``None`` -- there you write the
    reserved value yourself -- so it is substituted here, which is exactly the
    asymmetry the translation under test has to bridge.
    """
    Row = dataclasses.make_dataclass(
        "XRow", [("n", annotation, blosc2.field(factory(nullable=True, null_storage=storage)))]
    )
    t = blosc2.CTable(Row, expected_size=16)
    if values:
        if storage == "sentinel":
            sentinel = t["n"].null_value
            values = [sentinel if v is None else v for v in values]
        t.extend([(v,) for v in values])
    return t


@pytest.mark.parametrize(("label", "factory", "annotation", "values"), CROSS_STORAGE_KINDS)
@pytest.mark.parametrize("src_storage", ["mask", "sentinel"])
@pytest.mark.parametrize("dst_storage", ["mask", "sentinel"])
def test_extend_between_storages_keeps_the_nulls(
    label, factory, annotation, values, src_storage, dst_storage
):
    """What stands in for a null differs per column, so it has to be translated.

    A raw copy carries the source's stand-in -- a mask column's fill, or a
    sentinel column's reserved value -- into the destination as ordinary data.
    Both directions lost the null silently: mask to sentinel wrote a real ``0``,
    and sentinel to mask wrote the reserved ``int64`` minimum.
    """
    source = one_col_table(factory, annotation, values, src_storage)
    dest = one_col_table(factory, annotation, [], dst_storage)
    dest.extend(source)

    assert dest["n"].is_null().tolist() == [v is None for v in values]
    kept = [None if dest["n"].is_null()[i] else dest["n"][i] for i in range(len(dest))]
    assert all(a == b for a, b in zip(kept, values, strict=True) if b is not None)


def test_extend_between_two_different_sentinels_translates_the_value():
    """Same storage is not the same spelling: each column reserves its own."""
    left = dataclasses.make_dataclass(
        "L", [("n", int, blosc2.field(blosc2.int64(nullable=True, null_value=-1)))]
    )
    right = dataclasses.make_dataclass(
        "R", [("n", int, blosc2.field(blosc2.int64(nullable=True, null_value=-9)))]
    )
    src = blosc2.CTable(left, expected_size=8)
    src.extend([(1,), (-1,), (3,)])
    dest = blosc2.CTable(right, expected_size=8)
    dest.extend(src)

    assert dest["n"].is_null().tolist() == [False, True, False]
    assert dest["n"][1] == -9  # the destination's own reserved value, not -1


def test_extend_to_a_narrower_text_column_does_not_truncate_a_sentinel():
    """A sentinel column is widened to fit its sentinel; the source may not be.

    Writing ``__BLOSC2_NULL__`` into the source's ``U4`` left ``__BL`` behind,
    which reads back as data rather than as a null.
    """
    source = one_col_table(lambda **k: blosc2.string(max_length=4, **k), str, ["x", None, "z"], "mask")
    dest = one_col_table(lambda **k: blosc2.string(max_length=4, **k), str, [], "sentinel")
    dest.extend(source)
    assert dest["n"].is_null().tolist() == [False, True, False]
    assert list(dest["n"][:])[0] == "x"
