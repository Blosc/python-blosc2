#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""``convert_nulls``: moving a column's nulls between channels (Phase 8).

Two guarantees frame everything here.  **Nothing auto-migrates** -- opening,
copying and saving a table all preserve each column's ``null_storage``, so this
is the only thing that changes it.  And a conversion either happens or does not:
every reason one can fail is decided before a byte is written, so a refusal
leaves the table exactly as it was rather than half converted.

The crash-safety ordering is asserted directly, because it is the whole argument
for why an in-place conversion is allowed at all: each intermediate state on disk
has to read correctly under *some* schema, and the schema is only ever the last
thing to move.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from utf8_compat import needs_utf8

import blosc2


def annotation_for(spec):
    if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp)):
        return object
    return spec.python_type


def table(rows, capacity=64, urlpath=None, **cols):
    Row = dataclasses.make_dataclass(
        "MigrateRow", [(n, annotation_for(s), blosc2.field(s)) for n, s in cols.items()]
    )
    kwargs = {"urlpath": str(urlpath), "mode": "w"} if urlpath is not None else {}
    t = blosc2.CTable(Row, expected_size=max(capacity, len(rows)), **kwargs)
    if rows:
        t.extend(rows)
    return t


def one_col(values, spec, capacity=64, urlpath=None):
    return table([(v,) for v in values], capacity=capacity, urlpath=urlpath, a=spec)


def as_list(col):
    """A column's live values with nulls spelled ``None``."""
    null = col.is_null()
    return [None if null[i] else _scalar(col[i]) for i in range(len(col))]


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


#: Every V1 kind, with a value it can hold and the sentinel its policy picks.
V1_KINDS = [
    ("int64", blosc2.int64, {}, [5, None, 1]),
    ("int8", blosc2.int8, {}, [5, None, 1]),
    ("uint8", blosc2.uint8, {}, [5, None, 1]),
    ("float64", blosc2.float64, {}, [1.5, None, 2.5]),
    ("bool", blosc2.bool, {}, [True, None, False]),
    ("string", blosc2.string, {"max_length": 4}, ["ab", None, "cd"]),
    ("bytes", blosc2.bytes, {"max_length": 4}, [b"ab", None, b"cd"]),
    pytest.param("utf8", blosc2.utf8, {}, ["ab", None, "cdefgh"], marks=needs_utf8),
    (
        "timestamp",
        blosc2.timestamp,
        {},
        [np.datetime64("2020-01-01"), None, np.datetime64("2021-06-01")],
    ),
]


# ---------------------------------------------------------------------------
# Round trip, both directions, every kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "factory", "kw", "values"), V1_KINDS)
def test_mask_to_sentinel_to_mask_preserves_the_data(label, factory, kw, values):
    """Both directions, back to back, for each kind that has two channels.

    The sentinel leg is the lossy one by construction -- it has to steal a value
    from the range -- so what this pins is that the *data present here* survives,
    not that any data would.  ``test_to_sentinel_refuses_*`` covers the rest.
    """
    mask = one_col(values, factory(null_storage="mask", **kw))
    assert mask["a"].null_storage == "mask"

    sent = mask.convert_nulls("a", to="sentinel")
    assert sent["a"].null_storage == "sentinel"
    assert sent["a"].null_value is not None
    assert as_list(sent["a"]) == as_list(mask["a"])

    back = sent.convert_nulls("a", to="mask")
    assert back["a"].null_storage == "mask"
    assert as_list(back["a"]) == as_list(mask["a"])
    assert back["a"].dtype == mask["a"].dtype or label in ("string", "bytes")


def test_conversion_leaves_the_source_untouched():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    converted = t.convert_nulls("a", to="sentinel")
    assert t["a"].null_storage == "mask"
    assert t["a"].null_value is None
    assert converted["a"].null_storage == "sentinel"


def test_implicit_sweep_converts_every_convertible_column():
    t = table(
        [(5, "ab", 1), (None, None, 2)],
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, null_storage="mask"),
        plain=blosc2.int64(),
    )
    sent = t.convert_nulls(to="sentinel")
    assert sent["k"].null_storage == "sentinel"
    assert sent["s"].null_storage == "sentinel"
    assert sent["plain"].null_storage == "none"  # not nullable: nothing to convert


def test_implicit_sweep_skips_what_it_cannot_convert():
    """A dictionary column has one representation, so a sweep passes over it."""
    t = table(
        [("x", 5), (None, None)],
        d=blosc2.dictionary(),
        k=blosc2.int64(null_storage="mask"),
    )
    sent = t.convert_nulls(to="sentinel")
    assert sent["d"].null_storage == "code"
    assert sent["k"].null_storage == "sentinel"


def test_converting_is_idempotent():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    once = t.convert_nulls("a", to="mask")
    twice = once.convert_nulls("a", to="mask")
    assert twice["a"].null_storage == "mask"
    assert as_list(twice["a"]) == [5, None, 1]


# ---------------------------------------------------------------------------
# A null-free column converts as a pure schema update
# ---------------------------------------------------------------------------


def test_null_free_sentinel_column_writes_no_sidecar():
    """Decision 9 reaching migration: an absent sidecar already says all-valid."""
    t = one_col([1, 2, 3], blosc2.int64(nullable=True, null_value=-1))
    converted = t.convert_nulls("a", to="mask")
    assert converted["a"].null_storage == "mask"
    assert converted._null_mask("a") is None
    assert converted["a"].is_null().tolist() == [False, False, False]
    assert as_list(converted["a"]) == [1, 2, 3]


def test_null_free_mask_column_needs_no_value_rewrite():
    t = one_col([1, 2, 3], blosc2.int64(null_storage="mask"))
    assert t._null_mask("a") is None
    converted = t.convert_nulls("a", to="sentinel")
    assert converted["a"].null_storage == "sentinel"
    assert as_list(converted["a"]) == [1, 2, 3]


def test_converting_drops_the_sidecar_on_the_way_to_sentinel():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    converted = t.convert_nulls("a", to="sentinel")
    assert converted._null_mask("a") is None
    assert converted["a"].is_null().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# to="sentinel" refuses what it cannot represent
# ---------------------------------------------------------------------------


def test_to_sentinel_refuses_a_full_range_int8():
    """The case that motivated the design, refused rather than silently lossy.

    ``int8`` using all 256 values has nothing left to reserve, which is exactly
    why mask storage exists; converting back would relabel a real ``-128``.
    """
    t = one_col([*range(-128, 128), None], blosc2.int8(null_storage="mask"), capacity=300)
    with pytest.raises(ValueError, match="already contains -128"):
        t.convert_nulls("a", to="sentinel")


@needs_utf8
def test_to_sentinel_refuses_utf8_holding_the_sentinel_string():
    t = one_col(["__BLOSC2_NULL__", None], blosc2.utf8(null_storage="mask"))
    with pytest.raises(ValueError, match="already contains"):
        t.convert_nulls("a", to="sentinel")


@needs_utf8
def test_to_sentinel_accepts_a_different_sentinel_instead():
    """The refusal names the offending value, and a free one is accepted."""
    t = one_col(["__BLOSC2_NULL__", None], blosc2.utf8(null_storage="mask"))
    converted = t.convert_nulls("a", to="sentinel", null_value="\x00\x01")
    assert converted["a"].null_value == "\x00\x01"
    assert as_list(converted["a"]) == ["__BLOSC2_NULL__", None]


def test_to_sentinel_refuses_complex():
    """No value can be spared from the complex plane, so it is mask or nothing."""
    t = one_col([1 + 2j, None], blosc2.complex128(null_storage="mask"))
    with pytest.raises(ValueError, match="no value can be reserved"):
        t.convert_nulls("a", to="sentinel")


def test_a_nan_already_present_does_not_block_the_nan_sentinel():
    """Folding NaN into "null" is the documented semantic change, not data loss.

    A sentinel float column has always spelled its nulls ``NaN``; converting a
    mask column that contains a real NaN therefore *changes what that row means*,
    which decision 6 covers, rather than losing a value the way an integer
    sentinel collision would.
    """
    t = one_col([1.0, float("nan"), None], blosc2.float64(null_storage="mask"))
    converted = t.convert_nulls("a", to="sentinel")
    assert converted["a"].is_null().tolist() == [False, True, True]


def test_the_refusal_leaves_the_column_alone():
    t = one_col([*range(-128, 128), None], blosc2.int8(null_storage="mask"), capacity=300)
    with pytest.raises(ValueError):
        t.convert_nulls("a", to="sentinel")
    assert t["a"].null_storage == "mask"
    assert t["a"].null_value is None
    assert t["a"][:-1].tolist() == list(range(-128, 128))


# ---------------------------------------------------------------------------
# Naming a column that cannot convert is an error
# ---------------------------------------------------------------------------


def test_naming_a_dictionary_column_raises():
    t = one_col(["x", None], blosc2.dictionary())
    with pytest.raises(ValueError, match="only representation its kind has"):
        t.convert_nulls("a", to="mask")


def test_naming_a_non_nullable_column_raises():
    t = one_col([1, 2], blosc2.int64())
    with pytest.raises(ValueError, match="not nullable"):
        t.convert_nulls("a", to="mask")


def test_naming_an_unknown_column_raises():
    t = one_col([1, 2], blosc2.int64())
    with pytest.raises(KeyError, match="not found"):
        t.convert_nulls("nope")


def test_bad_to_value_raises():
    t = one_col([1, 2], blosc2.int64())
    with pytest.raises(ValueError, match="must be 'mask' or 'sentinel'"):
        t.convert_nulls(to="bitmap")


def test_null_value_with_to_mask_raises():
    t = one_col([1, 2], blosc2.int64())
    with pytest.raises(ValueError, match="only applies to to='sentinel'"):
        t.convert_nulls(to="mask", null_value=7)


def test_null_value_across_several_columns_raises():
    """One value cannot be right for several kinds, so it has to name a column."""
    t = table(
        [(5, "ab"), (None, None)],
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, null_storage="mask"),
    )
    with pytest.raises(ValueError, match="applies to a single column"):
        t.convert_nulls(to="sentinel", null_value=-1)


# ---------------------------------------------------------------------------
# ndarray columns
# ---------------------------------------------------------------------------


def ndarray_table(dtype, item_shape, rows, **spec_kw):
    spec = blosc2.ndarray(dtype=dtype, item_shape=item_shape, **spec_kw)
    Row = dataclasses.make_dataclass("NdRow", [("v", object, blosc2.field(spec))])
    t = blosc2.CTable(Row, expected_size=16)
    t.extend([(r,) for r in rows])
    return t


def test_ndarray_column_converts_both_ways():
    t = ndarray_table(np.int32, (3,), [[1, 2, 3], None, [4, 5, 6]], null_storage="mask")
    sent = t.convert_nulls("v", to="sentinel")
    assert sent["v"].is_null().tolist() == [False, True, False]
    # The old rule: a row is null only when *every* element holds the sentinel.
    assert sent["v"][:][1].tolist() == [np.iinfo(np.int32).min] * 3
    back = sent.convert_nulls("v", to="mask")
    assert back["v"].is_null().tolist() == [False, True, False]
    assert back["v"][:][0].tolist() == [1, 2, 3]


def test_bool_ndarray_column_changes_dtype_both_ways():
    t = ndarray_table(np.bool_, (2,), [[True, False], None], null_storage="mask")
    assert t["v"].dtype == np.dtype(np.bool_)
    sent = t.convert_nulls("v", to="sentinel")
    assert sent["v"].dtype == np.dtype(np.uint8)
    assert sent["v"][:].tolist() == [[1, 0], [255, 255]]
    back = sent.convert_nulls("v", to="mask")
    assert back["v"].dtype == np.dtype(np.bool_)
    assert back["v"].is_null().tolist() == [False, True]
    assert back["v"][:][0].tolist() == [True, False]


def test_a_declared_uint8_ndarray_column_is_not_mistaken_for_a_widened_bool():
    """The unflip must undo only the flip it made.

    ``bool`` columns are physically ``uint8`` under sentinel storage, and mask
    storage undoes that -- but a column whose *declared* dtype is ``uint8`` holds
    real byte values, and turning it into ``np.bool_`` would truncate every one
    of them to a flag.  Keyed off a recorded flag rather than off the dtype.
    """
    spec = blosc2.ndarray(dtype=np.uint8, item_shape=(2,), nullable=True, null_storage="mask")
    Row = dataclasses.make_dataclass("U8Row", [("v", object, blosc2.field(spec))])
    t = blosc2.CTable(Row, expected_size=8)
    t.extend([([7, 200],), (None,)])
    assert t["v"].dtype == np.dtype(np.uint8)
    assert t["v"][:][0].tolist() == [7, 200]


# ---------------------------------------------------------------------------
# Persistence, in place, and the crash ordering
# ---------------------------------------------------------------------------


def test_inplace_conversion_survives_a_reopen(tmp_path):
    t = one_col([5, -1, 1], blosc2.int64(nullable=True, null_value=-1), urlpath=tmp_path / "t.b2t")
    assert t.convert_nulls("a", to="mask", inplace=True) is t
    assert t["a"].null_storage == "mask"

    reopened = blosc2.open(str(tmp_path / "t.b2t"))
    assert reopened["a"].null_storage == "mask"
    assert reopened._null_mask("a") is not None
    assert as_list(reopened["a"]) == [5, None, 1]


def test_inplace_conversion_back_to_sentinel_survives_a_reopen(tmp_path):
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"), urlpath=tmp_path / "t.b2t")
    t.convert_nulls("a", to="sentinel", inplace=True)
    reopened = blosc2.open(str(tmp_path / "t.b2t"))
    assert reopened["a"].null_storage == "sentinel"
    assert reopened._null_mask("a") is None
    assert as_list(reopened["a"]) == [5, None, 1]


def test_inplace_on_a_persistent_table_refuses_a_dtype_change(tmp_path):
    """No ordering of "replace the array" and "update the schema" is crash-safe."""
    t = one_col([1, 255, 0], blosc2.bool(nullable=True, null_value=255), urlpath=tmp_path / "t.b2t")
    with pytest.raises(ValueError, match="changes its physical dtype"):
        t.convert_nulls("a", to="mask", inplace=True)
    assert t["a"].null_storage == "sentinel"
    assert t["a"].dtype == np.dtype(np.uint8)


def test_the_same_column_converts_fine_out_of_place(tmp_path):
    t = one_col([1, 255, 0], blosc2.bool(nullable=True, null_value=255), urlpath=tmp_path / "t.b2t")
    converted = t.convert_nulls("a", to="mask")
    assert converted["a"].dtype == np.dtype(np.bool_)
    assert as_list(converted["a"]) == [True, None, False]
    landed = converted.copy(urlpath=str(tmp_path / "out.b2d"))
    assert landed["a"].null_storage == "mask"
    assert as_list(landed["a"]) == [True, None, False]


def test_inplace_on_an_in_memory_table_may_change_dtype():
    """There is no crash window in memory, so the restriction does not apply."""
    t = one_col([1, 255, 0], blosc2.bool(nullable=True, null_value=255))
    t.convert_nulls("a", to="mask", inplace=True)
    assert t["a"].dtype == np.dtype(np.bool_)
    assert as_list(t["a"]) == [True, None, False]


def test_inplace_refuses_a_view():
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"))
    with pytest.raises(ValueError, match="view"):
        t.where("a > 0").convert_nulls("a", to="sentinel", inplace=True)


def test_inplace_refuses_a_read_only_table(tmp_path):
    one_col([5, -1, 1], blosc2.int64(nullable=True, null_value=-1), urlpath=tmp_path / "t.b2t")
    ro = blosc2.open(str(tmp_path / "t.b2t"), mode="r")
    with pytest.raises(ValueError, match="read-only"):
        ro.convert_nulls("a", to="mask", inplace=True)


def test_an_orphan_sidecar_still_reads_as_sentinel(tmp_path):
    """The crash-safety argument for ``to="mask"``, in the state it protects.

    Step 1 writes the complete sidecar; step 2 flips the schema.  A crash in
    between leaves a ``.notnull`` key beside a schema that still says
    ``sentinel`` -- and a sentinel column never opens a sidecar, so the table
    reads exactly as it did before the conversion started.
    """
    t = one_col([5, -1, 1], blosc2.int64(nullable=True, null_value=-1), urlpath=tmp_path / "t.b2t")
    # Simulate an interruption after step 1 by writing the sidecar by hand.
    mask = t._ensure_null_mask("a")
    mask[:] = True
    mask[1] = False
    del t

    reopened = blosc2.open(str(tmp_path / "t.b2t"), mode="a")
    assert reopened["a"].null_storage == "sentinel"
    assert reopened._storage.has_null_mask("a")  # the orphan is on disk
    assert as_list(reopened["a"]) == [5, None, 1]  # and is not consulted
    # Finishing the conversion is then just the schema update.
    reopened.convert_nulls("a", to="mask", inplace=True)
    assert as_list(reopened["a"]) == [5, None, 1]


def test_a_mask_column_whose_nulls_still_hold_the_sentinel_reads_correctly(tmp_path):
    """The other side of the same window: schema flipped, values not normalized.

    Step 3 rewrites the null slots to the fill and is purely cosmetic -- what
    sits under ``valid=False`` is unobservable through the Column API, and the
    fill is explicitly not part of the format contract.
    """
    t = one_col([5, None, 1], blosc2.int64(null_storage="mask"), urlpath=tmp_path / "t.b2t")
    t._cols["a"][1] = -12345  # as if step 3 had never run
    assert t["a"].is_null().tolist() == [False, True, False]
    assert t["a"].null_count() == 1
    assert t["a"].fillna(0).tolist() == [5, 0, 1]
    assert t["a"].min() == 1


# ---------------------------------------------------------------------------
# Nothing auto-migrates
# ---------------------------------------------------------------------------


def test_copy_preserves_each_columns_null_storage():
    t = table(
        [(5, "ab"), (None, "__BLOSC2_NULL__")],
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, nullable=True, null_value="__BLOSC2_NULL__"),
    )
    c = t.copy()
    assert c["k"].null_storage == "mask"
    assert c["s"].null_storage == "sentinel"


def test_saving_preserves_null_storage_under_a_mask_default_policy(tmp_path):
    """A policy governs *creation*; it must never rewrite what already exists."""
    t = one_col([5, -1, 1], blosc2.int64(nullable=True, null_value=-1))
    with blosc2.null_policy(blosc2.NullPolicy(null_storage="mask")):
        t.save(str(tmp_path / "t.b2d"))
        reopened = blosc2.open(str(tmp_path / "t.b2d"))
        assert reopened["a"].null_storage == "sentinel"
        assert reopened["a"].null_value == -1


def test_opening_an_old_table_changes_nothing(tmp_path):
    t = one_col([1, 255, 0], blosc2.bool(nullable=True, null_value=255), urlpath=tmp_path / "t.b2t")
    del t
    with blosc2.null_policy(blosc2.NullPolicy(null_storage="mask")):
        reopened = blosc2.open(str(tmp_path / "t.b2t"))
        assert reopened["a"].null_storage == "sentinel"
        assert reopened["a"].dtype == np.dtype(np.uint8)
        assert reopened["a"].null_value == 255


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_null_storage_is_reported_per_column():
    t = table(
        [(5, "ab", 1, "x")],
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, nullable=True, null_value="__BLOSC2_NULL__"),
        plain=blosc2.int64(),
        d=blosc2.dictionary(),
    )
    assert t["k"].null_storage == "mask"
    assert t["s"].null_storage == "sentinel"
    assert t["plain"].null_storage == "none"
    assert t["d"].null_storage == "code"


def test_info_tags_each_column_with_where_its_nulls_live():
    """What you read to decide whether a table needs converting."""
    t = table(
        [(5, "ab", 1)],
        k=blosc2.int64(null_storage="mask"),
        s=blosc2.string(max_length=2, nullable=True, null_value="__BLOSC2_NULL__"),
        plain=blosc2.int64(),
    )
    summary = dict(t.info_items)["columns"]
    assert "nullable[mask]" in str(summary["k"])
    assert "nullable[sentinel]" in str(summary["s"])
    assert "nullable" not in str(summary["plain"])


def test_column_info_reports_whether_a_sidecar_exists_yet():
    """The visible difference between "nullable" and "has ever held a null"."""
    t = one_col([1, 2], blosc2.int64(null_storage="mask"))
    assert dict(t["a"].info_items)["null_sidecar"] is False
    t["a"][0] = None
    assert dict(t["a"].info_items)["null_sidecar"] is True


# ---------------------------------------------------------------------------
# The converted table behaves like a natively created one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("to", ["mask", "sentinel"])
def test_a_converted_column_still_sorts_groups_and_queries(to):
    """A converted column is indistinguishable from a natively created one."""
    if to == "mask":
        t = one_col([5, -1, 1, 9, -1, 2], blosc2.int64(nullable=True, null_value=-1))
    else:
        t = one_col([5, None, 1, 9, None, 2], blosc2.int64(null_storage="mask"))
    converted = t.convert_nulls("a", to=to)

    assert as_list(converted.sort_by("a")["a"]) == [1, 2, 5, 9, None, None]
    assert converted["a"].null_count() == 2
    assert converted["a"].min() == 1
    assert len(converted.where("a < 3")) == 2
    grouped = converted.group_by(["a"], dropna=True, sort=True).agg(n=("a", "count"))
    assert len(grouped) == 4


def test_a_converted_column_round_trips_through_arrow():
    """The point of converting to mask in the first place."""
    pa = pytest.importorskip("pyarrow")
    t = one_col([True, 255, False], blosc2.bool(nullable=True, null_value=255))
    converted = t.convert_nulls("a", to="mask")
    arrow = converted.to_arrow()
    assert arrow.column("a").type == pa.bool_()
    assert arrow.column("a").to_pylist() == [True, None, False]


def test_extending_a_converted_column_accepts_none():
    t = one_col([5, -1], blosc2.int64(nullable=True, null_value=-1))
    converted = t.convert_nulls("a", to="mask")
    converted.append((None,))
    converted.extend([(7,), (None,)])
    assert as_list(converted["a"]) == [5, None, None, 7, None]
