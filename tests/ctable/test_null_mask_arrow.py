#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Lossless Arrow/Parquet interop for mask-storage columns (Phase 6).

This is what the whole design is for.  A sentinel steals a value from the
dtype's range, so importing Arrow data has always been lossy for any column
whose data happens to use that value — silently lossy, since nothing raises.
``test_sentinel_storage_is_lossy_where_mask_storage_is_not`` measures exactly
that, and is the reason the rest of this file exists.

Phase 6 ships **opt-in**: ``null_storage="mask"`` on ``from_arrow`` /
``from_parquet``, or a ``NullPolicy``.  The default stays ``"sentinel"`` until
Phase 9, so every existing caller is unaffected.

On comparing round-trips: ``pyarrow.Array.equals`` uses IEEE semantics, so two
*identical* arrays containing NaN compare unequal, and the plan's literal
``to_arrow(from_arrow(x)).equals(x)`` contract is unachievable for float data
by construction.  :func:`assert_same_logical` compares what is actually
observable — the validity bitmap, and the values under valid rows — treating
NaN as equal to NaN and keeping signed zeros distinct.
"""

from __future__ import annotations

import numpy as np
import pytest
from utf8_compat import needs_utf8, utf8_spec

import blosc2

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def _flat(column):
    """A plain Array, whether the input was one already or a ChunkedArray."""
    return column.combine_chunks() if hasattr(column, "combine_chunks") else column


def assert_same_logical(got, want):
    """Assert two Arrow columns hold the same observable data.

    Values sitting under ``valid=False`` are deliberately *not* compared: they
    are the fill, which is explicitly not part of the format contract.
    """
    got, want = _flat(got), _flat(want)
    assert len(got) == len(want), "length"
    assert got.null_count == want.null_count, "null_count"

    got_valid = got.is_valid().to_numpy(zero_copy_only=False)
    want_valid = want.is_valid().to_numpy(zero_copy_only=False)
    assert np.array_equal(got_valid, want_valid), "validity bitmap"

    for i in range(len(got)):
        if not want_valid[i]:
            continue
        g, w = got[i], want[i]
        if pa.types.is_timestamp(g.type):
            g, w = g.cast(pa.int64()), w.cast(pa.int64())
        g, w = g.as_py(), w.as_py()
        if isinstance(g, float) and isinstance(w, float):
            assert g == w or (np.isnan(g) and np.isnan(w)), f"row {i}: {g!r} != {w!r}"
            if g == 0.0:
                assert np.copysign(1, g) == np.copysign(1, w), f"row {i}: signed zero"
        else:
            assert g == w, f"row {i}: {g!r} != {w!r}"


def round_trip(arrow_array, **kwargs):
    """Import one Arrow column into a mask-backed CTable and export it again."""
    table = pa.table({"v": arrow_array})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask", **kwargs)
    return ct, ct.to_arrow().column("v")


# ---------------------------------------------------------------------------
# The round-trip contract: none of these survive under a sentinel
# ---------------------------------------------------------------------------

ROUND_TRIP_CASES = [
    ("bool", pa.array([True, None, False, True], type=pa.bool_())),
    ("int8_full_range", pa.array([*range(-128, 128), None], type=pa.int8())),
    ("uint8_full_range", pa.array([*range(256), None], type=pa.uint8())),
    (
        "float64_specials",
        pa.array(
            [float("nan"), None, 0.0, -0.0, float("inf"), float("-inf")],
            type=pa.float64(),
        ),
    ),
    (
        "utf8_free_text",
        pa.array(["", "\x00", "__BLOSC2_NULL__", None, "\U0001f389x"], type=pa.string()),
    ),
    (
        "timestamp_int64_min",
        pa.array(
            [
                np.datetime64("2020-01-01", "us"),
                None,
                np.datetime64(np.iinfo(np.int64).min + 1, "us"),
            ],
            type=pa.timestamp("us"),
        ),
    ),
]


@pytest.mark.parametrize(("label", "arrow_array"), ROUND_TRIP_CASES)
def test_arrow_round_trip_is_lossless(label, arrow_array):
    _, exported = round_trip(arrow_array)
    assert_same_logical(exported, arrow_array)


@pytest.mark.parametrize(("label", "arrow_array"), ROUND_TRIP_CASES)
def test_parquet_round_trip_is_lossless(label, arrow_array, tmp_path):
    src = str(tmp_path / "in.parquet")
    out = str(tmp_path / "out.parquet")
    original = pa.table({"v": arrow_array})
    pq.write_table(original, src)

    ct = blosc2.CTable.from_parquet(src, null_storage="mask")
    ct.to_parquet(out)
    assert_same_logical(pq.read_table(out).column("v"), original.column("v"))


@pytest.mark.parametrize("max_length", [4])
def test_fixed_width_string_fully_occupying_its_width(max_length):
    arrow_array = pa.array(["abcd", None, "", "wxyz"], type=pa.string())
    table = pa.table({"v": arrow_array})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask", string_max_length=max_length)
    assert ct["v"].dtype == np.dtype(f"U{max_length}")  # no widening for a sentinel
    assert_same_logical(ct.to_arrow().column("v"), arrow_array)


def test_fixed_width_bytes_fully_occupying_its_width():
    arrow_array = pa.array([b"abcd", None, b"", b"wxyz"], type=pa.binary())
    table = pa.table({"v": arrow_array})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask", string_max_length=4)
    assert ct["v"].dtype == np.dtype("S4")
    exported = ct.to_arrow().column("v")
    assert exported.to_pylist() == [b"abcd", None, b"", b"wxyz"]


def test_ndarray_column_round_trip():
    arrow_array = pa.array([[1, 2], None, [3, 4]], type=pa.list_(pa.int64(), 2))
    ct, exported = round_trip(arrow_array)
    assert ct["v"].null_storage == "mask"
    assert exported.to_pylist() == [[1, 2], None, [3, 4]]


# ---------------------------------------------------------------------------
# The measurement that justifies the design
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "arrow_array", "lossy_result"),
    [
        # -128 is the sentinel int8 picks, so real -128 data reads back as null.
        ("int8_min", pa.array([-128, None, 127], type=pa.int8()), [None, None, 127]),
        # "__BLOSC2_NULL__" is literally the utf8 sentinel.  Only on NumPy >= 2:
        # without StringDType an Arrow string column imports as vlstring, whose
        # nulls are native None, so there is no sentinel to collide with and
        # nothing is lost.
        pytest.param(
            "utf8_sentinel_literal",
            pa.array(["", "__BLOSC2_NULL__", None], type=pa.string()),
            ["", None, None],
            marks=needs_utf8,
        ),
    ],
)
def test_sentinel_storage_is_lossy_where_mask_storage_is_not(label, arrow_array, lossy_result):
    """Silent corruption, not an error — which is what makes it worth fixing."""
    table = pa.table({"v": arrow_array})

    sentinel = blosc2.CTable.from_arrow(table, null_storage="sentinel")
    assert sentinel.to_arrow().column("v").to_pylist() == lossy_result

    masked = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert masked.to_arrow().column("v").to_pylist() == arrow_array.to_pylist()


def test_nullable_bool_imports_without_the_255_reservation():
    table = pa.table({"v": pa.array([True, None, False], type=pa.bool_())})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert ct["v"].dtype == np.dtype(np.bool_)
    assert ct["v"][:].tolist() == [True, False, False]
    assert ct["v"].is_null().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# Choosing the storage
# ---------------------------------------------------------------------------


def test_default_is_a_mask():
    """Lossless round-trip is why the sidecar exists, so imports get one.

    Which is the point of the flip: an Arrow column carries a validity bitmap,
    and reading it into a sidecar preserves it exactly, where a sentinel has to
    steal a value from the range to stand in for it.
    """
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    assert blosc2.CTable.from_arrow(table)["v"].null_storage == "mask"


def test_sentinel_storage_is_one_parameter_away():
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    ct = blosc2.CTable.from_arrow(table, null_storage="sentinel")
    assert ct["v"].null_storage == "sentinel"
    assert ct["v"].null_value == np.iinfo(np.int64).min


def test_explicit_parameter_selects_mask():
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    assert blosc2.CTable.from_arrow(table, null_storage="mask")["v"].null_storage == "mask"


def test_null_policy_selects_mask():
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    with blosc2.null_policy(blosc2.NullPolicy(null_storage="mask")):
        assert blosc2.CTable.from_arrow(table)["v"].null_storage == "mask"


def test_explicit_parameter_overrides_the_policy():
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    with blosc2.null_policy(blosc2.NullPolicy(null_storage="mask")):
        ct = blosc2.CTable.from_arrow(table, null_storage="sentinel")
    assert ct["v"].null_storage == "sentinel"


def test_column_null_values_still_forces_sentinel_per_column():
    table = pa.table(
        {
            "a": pa.array([1, None, 3], type=pa.int64()),
            "b": pa.array([1, None, 3], type=pa.int64()),
        }
    )
    with blosc2.null_policy(blosc2.NullPolicy(column_null_values={"a": -7})):
        ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert ct["a"].null_storage == "sentinel"
    assert ct["a"].null_value == -7
    assert ct["b"].null_storage == "mask"


def test_non_nullable_columns_get_no_null_channel():
    """Arrow fields are nullable by default, so this needs an explicit schema."""
    schema = pa.schema([pa.field("v", pa.int64(), nullable=False)])
    table = pa.table({"v": pa.array([1, 2, 3], type=pa.int64())}, schema=schema)
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert ct["v"].null_storage == "none"


def test_a_nullable_column_with_no_nulls_writes_no_sidecar():
    """Decision 9 survives the import path."""
    field = pa.field("v", pa.int64(), nullable=True)
    table = pa.table({"v": pa.array([1, 2, 3], type=pa.int64())}, schema=pa.schema([field]))
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert ct["v"].null_storage == "mask"
    assert ct._null_mask("v") is None
    assert ct["v"].null_count() == 0


def test_import_error_now_points_at_mask_storage():
    """A type with no available sentinel still fails under sentinel storage...

    ...but the message now names the way out, which is the whole point: mask
    storage needs no sentinel, so every type can carry nulls.
    """
    table = pa.table({"v": pa.array([1.0, None], type=pa.float64())})
    with blosc2.null_policy(blosc2.NullPolicy(float_value=None)):
        with pytest.raises(TypeError, match="null_storage='mask'"):
            blosc2.CTable.from_arrow(table, null_storage="sentinel")


def test_mask_storage_imports_what_no_sentinel_could():
    """The same input the previous test rejects, accepted."""
    table = pa.table({"v": pa.array([1.0, None], type=pa.float64())})
    with blosc2.null_policy(blosc2.NullPolicy(float_value=None)):
        ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    assert ct["v"].is_null().tolist() == [False, True]


def test_auto_null_sentinels_false_is_irrelevant_under_mask():
    """Mask storage picks no sentinel, so disabling the picker changes nothing."""
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask", auto_null_sentinels=False)
    assert ct["v"].is_null().tolist() == [False, True, False]


# ---------------------------------------------------------------------------
# The utf8 validity bitmap: bit order, pinned to literal bytes
# ---------------------------------------------------------------------------


def test_utf8_validity_bitmap_is_lsb_first():
    """Arrow validity bitmaps are LSB-first, and a round-trip cannot prove it.

    Import would unpack a wrongly-packed bitmap the same wrong way, so this
    asserts the literal buffer bytes for a known pattern instead.  Rows
    ``[valid, null, valid, valid, valid, null, valid, valid]`` must pack to
    ``0b1101_1101`` = 0xDD, not to the MSB-first ``0b1011_1011`` = 0xBB.
    """
    values = ["a", None, "c", "d", "e", None, "g", "h"]
    table = pa.table({"v": pa.array(values, type=pa.string())})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    exported = ct.to_arrow().column("v").combine_chunks()

    validity = exported.buffers()[0]
    assert validity is not None, "a column with nulls must carry a validity buffer"
    assert validity.to_pybytes()[0] == 0xDD


def test_utf8_dense_buffer_export_matches_the_generic_path():
    """The dense fast path builds Arrow buffers directly; it must not diverge."""
    values = ["a", None, "c", None, "e"]
    table = pa.table({"v": pa.array(values, type=pa.string())})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")

    dense = ct.to_arrow().column("v")
    # Deleting and re-compacting leaves a table the fast path declines (its
    # guard is _last_pos == _n_rows over a dense root table), so this exercises
    # the per-row path over the same data.
    generic = ct.where("True" if False else ct["v"].notnull() | ct["v"].is_null()).to_arrow().column("v")
    assert dense.to_pylist() == generic.to_pylist() == values


def test_utf8_export_of_a_filtered_view_keeps_its_nulls():
    values = ["a", None, "c", None, "e"]
    table = pa.table({"v": pa.array(values, type=pa.string()), "k": pa.array([0, 1, 1, 1, 0])})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask")
    view = ct.where("k == 1")
    assert view.to_arrow().column("v").to_pylist() == [None, "c", None]


# ---------------------------------------------------------------------------
# Export of every V1 kind, from a natively-built table
# ---------------------------------------------------------------------------


def build(spec, values, capacity=32):
    import dataclasses

    ann = (
        object
        if isinstance(spec, (blosc2.schema.NDArraySpec, blosc2.schema.timestamp))
        else spec.python_type
    )
    Row = dataclasses.make_dataclass("R", [("v", ann, blosc2.field(spec))])
    t = blosc2.CTable(Row, expected_size=capacity)
    t.extend([(v,) for v in values])
    return t


EXPORT_CASES = [
    ("bool", blosc2.bool(null_storage="mask"), [True, None, False], [True, None, False]),
    ("int8", blosc2.int8(null_storage="mask"), [-128, None, 127], [-128, None, 127]),
    ("uint8", blosc2.uint8(null_storage="mask"), [0, None, 255], [0, None, 255]),
    (
        "string",
        blosc2.string(max_length=4, null_storage="mask"),
        ["abcd", None, ""],
        ["abcd", None, ""],
    ),
    (
        "bytes",
        blosc2.bytes(max_length=4, null_storage="mask"),
        [b"abcd", None, b""],
        [b"abcd", None, b""],
    ),
    pytest.param(
        "utf8",
        utf8_spec(null_storage="mask"),
        ["", "\x00", None],
        ["", "\x00", None],
        marks=needs_utf8,
    ),
]


@pytest.mark.parametrize(("label", "spec", "values", "expected"), EXPORT_CASES)
def test_export_carries_the_sidecar_into_arrow(label, spec, values, expected):
    exported = build(spec, values).to_arrow().column("v")
    assert exported.to_pylist() == expected
    assert exported.null_count == 1


def test_export_of_a_null_free_mask_column_has_no_nulls():
    exported = build(blosc2.int64(null_storage="mask"), [1, 2, 3]).to_arrow().column("v")
    assert exported.null_count == 0
    assert exported.to_pylist() == [1, 2, 3]


def test_export_of_a_table_with_deletions_remaps_nulls():
    t = build(blosc2.int64(null_storage="mask"), [1, None, 3, None, 5])
    t.delete(0)
    assert t.to_arrow().column("v").to_pylist() == [None, 3, None, 5]


def test_ndarray_export_uses_row_level_validity():
    """A sentinel ndarray column needs *every* element to match; a mask does not."""
    spec = blosc2.ndarray((2,), dtype=blosc2.int64(), null_storage="mask")
    t = build(spec, [np.array([0, 0]), None, np.array([1, 2])])
    exported = t.to_arrow().column("v")
    # Row 0 is all-zero — the sentinel rule's definition of null — but is a
    # real value here, and only row 1 is null.
    assert exported.to_pylist() == [[0, 0], None, [1, 2]]


# ---------------------------------------------------------------------------
# Persistence of imported nulls
# ---------------------------------------------------------------------------


def test_imported_nulls_survive_a_persistent_import(tmp_path):
    path = str(tmp_path / "imported.b2d")
    table = pa.table({"v": pa.array([1, None, 3], type=pa.int64())})
    ct = blosc2.CTable.from_arrow(table, null_storage="mask", urlpath=path, mode="w")
    ct.close()

    reopened = blosc2.CTable.open(path)
    try:
        assert reopened["v"].null_storage == "mask"
        assert reopened["v"].is_null().tolist() == [False, True, False]
        assert reopened.to_arrow().column("v").to_pylist() == [1, None, 3]
    finally:
        reopened.close()


def test_import_spanning_several_batches_keeps_its_nulls():
    """The sidecar is created mid-import, on the first batch that has a null."""
    n = 300
    values = [None if i % 50 == 49 else i for i in range(n)]
    table = pa.table({"v": pa.array(values, type=pa.int64())})
    batches = table.to_batches(max_chunksize=64)
    ct = blosc2.CTable.from_arrow(table.schema, iter(batches), null_storage="mask")
    assert ct["v"].null_count() == n // 50
    assert ct.to_arrow().column("v").to_pylist() == values
