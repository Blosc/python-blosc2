#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.take."""

import dataclasses

import numpy as np
import pytest

import blosc2


@dataclasses.dataclass
class Row:
    id: int = blosc2.field(blosc2.int32())
    value: float = blosc2.field(blosc2.float64())
    vector: np.ndarray = blosc2.field(blosc2.ndarray((2,), blosc2.int32()))  # noqa: RUF009


@dataclasses.dataclass
class MixedRow:
    id: int = blosc2.field(blosc2.int32())
    note: str = blosc2.field(blosc2.vlstring(nullable=True))
    tags: list[int] = blosc2.field(blosc2.list(blosc2.int64(), nullable=True))  # noqa: RUF009


def make_table(n=8):
    t = blosc2.CTable(Row)
    for i in range(n):
        t.append([i, float(i) * 1.5, np.array([i, i + 10], dtype=np.int32)])
    return t


def test_ctable_take_preserves_order_duplicates_and_negative_indices():
    t = make_table(8)
    t.delete(2)
    t.delete(5)
    # Live logical ids are [0, 1, 3, 4, 6, 7].

    result = t.take([3, 0, -1, 3])
    top_level_result = blosc2.take(t, [3, 0, -1, 3])

    assert isinstance(top_level_result, blosc2.CTable)
    assert result.nrows == 4
    np.testing.assert_array_equal(result["id"][:], np.array([4, 0, 7, 4], dtype=np.int32))
    np.testing.assert_array_equal(top_level_result["id"][:], result["id"][:])
    np.testing.assert_allclose(result["value"][:], np.array([6.0, 0.0, 10.5, 6.0]))
    np.testing.assert_array_equal(
        result["vector"][:],
        np.array([[4, 14], [0, 10], [7, 17], [4, 14]], dtype=np.int32),
    )


def test_ctable_take_empty_returns_empty_compact_table():
    t = make_table(4)

    result = t.take([])

    assert result.nrows == 0
    assert result["id"][:].tolist() == []


def test_ctable_take_handles_varlen_and_list_columns():
    t = blosc2.CTable(
        MixedRow,
        new_data=[
            (0, "zero", [0]),
            (1, None, None),
            (2, "two", [2, 20]),
            (3, "three", [3]),
        ],
    )

    result = t.take([2, 0, 2, 1])

    assert result["id"][:].tolist() == [2, 0, 2, 1]
    assert list(result["note"][:]) == ["two", "zero", "two", None]
    assert list(result["tags"][:]) == [[2, 20], [0], [2, 20], None]


def test_column_take_preserves_order_duplicates_and_negative_indices():
    t = make_table(8)
    t.delete(2)
    t.delete(5)
    # Live logical ids are [0, 1, 3, 4, 6, 7].

    col = t["id"].take([3, 0, -1, 3])
    top_level_col = blosc2.take(t["id"], [3, 0, -1, 3])
    vector_col = t["vector"].take([3, 0, -1, 3])

    assert isinstance(top_level_col, blosc2.Column)
    assert len(col) == 4
    np.testing.assert_array_equal(col[:], np.array([4, 0, 7, 4], dtype=np.int32))
    np.testing.assert_array_equal(top_level_col[:], col[:])
    np.testing.assert_array_equal(
        vector_col[:],
        np.array([[4, 14], [0, 10], [7, 17], [4, 14]], dtype=np.int32),
    )


def test_column_take_respects_column_view_mask():
    t = make_table(8)
    sub = t["id"].view[[1, 3, 6]]

    result = sub.take([2, 0, 2])

    np.testing.assert_array_equal(result[:], np.array([6, 1, 6], dtype=np.int32))


def test_column_take_handles_varlen_and_list_columns():
    t = blosc2.CTable(
        MixedRow,
        new_data=[
            (0, "zero", [0]),
            (1, None, None),
            (2, "two", [2, 20]),
            (3, "three", [3]),
        ],
    )

    notes = t["note"].take([2, 0, 2, 1])
    tags = t["tags"].take([2, 0, 2, 1])

    assert list(notes[:]) == ["two", "zero", "two", None]
    assert list(tags[:]) == [[2, 20], [0], [2, 20], None]


def test_ctable_take_rejects_bad_indices():
    t = make_table(4)

    with pytest.raises(TypeError, match="integers"):
        t.take([1.5])
    with pytest.raises(ValueError, match="1-D"):
        t.take([[0, 1]])
    with pytest.raises(IndexError, match="bounds"):
        t.take([4])


def test_column_take_rejects_bad_indices():
    col = make_table(4)["id"]

    with pytest.raises(TypeError, match="integers"):
        col.take([1.5])
    with pytest.raises(ValueError, match="1-D"):
        col.take([[0, 1]])
    with pytest.raises(IndexError, match="bounds"):
        col.take([4])


def test_top_level_take_rejects_axis_for_ctable_and_column():
    t = make_table(4)

    with pytest.raises(ValueError, match="axis"):
        blosc2.take(t, [0], axis=0)
    with pytest.raises(ValueError, match="axis"):
        blosc2.take(t["id"], [0], axis=0)


# ── CTable.slice (contiguous range of live rows; copy or zero-copy view) ──


def test_slice_range_styles_agree():
    """slice(stop), slice(start, stop) and slice(slice(...)) select the same rows."""
    t = make_table(10)
    expected = np.arange(2, 6, dtype=np.int32)
    np.testing.assert_array_equal(t.slice(2, 6)["id"][:], expected)
    np.testing.assert_array_equal(t.slice(slice(2, 6))["id"][:], expected)
    # single-arg form behaves like range(stop)
    np.testing.assert_array_equal(t.slice(4)["id"][:], np.arange(0, 4, dtype=np.int32))


def test_slice_negative_and_out_of_range_bounds_clamp():
    t = make_table(10)
    # negative start counts from the end
    np.testing.assert_array_equal(t.slice(-3, 10)["id"][:], np.arange(7, 10, dtype=np.int32))
    # stop past the end clamps; start past stop is empty
    np.testing.assert_array_equal(t.slice(8, 999)["id"][:], np.arange(8, 10, dtype=np.int32))
    assert t.slice(6, 2).nrows == 0


def test_slice_copy_false_is_a_zero_copy_view():
    t = make_table(8)
    view = t.slice(2, 6, copy=False)
    # Shares the parent's column storage (no copy) and re-indexes from 0.
    assert view._cols is t._cols
    assert view.base is t
    assert view.nrows == 4
    np.testing.assert_array_equal(view["id"][:], np.arange(2, 6, dtype=np.int32))


def test_slice_copy_true_is_an_independent_compact_table():
    t = make_table(8)
    sub = t.slice(2, 6)  # copy=True by default
    assert sub._cols is not t._cols
    assert sub.nrows == 4
    np.testing.assert_array_equal(sub["id"][:], np.arange(2, 6, dtype=np.int32))


def test_slice_skips_deleted_rows_in_logical_space():
    t = make_table(8)
    t.delete(2)
    t.delete(5)
    # Live logical ids are [0, 1, 3, 4, 6, 7]; logical [1:4] -> ids [1, 3, 4].
    for copy in (True, False):
        sub = t.slice(1, 4, copy=copy)
        np.testing.assert_array_equal(sub["id"][:], np.array([1, 3, 4], dtype=np.int32))


def test_slice_rejects_step_and_double_bounds():
    t = make_table(4)
    with pytest.raises(ValueError, match="step"):
        t.slice(slice(0, 4, 2))
    with pytest.raises(TypeError):
        t.slice(slice(0, 4), 4)
