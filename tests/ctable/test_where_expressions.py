from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2


@dataclass
class Row:
    value: int = blosc2.field(blosc2.int32())
    category: int = blosc2.field(blosc2.int32())


DATA = [(10, 1), (20, 8), (30, 5), (2, 99)]


def test_where_accepts_string_expression():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where("value * category >= 150")

    np.testing.assert_array_equal(view.value[:], np.array([20, 30, 2], dtype=np.int32))
    np.testing.assert_array_equal(view.category[:], np.array([8, 5, 99], dtype=np.int32))


def test_where_accepts_column_arithmetic_expression():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where((t.value * t.category) >= 150)

    np.testing.assert_array_equal(view.value[:], np.array([20, 30, 2], dtype=np.int32))
    np.testing.assert_array_equal(view.category[:], np.array([8, 5, 99], dtype=np.int32))


def test_where_column_arithmetic_can_be_composed():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where(((t.value + 2) * t.category) >= 100)

    np.testing.assert_array_equal(view.value[:], np.array([20, 30, 2], dtype=np.int32))


def test_where_col_expr_accepts_transcendentals():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where(((t.value + 2) * blosc2.sin(t.category)) >= 10)

    np.testing.assert_array_equal(view.value[:], np.array([10, 20], dtype=np.int32))


def test_where_str_expr_accepts_transcendentals():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where("(value + 2) * sin(category) >= 10")

    np.testing.assert_array_equal(view.value[:], np.array([10, 20], dtype=np.int32))


def test_where_str_expr_uses_computed_cols():
    t = blosc2.CTable(Row, new_data=DATA)
    t.add_computed_column("score", "value * category")

    view = t.where("score >= 150")

    np.testing.assert_array_equal(view.value[:], np.array([20, 30, 2], dtype=np.int32))


def test_where_string_expression_must_be_boolean():
    t = blosc2.CTable(Row, new_data=DATA)

    with pytest.raises(TypeError, match="Expected boolean"):
        t.where("value * category")


@dataclass
class IdRow:
    id: int = blosc2.field(blosc2.int64(), default=0)


def _id_table(n=6):
    t = blosc2.CTable(IdRow, expected_size=n)
    arr = np.empty(n, dtype=[("id", "<i8")])
    arr["id"] = np.arange(n)
    t.extend(arr, validate=False)
    return t


def test_where_short_mask_is_relative_to_the_view():
    """A mask shorter than the view selects the view's own rows.

    It used to be padded out to the physical column length, which aligned it
    with the underlying column and let rows outside the view through.
    """
    view = _id_table()[1:4]  # ids [1, 2, 3]

    # Predicate built from a slice of the view's column: [1, 2] > 1 -> [F, T].
    result = view.where(view["id"][0:2] > 1)

    np.testing.assert_array_equal(result.id[:], np.array([2], dtype=np.int64))


def test_where_short_mask_skips_deleted_rows():
    t = _id_table()
    t.delete([0, 2])  # live ids [1, 3, 4, 5]

    result = t.where(np.array([True, False]))

    np.testing.assert_array_equal(result.id[:], np.array([1], dtype=np.int64))


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        (np.array([], dtype=bool), []),
        (np.array([True, True]), [1, 2]),
        (np.array([False, True, True]), [2, 3]),
    ],
)
def test_where_mask_lengths_on_a_sliced_view(mask, expected):
    view = _id_table()[1:4]  # ids [1, 2, 3]

    result = view.where(mask)

    np.testing.assert_array_equal(result.id[:], np.array(expected, dtype=np.int64))


def test_where_short_mask_on_a_sorted_view_follows_sort_order():
    ordered = _id_table().sort_by("id", ascending=False)  # ids [5, 4, 3, 2, 1, 0]

    result = ordered.where(np.array([True, False]))

    np.testing.assert_array_equal(result.id[:], np.array([5], dtype=np.int64))
