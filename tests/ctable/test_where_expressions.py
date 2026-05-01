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


def test_where_column_expression_accepts_transcendental_functions():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where(((t.value + 2) * blosc2.sin(t.category)) >= 10)

    np.testing.assert_array_equal(view.value[:], np.array([10, 20], dtype=np.int32))


def test_where_string_expression_accepts_transcendental_functions():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where("(value + 2) * sin(category) >= 10")

    np.testing.assert_array_equal(view.value[:], np.array([10, 20], dtype=np.int32))


def test_where_string_expression_can_reference_computed_columns():
    t = blosc2.CTable(Row, new_data=DATA)
    t.add_computed_column("score", "value * category")

    view = t.where("score >= 150")

    np.testing.assert_array_equal(view.value[:], np.array([20, 30, 2], dtype=np.int32))


def test_where_string_expression_must_be_boolean():
    t = blosc2.CTable(Row, new_data=DATA)

    with pytest.raises(TypeError, match="Expected boolean"):
        t.where("value * category")
