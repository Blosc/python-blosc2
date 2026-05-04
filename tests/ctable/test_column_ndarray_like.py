from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Row:
    x: int = blosc2.field(blosc2.int32())
    y: int = blosc2.field(blosc2.int32())
    flag: bool = blosc2.field(blosc2.bool())


DATA = [(1, 5, True), (-1, 5, True), (2, 20, False), (3, 7, False)]


def test_column_logical_metadata():
    t = blosc2.CTable(Row, new_data=DATA)
    view = t.where(t.x > 0)

    assert view.x.shape == (3,)
    assert view.x.ndim == 1
    assert view.x.size == 3


def test_column_boolean_operators_build_lazy_expressions():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where(t.flag & (t.x > 0))

    np.testing.assert_array_equal(view.x[:], np.array([1], dtype=np.int32))


def test_column_boolean_invert_builds_lazy_expression():
    t = blosc2.CTable(Row, new_data=DATA)

    view = t.where(~t.flag)

    np.testing.assert_array_equal(view.x[:], np.array([2, 3], dtype=np.int32))


def test_column_sum_accepts_dtype():
    t = blosc2.CTable(Row, new_data=DATA)

    result = t.x.sum(dtype=np.float64)

    assert isinstance(result, np.floating)
    assert result == 5.0
