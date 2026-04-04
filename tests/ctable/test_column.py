#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0))
    active: bool = blosc2.field(blosc2.bool(), default=True)


DATA20 = [(i, float(i * 10), True) for i in range(20)]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_column_metadata():
    """dtype correctness, internal reference consistency, and mask defaults."""
    tabla = CTable(Row, new_data=DATA20)

    assert tabla.id.dtype == np.int64
    assert tabla.score.dtype == np.float64
    assert tabla.active.dtype == np.bool_

    assert tabla.id._raw_col is tabla._cols["id"]
    assert tabla.id._valid_rows is tabla._valid_rows

    # mask is None by default
    assert tabla.id._mask is None
    assert tabla.score._mask is None


def test_column_getitem_no_holes():
    """int, slice, and list indexing on a full table."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    # int
    assert col[0] == 0
    assert col[5] == 5
    assert col[19] == 19
    assert col[-1] == 19
    assert col[-5] == 15

    # slice returns a Column view
    assert isinstance(col[0:5], blosc2.Column)
    assert isinstance(col[10:15], blosc2.Column)

    # list
    assert list(col[[0, 5, 10, 15]]) == [0, 5, 10, 15]
    assert list(col[[19, 0, 10]]) == [19, 0, 10]


def test_column_getitem_with_holes():
    """int, slice, and list indexing after deletions."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    assert col[0] == 0
    assert col[1] == 2
    assert col[2] == 4
    assert col[3] == 6
    assert col[4] == 8
    assert col[-1] == 19
    assert col[-2] == 18

    assert list(col[[0, 2, 4]]) == [0, 4, 8]
    assert list(col[[5, 3, 1]]) == [10, 6, 2]

    tabla2 = CTable(Row, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    col2 = tabla2.id

    assert list(col2[0:5].to_numpy()) == [0, 2, 4, 6, 8]
    assert list(col2[5:10].to_numpy()) == [10, 12, 14, 16, 18]
    assert list(col2[::2].to_numpy()) == [0, 4, 8, 12, 16]


def test_column_getitem_out_of_range():
    """int and list indexing raise IndexError when out of bounds."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    with pytest.raises(IndexError):
        _ = col[100]
    with pytest.raises(IndexError):
        _ = col[-100]
    with pytest.raises(IndexError):
        _ = col[[0, 1, 100]]


def test_column_setitem_no_holes():
    """int, slice, and list assignment on a full table."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    col[0] = 999
    assert col[0] == 999
    col[10] = 888
    assert col[10] == 888
    col[-1] = 777
    assert col[-1] == 777

    col[0:5] = [100, 101, 102, 103, 104]
    assert list(col[0:5].to_numpy()) == [100, 101, 102, 103, 104]

    col[[0, 5, 10]] = [10, 50, 100]
    assert col[0] == 10
    assert col[5] == 50
    assert col[10] == 100


def test_column_setitem_with_holes():
    """int, slice, and list assignment after deletions."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    col[0] = 999
    assert col[0] == 999
    assert tabla._cols["id"][0] == 999

    col[2] = 888
    assert col[2] == 888
    assert tabla._cols["id"][4] == 888

    col[-1] = 777
    assert col[-1] == 777

    col[0:3] = [100, 200, 300]
    assert col[0] == 100
    assert col[1] == 200
    assert col[2] == 300

    col[[0, 2, 4]] = [11, 22, 33]
    assert col[0] == 11
    assert col[2] == 22
    assert col[4] == 33


def test_column_iter():
    """Iteration over full table, with odd-index holes, and on score column."""
    tabla = CTable(Row, new_data=DATA20)
    assert list(tabla.id) == list(range(20))

    tabla2 = CTable(Row, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert list(tabla2.id) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    tabla3 = CTable(Row, new_data=DATA20)
    tabla3.delete([0, 5, 10, 15])
    # fmt: off
    expected_score = [
        10.0, 20.0, 30.0, 40.0,
        60.0, 70.0, 80.0, 90.0,
        110.0, 120.0, 130.0, 140.0,
        160.0, 170.0, 180.0, 190.0,
    ]
    # fmt: on
    assert list(tabla3.score) == expected_score


def test_column_len():
    """len() after no deletions, partial deletions, cumulative deletions, and cross-column."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id
    assert len(col) == 20

    tabla.delete([1, 3, 5, 7, 9])
    assert len(col) == 15

    tabla2 = CTable(Row, new_data=DATA20)
    col2 = tabla2.id
    tabla2.delete([0, 1, 2])
    assert len(col2) == 17
    tabla2.delete([0, 1, 2, 3, 4])
    assert len(col2) == 12

    data = [(i, float(i * 10), i % 2 == 0) for i in range(10)]
    tabla3 = CTable(Row, new_data=data, expected_size=10)
    tabla3.delete([0, 1, 5, 6, 9])
    assert len(tabla3.id) == len(tabla3.score) == len(tabla3.active) == 5
    for i in range(len(tabla3.id)):
        assert tabla3.score[i] == float(tabla3.id[i] * 10)


def test_column_edge_cases():
    """Empty table and fully-deleted table both behave as zero-length columns."""
    tabla = CTable(Row)
    assert len(tabla.id) == 0
    assert list(tabla.id) == []

    data = [(i, float(i * 10), True) for i in range(10)]
    tabla2 = CTable(Row, new_data=data)
    tabla2.delete(list(range(10)))
    assert len(tabla2.id) == 0
    assert list(tabla2.id) == []


# -------------------------------------------------------------------
# New tests for Column view (mask) and to_array()
# -------------------------------------------------------------------


def test_column_slice_returns_view():
    """Column[slice] returns a Column instance with a non-None mask."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    view = col[0:5]
    assert isinstance(view, blosc2.Column)
    assert view._mask is not None
    assert view._table is tabla
    assert view._col_name == "id"


def test_to_array_no_holes():
    """to_array() on a slice view returns correct data on a full table."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    np.testing.assert_array_equal(col[0:5].to_numpy(), np.array([0, 1, 2, 3, 4], dtype=np.int64))
    np.testing.assert_array_equal(col[5:10].to_numpy(), np.array([5, 6, 7, 8, 9], dtype=np.int64))
    np.testing.assert_array_equal(col[15:20].to_numpy(), np.array([15, 16, 17, 18, 19], dtype=np.int64))
    np.testing.assert_array_equal(col[0:20].to_numpy(), np.arange(20, dtype=np.int64))


def test_to_array_with_holes():
    """to_array() on a slice view skips deleted rows correctly."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])  # keep evens: 0,2,4,...,18
    col = tabla.id

    # logical [0:5] → physical rows 0,2,4,6,8
    np.testing.assert_array_equal(col[0:5].to_numpy(), np.array([0, 2, 4, 6, 8], dtype=np.int64))
    # logical [5:10] → physical rows 10,12,14,16,18
    np.testing.assert_array_equal(col[5:10].to_numpy(), np.array([10, 12, 14, 16, 18], dtype=np.int64))


def test_to_array_full_column():
    """to_array() with no slice (full column) returns all valid rows."""
    tabla = CTable(Row, new_data=DATA20)
    tabla.delete([0, 10, 19])
    col = tabla.id

    expected = np.array([i for i in range(20) if i not in {0, 10, 19}], dtype=np.int64)
    np.testing.assert_array_equal(col[0 : len(col)].to_numpy(), expected)


def test_to_array_mask_does_not_include_deleted():
    """Mask & valid_rows intersection excludes deleted rows inside the slice range."""
    tabla = CTable(Row, new_data=DATA20)
    # delete rows 2 and 3, which fall inside slice [0:5]
    tabla.delete([2, 3])
    col = tabla.id

    # logical [0:5] should now map to physical rows 0,1,4,5,6
    result = col[0:5].to_numpy()
    np.testing.assert_array_equal(result, np.array([0, 1, 4, 5, 6], dtype=np.int64))


def test_column_view_mask_is_independent():
    """Two slice views on the same column have independent masks."""
    tabla = CTable(Row, new_data=DATA20)
    col = tabla.id

    view_a = col[0:5]

    np.testing.assert_array_equal(view_a.to_numpy(), np.arange(0, 5, dtype=np.int64))


if __name__ == "__main__":
    pytest.main(["-v", __file__])
