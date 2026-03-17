#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel, Field

from blosc2 import CTable


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


DATA20 = [(i, float(i * 10), True) for i in range(20)]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_column_metadata():
    """dtype correctness and internal reference consistency."""
    tabla = CTable(RowModel, new_data=DATA20)

    assert tabla.id.dtype == np.int64
    assert tabla.score.dtype == np.float64
    assert tabla.active.dtype == np.bool_

    assert tabla.id._raw_col is tabla._cols["id"]
    assert tabla.id._valid_rows is tabla._valid_rows


def test_column_getitem_no_holes():
    """int, slice, and list indexing on a full table."""
    tabla = CTable(RowModel, new_data=DATA20)
    col = tabla.id

    # int
    assert col[0] == 0
    assert col[5] == 5
    assert col[19] == 19
    assert col[-1] == 19
    assert col[-5] == 15

    # slice
    assert list(col[0:5]) == [0, 1, 2, 3, 4]
    assert list(col[10:15]) == [10, 11, 12, 13, 14]
    assert list(col[::2]) == list(range(0, 20, 2))

    # list
    assert list(col[[0, 5, 10, 15]]) == [0, 5, 10, 15]
    assert list(col[[19, 0, 10]]) == [19, 0, 10]


def test_column_getitem_with_holes():
    """int, slice, and list indexing after deletions."""
    # int + list: delete odd indices [1,3,5,7,9] → valid: 0,2,4,6,8,10…19
    tabla = CTable(RowModel, new_data=DATA20)
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

    # slice: delete all odd indices → valid: 0,2,4,…,18
    tabla2 = CTable(RowModel, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    col2 = tabla2.id

    assert list(col2[0:5]) == [0, 2, 4, 6, 8]
    assert list(col2[5:10]) == [10, 12, 14, 16, 18]
    assert list(col2[::2]) == [0, 4, 8, 12, 16]


def test_column_getitem_out_of_range():
    """int and list indexing raise IndexError when out of bounds."""
    tabla = CTable(RowModel, new_data=DATA20)
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
    tabla = CTable(RowModel, new_data=DATA20)
    col = tabla.id

    # int
    col[0] = 999
    assert col[0] == 999
    col[10] = 888
    assert col[10] == 888
    col[-1] = 777
    assert col[-1] == 777

    # slice
    col[0:5] = [100, 101, 102, 103, 104]
    assert list(col[0:5]) == [100, 101, 102, 103, 104]

    # list
    col[[0, 5, 10]] = [10, 50, 100]
    assert col[0] == 10
    assert col[5] == 50
    assert col[10] == 100


def test_column_setitem_with_holes():
    """int, slice, and list assignment after deletions."""
    tabla = CTable(RowModel, new_data=DATA20)
    tabla.delete([1, 3, 5, 7, 9])
    col = tabla.id

    # int: logical index 2 → physical index 4
    col[0] = 999
    assert col[0] == 999
    assert tabla._cols["id"][0] == 999

    col[2] = 888
    assert col[2] == 888
    assert tabla._cols["id"][4] == 888

    col[-1] = 777
    assert col[-1] == 777

    # slice
    col[0:3] = [100, 200, 300]
    assert col[0] == 100
    assert col[1] == 200
    assert col[2] == 300

    # list
    col[[0, 2, 4]] = [11, 22, 33]
    assert col[0] == 11
    assert col[2] == 22
    assert col[4] == 33


def test_column_iter():
    """Iteration over full table, with odd-index holes, and on score column."""
    # No holes
    tabla = CTable(RowModel, new_data=DATA20)
    assert list(tabla.id) == list(range(20))

    # All odd indices deleted
    tabla2 = CTable(RowModel, new_data=DATA20)
    tabla2.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    assert list(tabla2.id) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    # score column with scattered deletions
    tabla3 = CTable(RowModel, new_data=DATA20)
    tabla3.delete([0, 5, 10, 15])
    expected_score = [
        10.0,
        20.0,
        30.0,
        40.0,
        60.0,
        70.0,
        80.0,
        90.0,
        110.0,
        120.0,
        130.0,
        140.0,
        160.0,
        170.0,
        180.0,
        190.0,
    ]
    assert list(tabla3.score) == expected_score


def test_column_len():
    """len() after no deletions, partial deletions, cumulative deletions, and cross-column."""
    tabla = CTable(RowModel, new_data=DATA20)
    col = tabla.id
    assert len(col) == 20

    tabla.delete([1, 3, 5, 7, 9])
    assert len(col) == 15

    # Cumulative deletes
    tabla2 = CTable(RowModel, new_data=DATA20)
    col2 = tabla2.id
    tabla2.delete([0, 1, 2])
    assert len(col2) == 17
    tabla2.delete([0, 1, 2, 3, 4])
    assert len(col2) == 12

    # Cross-column consistency
    data = [(i, float(i * 10), i % 2 == 0) for i in range(10)]
    tabla3 = CTable(RowModel, new_data=data, expected_size=10)
    tabla3.delete([0, 1, 5, 6, 9])
    assert len(tabla3.id) == len(tabla3.score) == len(tabla3.active) == 5
    for i in range(len(tabla3.id)):
        assert tabla3.score[i] == float(tabla3.id[i] * 10)


def test_column_edge_cases():
    """Empty table and fully-deleted table both behave as zero-length columns."""
    # Empty table from the start
    tabla = CTable(RowModel)
    assert len(tabla.id) == 0
    assert list(tabla.id) == []

    # All rows deleted
    data = [(i, float(i * 10), True) for i in range(10)]
    tabla2 = CTable(RowModel, new_data=data)
    tabla2.delete(list(range(10)))
    assert len(tabla2.id) == 0
    assert list(tabla2.id) == []


if __name__ == "__main__":
    pytest.main(["-v", __file__])
