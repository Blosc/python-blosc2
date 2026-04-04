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
from blosc2.ctable import Column


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0))
    active: bool = blosc2.field(blosc2.bool(), default=True)


def generate_test_data(n_rows: int, start_id: int = 0) -> list:
    return [(start_id + i, float(i * 10), i % 2 == 0) for i in range(n_rows)]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_row_int_indexing():
    """int indexing: no holes, with holes, negative indices, and out-of-range."""
    data = generate_test_data(20)

    # No holes: spot checks
    t = CTable(Row, new_data=data)
    r = t.row[0]
    assert isinstance(r, CTable)
    assert len(r) == 1
    assert r.id[0] == 0
    assert r.score[0] == 0.0
    assert r.active[0]
    assert t.row[10].id[0] == 10
    assert t.row[10].score[0] == 100.0

    # Negative indices
    assert t.row[-1].id[0] == 19
    assert t.row[-5].id[0] == 15

    # With holes: delete odd positions -> valid: 0,2,4,6,8,10...
    t.delete([1, 3, 5, 7, 9])
    assert t.row[0].id[0] == 0
    assert t.row[1].id[0] == 2
    assert t.row[5].id[0] == 10

    # Out of range
    t2 = CTable(Row, new_data=generate_test_data(10))
    for idx in [10, 100, -11]:
        with pytest.raises(IndexError):
            _ = t2.row[idx]


def test_row_slice_indexing():
    """Slice indexing: no holes, with holes, step, negative, beyond bounds, empty/full."""
    data = generate_test_data(20)

    # No holes
    t = CTable(Row, new_data=data)
    assert isinstance(t.row[0:5], CTable)
    assert list(t.row[0:5].id) == [0, 1, 2, 3, 4]
    assert list(t.row[10:15].id) == [10, 11, 12, 13, 14]
    assert list(t.row[::2].id) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    # With step
    assert list(t.row[0:10:2].id) == [0, 2, 4, 6, 8]
    assert list(t.row[1:10:3].id) == [1, 4, 7]

    # Negative indices
    assert list(t.row[-5:].id) == [15, 16, 17, 18, 19]
    assert list(t.row[-10:-5].id) == [10, 11, 12, 13, 14]

    # With holes: delete odd positions
    t.delete([1, 3, 5, 7, 9])
    assert list(t.row[0:5].id) == [0, 2, 4, 6, 8]
    assert list(t.row[5:10].id) == [10, 11, 12, 13, 14]

    # Beyond bounds
    t2 = CTable(Row, new_data=generate_test_data(10))
    assert len(t2.row[11:20]) == 0
    assert list(t2.row[5:100].id) == [5, 6, 7, 8, 9]
    assert len(t2.row[100:]) == 0

    # Empty and full slices
    assert len(t2.row[5:5]) == 0
    assert len(t2.row[0:0]) == 0
    result = t2.row[:]
    assert len(result) == 10
    assert list(result.id) == list(range(10))


def test_row_list_indexing():
    """List indexing: no holes, with holes, out-of-range, edge cases."""
    data = generate_test_data(20)

    # No holes
    t = CTable(Row, new_data=data)
    r = t.row[[0, 5, 10, 15]]
    assert isinstance(r, CTable)
    assert len(r) == 4
    assert set(r.id) == {0, 5, 10, 15}
    assert set(t.row[[19, 0, 10]].id) == {0, 10, 19}

    # With holes: delete [1,3,5,7,9] -> logical 0->id0, 1->id2, 2->id4...
    t.delete([1, 3, 5, 7, 9])
    assert set(t.row[[0, 2, 4]].id) == {0, 4, 8}
    assert set(t.row[[5, 3, 1]].id) == {2, 6, 10}

    # Negative indices in list
    t2 = CTable(Row, new_data=generate_test_data(10))
    assert set(t2.row[[0, -1, 5]].id) == {0, 5, 9}

    # Single element
    assert t2.row[[5]].id[0] == 5

    # Duplicate indices -> deduplicated
    r_dup = t2.row[[5, 5, 5]]
    assert len(r_dup) == 1
    assert r_dup.id[0] == 5

    # Empty list
    assert len(t2.row[[]]) == 0

    # Out of range
    for bad in [[0, 5, 100], [0, 1, -11]]:
        with pytest.raises(IndexError):
            _ = t2.row[bad]


def test_row_view_properties():
    """View metadata, base chain, mask integrity, column liveness, and chained views."""
    data = generate_test_data(100)
    tabla0 = CTable(Row, new_data=data)

    # Base is None on root table
    assert tabla0.base is None

    # View properties are shared with parent
    v = tabla0.row[0:10]
    assert v.base is tabla0
    assert v._row_type == tabla0._row_type
    assert v._cols is tabla0._cols
    assert v._col_widths == tabla0._col_widths
    assert v.col_names == tabla0.col_names

    # Read ops on view
    view = tabla0.row[5:15]
    assert view.id[0] == 5
    assert view.score[0] == 50.0
    assert not view.active[0]
    assert list(view.id) == list(range(5, 15))

    # Mask integrity
    assert np.count_nonzero(view._valid_rows[:]) == 10

    # Column is live (points back to its view)
    col = view.id
    assert isinstance(col, Column)
    assert col._table is view

    # Chained views: base always points to immediate parent
    tabla1 = tabla0.row[:50]
    assert tabla1.base is tabla0
    assert len(tabla1) == 50

    tabla2 = tabla1.row[:10]
    assert tabla2.base is tabla1
    assert len(tabla2) == 10
    assert list(tabla2.id) == list(range(10))

    tabla3 = tabla2.row[5:]
    assert tabla3.base is tabla2
    assert len(tabla3) == 5
    assert list(tabla3.id) == [5, 6, 7, 8, 9]

    # Chained view with holes on parent
    tabla0.delete([5, 10, 15, 20, 25])
    tv1 = tabla0.row[:30]
    assert tv1.base is tabla0
    assert len(tv1) == 30
    tv2 = tv1.row[10:20]
    assert tv2.base is tv1
    assert len(tv2) == 10


def test_row_edge_cases():
    """Empty table, fully-deleted table: int raises IndexError, slice returns empty."""
    # Empty table
    empty = CTable(Row)
    with pytest.raises(IndexError):
        _ = empty.row[0]
    assert len(empty.row[:]) == 0
    assert len(empty.row[0:10]) == 0

    # All rows deleted
    data = generate_test_data(10)
    t = CTable(Row, new_data=data)
    t.delete(list(range(10)))
    with pytest.raises(IndexError):
        _ = t.row[0]
    assert len(t.row[:]) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
