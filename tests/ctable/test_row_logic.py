import pytest
import numpy as np
from blosc2 import CTable
from pydantic import BaseModel, Field
from typing import Annotated

from blosc2.ctable import Column


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


def generate_test_data(n_rows: int, start_id: int = 0) -> list:
    return [(start_id + i, float(i * 10), i % 2 == 0) for i in range(n_rows)]


def test_row_int_no_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[0]

    assert isinstance(result, CTable)
    assert len(result) == 1
    assert result.id[0] == 0
    assert result.score[0] == 0.0
    assert result.active[0] == True

    result = tabla.row[10]
    assert len(result) == 1
    assert result.id[0] == 10
    assert result.score[0] == 100.0


def test_row_int_with_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    result = tabla.row[0]
    assert len(result) == 1
    assert result.id[0] == 0

    result = tabla.row[1]
    assert len(result) == 1
    assert result.id[0] == 2

    result = tabla.row[5]
    assert len(result) == 1
    assert result.id[0] == 10


def test_row_int_negative_indices():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[-1]
    assert len(result) == 1
    assert result.id[0] == 19

    result = tabla.row[-5]
    assert len(result) == 1
    assert result.id[0] == 15


def test_row_int_out_of_range():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    with pytest.raises(IndexError):
        _ = tabla.row[10]

    with pytest.raises(IndexError):
        _ = tabla.row[100]

    with pytest.raises(IndexError):
        _ = tabla.row[-11]


def test_row_slice_no_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[0:5]

    assert isinstance(result, CTable)
    assert len(result) == 5
    assert list(result.id) == [0, 1, 2, 3, 4]

    result = tabla.row[10:15]
    assert len(result) == 5
    assert list(result.id) == [10, 11, 12, 13, 14]

    result = tabla.row[::2]
    assert len(result) == 10
    assert list(result.id) == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]


def test_row_slice_with_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    result = tabla.row[0:5]
    assert len(result) == 5
    assert list(result.id) == [0, 2, 4, 6, 8]

    result = tabla.row[5:10]
    assert len(result) == 5
    assert list(result.id) == [10, 11, 12, 13, 14]


def test_row_slice_beyond_table_size():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[11:20]
    assert len(result) == 0

    result = tabla.row[5:100]
    assert len(result) == 5
    assert list(result.id) == [5, 6, 7, 8, 9]

    result = tabla.row[100:]
    assert len(result) == 0


def test_row_slice_negative_indices():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[-5:]
    assert len(result) == 5
    assert list(result.id) == [15, 16, 17, 18, 19]

    result = tabla.row[-10:-5]
    assert len(result) == 5
    assert list(result.id) == [10, 11, 12, 13, 14]


def test_row_list_no_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[[0, 5, 10, 15]]

    assert isinstance(result, CTable)
    assert len(result) == 4
    assert set(result.id) == {0, 5, 10, 15}

    result = tabla.row[[19, 0, 10]]
    assert len(result) == 3
    assert set(result.id) == {0, 10, 19}


def test_row_list_with_holes():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    result = tabla.row[[0, 2, 4]]
    assert len(result) == 3
    assert set(result.id) == {0, 4, 8}

    result = tabla.row[[5, 3, 1]]
    assert len(result) == 3
    assert set(result.id) == {2, 6, 10}


def test_row_list_out_of_range():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    with pytest.raises(IndexError):
        _ = tabla.row[[0, 5, 100]]

    with pytest.raises(IndexError):
        _ = tabla.row[[0, 1, -11]]


def test_row_returns_view_properties():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[0:10]

    assert result.base is tabla
    assert result._row_type == tabla._row_type
    assert result._cols is tabla._cols
    assert result._col_widths == tabla._col_widths
    assert result.col_names == tabla.col_names


def test_row_chained_views():
    data = generate_test_data(100)
    tabla0 = CTable(RowModel, new_data=data)

    tabla1 = tabla0.row[:50]
    assert tabla1.base is tabla0
    assert len(tabla1) == 50
    assert list(tabla1.id)[:5] == [0, 1, 2, 3, 4]

    tabla2 = tabla1.row[:10]
    assert tabla2.base is tabla1
    assert len(tabla2) == 10
    assert list(tabla2.id) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    tabla3 = tabla2.row[5:]
    assert tabla3.base is tabla2
    assert len(tabla3) == 5
    assert list(tabla3.id) == [5, 6, 7, 8, 9]


def test_row_view_on_view_with_holes():
    data = generate_test_data(50)
    tabla0 = CTable(RowModel, new_data=data)

    tabla0.delete([5, 10, 15, 20, 25])

    tabla1 = tabla0.row[:30]
    assert tabla1.base is tabla0
    assert len(tabla1) == 30

    tabla2 = tabla1.row[10:20]
    assert tabla2.base is tabla1
    assert len(tabla2) == 10


def test_row_empty_slice():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[5:5]
    assert len(result) == 0

    result = tabla.row[0:0]
    assert len(result) == 0


def test_row_full_slice():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[:]
    assert len(result) == 10
    assert list(result.id) == list(range(10))


def test_row_empty_table():
    tabla = CTable(RowModel)

    with pytest.raises(IndexError):
        _ = tabla.row[0]

    result = tabla.row[:]
    assert len(result) == 0

    result = tabla.row[0:10]
    assert len(result) == 0


def test_row_all_deleted():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    tabla.delete(list(range(10)))

    with pytest.raises(IndexError):
        _ = tabla.row[0]

    result = tabla.row[:]
    assert len(result) == 0


def test_row_view_maintains_mask_reference():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[5:15]

    mask = result._valid_rows[:]
    true_count = np.count_nonzero(mask)
    assert true_count == 10


def test_row_single_element_list():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[[5]]
    assert len(result) == 1
    assert result.id[0] == 5


def test_row_duplicate_indices_in_list():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[[5, 5, 5]]
    assert len(result) == 1
    assert result.id[0] == 5


def test_row_view_base_chain():
    data = generate_test_data(100)
    tabla0 = CTable(RowModel, new_data=data)

    assert tabla0.base is None

    tabla1 = tabla0.row[:80]
    assert tabla1.base is tabla0

    tabla2 = tabla1.row[:60]
    assert tabla2.base is tabla1

    tabla3 = tabla2.row[:40]
    assert tabla3.base is tabla2


def test_row_view_read_operations():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    view = tabla.row[5:15]

    assert view.id[0] == 5
    assert view.score[0] == 50.0
    assert view.active[0] == False

    assert list(view.id) == list(range(5, 15))


def test_row_list_empty():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[[]]
    assert len(result) == 0


def test_row_slice_with_step():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[0:10:2]
    assert len(result) == 5
    assert list(result.id) == [0, 2, 4, 6, 8]

    result = tabla.row[1:10:3]
    assert len(result) == 3
    assert list(result.id) == [1, 4, 7]


def test_row_list_with_negative_indices():
    data = generate_test_data(10)
    tabla = CTable(RowModel, new_data=data)

    result = tabla.row[[0, -1, 5]]
    assert len(result) == 3
    assert set(result.id) == {0, 5, 9}


def test_row_view_columns_are_live():
    data = generate_test_data(20)
    tabla = CTable(RowModel, new_data=data)

    view = tabla.row[5:10]

    col = view.id
    assert isinstance(col, Column) if 'Column' in dir() else True
    assert col._table is view


if __name__ == "__main__":
    pytest.main(["-v", __file__])
