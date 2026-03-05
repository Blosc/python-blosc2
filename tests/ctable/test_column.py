import pytest
import numpy as np
import blosc2
from blosc2 import CTable
from pydantic import BaseModel, Field
from typing import Annotated


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


def test_column_dtype():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id
    col_score = tabla.score
    col_active = tabla.active

    assert col_id.dtype == np.int64
    assert col_score.dtype == np.float64
    assert col_active.dtype == np.bool_


def test_column_references():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    assert col_id._raw_col is tabla._cols["id"]
    assert col_id._valid_rows is tabla._valid_rows


def test_column_getitem_int_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id
    print("hola")
    assert col_id[0] == 0
    print("hola")
    assert col_id[5] == 5
    print("hola")
    assert col_id[19] == 19
    print("hola")
    assert col_id[-1] == 19
    print("hola")
    assert col_id[-5] == 15
    print("hola")


def test_column_getitem_int_with_holes():
    data = [(i, float(i * 10), i % 2 == 0) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    assert col_id[0] == 0
    assert col_id[1] == 2
    assert col_id[2] == 4
    assert col_id[3] == 6
    assert col_id[4] == 8
    assert col_id[-1] == 19
    assert col_id[-2] == 18


def test_column_getitem_slice_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    result = col_id[0:5]
    expected = [0, 1, 2, 3, 4]
    assert list(result) == expected

    result = col_id[10:15]
    expected = [10, 11, 12, 13, 14]
    assert list(result) == expected

    result = col_id[::2]
    expected = list(range(0, 20, 2))
    assert list(result) == expected


def test_column_getitem_slice_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

    col_id = tabla.id

    result = col_id[0:5]
    expected = [0, 2, 4, 6, 8]
    assert list(result) == expected

    result = col_id[5:10]
    expected = [10, 12, 14, 16, 18]
    assert list(result) == expected

    result = col_id[::2]
    expected = [0, 4, 8, 12, 16]
    assert list(result) == expected


def test_column_getitem_list_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    result = col_id[[0, 5, 10, 15]]
    expected = [0, 5, 10, 15]
    assert list(result) == expected

    result = col_id[[19, 0, 10]]
    expected = [19, 0, 10]
    assert list(result) == expected


def test_column_getitem_list_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    result = col_id[[0, 2, 4]]
    expected = [0, 4, 8]
    assert list(result) == expected

    result = col_id[[5, 3, 1]]
    expected = [10, 6, 2]
    assert list(result) == expected


def test_column_getitem_out_of_range_int():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    with pytest.raises(IndexError):
        _ = col_id[100]

    with pytest.raises(IndexError):
        _ = col_id[-100]


def test_column_getitem_out_of_range_list():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    with pytest.raises(IndexError):
        _ = col_id[[0, 1, 100]]


def test_column_setitem_int_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    col_id[0] = 999
    assert col_id[0] == 999

    col_id[10] = 888
    assert col_id[10] == 888

    col_id[-1] = 777
    assert col_id[-1] == 777


def test_column_setitem_int_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    col_id[0] = 999
    assert col_id[0] == 999
    assert tabla._cols["id"][0] == 999

    col_id[2] = 888
    assert col_id[2] == 888
    assert tabla._cols["id"][4] == 888

    col_id[-1] = 777
    assert col_id[-1] == 777


def test_column_setitem_slice_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    col_id[0:5] = [100, 101, 102, 103, 104]

    assert col_id[0] == 100
    assert col_id[1] == 101
    assert col_id[2] == 102
    assert col_id[3] == 103
    assert col_id[4] == 104


def test_column_setitem_slice_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    col_id[0:3] = [100, 200, 300]

    assert col_id[0] == 100
    assert col_id[1] == 200
    assert col_id[2] == 300


def test_column_setitem_list_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    col_id[[0, 5, 10]] = [100, 500, 1000]

    assert col_id[0] == 100
    assert col_id[5] == 500
    assert col_id[10] == 1000


def test_column_setitem_list_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    col_id[[0, 2, 4]] = [100, 200, 300]

    assert col_id[0] == 100
    assert col_id[2] == 200
    assert col_id[4] == 300


def test_column_iter_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    result = list(col_id)
    expected = list(range(20))

    assert result == expected


def test_column_iter_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

    col_id = tabla.id

    result = list(col_id)
    expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    assert result == expected


def test_column_iter_score():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([0, 5, 10, 15])

    col_score = tabla.score

    result = list(col_score)
    expected = [10.0, 20.0, 30.0, 40.0, 60.0, 70.0, 80.0, 90.0,
                110.0, 120.0, 130.0, 140.0, 160.0, 170.0, 180.0, 190.0]

    assert result == expected


def test_column_len_no_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    assert len(col_id) == 20


def test_column_len_with_holes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([1, 3, 5, 7, 9])

    col_id = tabla.id

    assert len(col_id) == 15


def test_column_len_after_multiple_deletes():
    data = [(i, float(i * 10), True) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    col_id = tabla.id

    assert len(col_id) == 20

    tabla.delete([0, 1, 2])
    assert len(col_id) == 17

    tabla.delete([0, 1, 2, 3, 4])
    assert len(col_id) == 12


def test_column_multiple_columns_consistency():
    data = [(i, float(i * 10), i % 2 == 0) for i in range(20)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete([2, 5, 8, 11, 14])

    col_id = tabla.id
    col_score = tabla.score
    col_active = tabla.active

    assert len(col_id) == len(col_score) == len(col_active) == 15

    for i in range(len(col_id)):
        expected_id = col_id[i]
        expected_score = col_score[i]
        expected_active = col_active[i]

        assert expected_score == float(expected_id * 10)


def test_column_empty_table():
    tabla = CTable(RowModel)

    col_id = tabla.id

    assert len(col_id) == 0

    result = list(col_id)
    assert result == []


def test_column_all_deleted():
    data = [(i, float(i * 10), True) for i in range(10)]
    tabla = CTable(RowModel, new_data=data)

    tabla.delete(list(range(10)))

    col_id = tabla.id

    assert len(col_id) == 0

    result = list(col_id)
    assert result == []
