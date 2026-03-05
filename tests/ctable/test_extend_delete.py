#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from typing import Annotated, TypeVar

import numpy as np
import pytest
from pydantic import BaseModel, Field

from blosc2 import CTable

RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


def generate_test_data(n_rows: int, start_id: int = 1) -> list:
    return [
        (start_id + i, complex(i, -i), float((i * 7) % 100), bool(i % 2))
        for i in range(n_rows)
    ]


def get_valid_mask(table: CTable) -> np.ndarray:
    return np.array(table._valid_rows[:len(table._valid_rows)], dtype=bool)


def get_column_values(table: CTable, col_name: str, length: int) -> np.ndarray:
    return np.array(table._cols[col_name][:length])


def assert_mask_matches(table: CTable, expected_mask: list):
    actual_mask = get_valid_mask(table)[:len(expected_mask)]
    expected = np.array(expected_mask, dtype=bool)

    np.testing.assert_array_equal(
        actual_mask, expected,
        err_msg=f"Mask mismatch.\nExpected: {expected}\nGot: {actual_mask}"
    )


def assert_data_at_positions(table: CTable, positions: list, expected_ids: list):
    id_col = table.id
    for pos, expected_id in zip(positions, expected_ids, strict=True):
        actual_id = int(table._cols["id"][pos])
        assert actual_id == expected_id, \
            f"Position {pos}: expected ID {expected_id}, got {actual_id}"


def test_insert_after_delete_fills_last_gap():
    data_c1 = generate_test_data(7, start_id=1)
    table = CTable(RowModel, new_data=data_c1, expected_size=10)

    table.delete([0, 2, 4, 6])

    expected_mask_after_delete = [False, True, False, True, False, True, False]
    assert_mask_matches(table, expected_mask_after_delete)
    assert len(table) == 3

    data_c2 = generate_test_data(3, start_id=8)
    table.extend(data_c2)

    expected_mask_final = [False, True, False, True, False, True, True, True, True]
    assert_mask_matches(table, expected_mask_final)
    assert len(table) == 6

    assert_data_at_positions(table, [6, 7, 8], [8, 9, 10])


def test_append_single_row_fills_gap():
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10)

    table.delete([1, 3])

    expected_mask = [True, False, True, False, True]
    assert_mask_matches(table, expected_mask)

    table.append((6, 1j, 50.0, True))

    expected_mask_after = [True, False, True, False, True, True]
    assert_mask_matches(table, expected_mask_after)

    table.append((7, 2j, 60.0, False))

    expected_mask_final = [True, False, True, False, True, True, True]
    assert_mask_matches(table, expected_mask_final)


def test_resize_when_capacity_full_with_gaps():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=False)

    table.delete(list(range(9)))

    assert len(table) == 1

    initial_capacity = len(table._valid_rows)

    table.append((11, 5j, 75.0, True))

    new_capacity = len(table._valid_rows)
    assert new_capacity > initial_capacity, \
        f"Expected resize, but capacity stayed {initial_capacity}"


def test_no_resize_with_compact_enabled():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=True)

    table.delete(list(range(9)))

    assert len(table) == 1

    initial_capacity = len(table._valid_rows)

    new_data = generate_test_data(3, start_id=11)
    table.extend(new_data)

    new_capacity = len(table._valid_rows)
    assert new_capacity <= initial_capacity * 2, \
        "Unexpected massive resize with auto_compact enabled"


def test_resize_when_extend_exceeds_capacity():
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=10, compact=False)

    table.delete([0, 2, 4])

    initial_capacity = len(table._valid_rows)

    large_data = generate_test_data(20, start_id=100)
    table.extend(large_data)

    new_capacity = len(table._valid_rows)
    assert new_capacity > initial_capacity


def test_extend_fills_from_last_valid_position():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    table.delete([2, 4, 6])

    new_data = generate_test_data(3, start_id=20)
    table.extend(new_data)

    assert_data_at_positions(table, [10, 11, 12], [20, 21, 22])


def test_multiple_extends_with_gaps():
    data = generate_test_data(5, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=20)

    table.extend(generate_test_data(3, start_id=10))
    assert len(table) == 8

    table.delete([2, 4, 6])
    assert len(table) == 5

    table.extend(generate_test_data(2, start_id=20))
    assert len(table) == 7

    table.delete([0, 1])
    assert len(table) == 5

    table.extend(generate_test_data(4, start_id=30))
    assert len(table) == 9


def test_append_and_extend_mixed_with_gaps():
    table = CTable(RowModel, expected_size=20)

    for i in range(5):
        table.append((i + 1, complex(i), float(i * 10), True))

    assert len(table) == 5

    table.extend(generate_test_data(5, start_id=10))
    assert len(table) == 10

    table.delete([1, 3, 5, 7, 9])
    assert len(table) == 5

    table.append((100, 0j, 50.0, False))
    assert len(table) == 6

    table.extend(generate_test_data(3, start_id=200))
    assert len(table) == 9


def test_fill_gaps_completely_then_extend():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    table.delete(list(range(0, 10, 2)))
    assert len(table) == 5

    table.extend(generate_test_data(5, start_id=20))
    assert len(table) == 10


def test_delete_all_then_extend():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15)

    table.delete(list(range(10)))
    assert len(table) == 0

    new_data = generate_test_data(5, start_id=100)
    table.extend(new_data)

    assert len(table) == 5


def test_sparse_table_with_many_gaps():
    data = generate_test_data(20, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=30)

    to_delete = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
    table.delete(to_delete)

    assert len(table) == 5

    table.extend(generate_test_data(10, start_id=100))

    assert len(table) == 15


def test_alternating_insert_delete_pattern():
    table = CTable(RowModel, expected_size=50)

    for cycle in range(5):
        table.extend(generate_test_data(10, start_id=cycle * 100))

        current_len = len(table)
        if current_len >= 5:
            to_delete = list(range(0, min(5, current_len)))
            table.delete(to_delete)


def test_manual_compact_before_extend():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15, compact=False)

    table.delete([1, 3, 5, 7, 9])
    assert len(table) == 5

    table.compact()

    expected_mask = [True] * 5 + [False] * 10
    assert_mask_matches(table, expected_mask)

    table.extend(generate_test_data(3, start_id=20))
    assert len(table) == 8


def test_auto_compact_on_extend():
    data = generate_test_data(10, start_id=1)
    table = CTable(RowModel, new_data=data, expected_size=15, compact=True)

    table.delete(list(range(0, 8)))
    assert len(table) == 2

    table.extend(generate_test_data(10, start_id=100))

    assert len(table) == 12


def test_data_integrity_after_gap_operations():
    data1 = [(1, 1j, 10.0, True), (2, 2j, 20.0, False), (3, 3j, 30.0, True)]
    table = CTable(RowModel, new_data=data1, expected_size=10)

    table.delete(1)

    assert table.row[0].id[0] == 1
    assert table.row[1].id[0] == 3

    data2 = [(10, 10j, 100.0, True), (11, 11j, 110.0, False)]
    table.extend(data2)

    assert table.row[0].id[0] == 1
    assert table.row[1].id[0] == 3
    assert table.row[2].id[0] == 10
    assert table.row[3].id[0] == 11


def test_complex_scenario_full_workflow():
    table = CTable(RowModel, expected_size=20, compact=False)

    table.extend(generate_test_data(10, start_id=1))
    assert len(table) == 10

    table.delete([0, 2, 4, 6, 8])
    assert len(table) == 5

    table.append((100, 0j, 50.0, True))
    table.append((101, 1j, 60.0, False))
    assert len(table) == 7

    table.extend(generate_test_data(5, start_id=200))
    assert len(table) == 12

    table.delete([3, 7, 10])
    assert len(table) == 9

    table.extend(generate_test_data(3, start_id=300))
    assert len(table) == 12

    assert table.nrows == 12
    assert table.ncols == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
