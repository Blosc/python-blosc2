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


def generate_test_data(n_rows: int) -> list:
    return [(i, complex(i, -i), float((i * 7) % 100), bool(i % 2)) for i in range(1, n_rows + 1)]


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_delete_single_element():
    """First, last, middle deletion once; and repeated deletion from front/back."""
    data = generate_test_data(50)

    # Delete first
    t = CTable(RowModel, new_data=data, expected_size=50)
    t.delete(0)
    assert len(t) == 49
    assert not t._valid_rows[0]

    # Delete last
    t2 = CTable(RowModel, new_data=data, expected_size=50)
    t2.delete(-1)
    assert len(t2) == 49

    # Delete middle
    t3 = CTable(RowModel, new_data=data, expected_size=50)
    t3.delete(25)
    assert len(t3) == 49

    # Delete first 10 times in a row
    t4 = CTable(RowModel, new_data=data, expected_size=50)
    for i in range(10):
        t4.delete(0)
        assert len(t4) == 50 - (i + 1)
    assert len(t4) == 40

    # Delete last 10 times in a row
    t5 = CTable(RowModel, new_data=data, expected_size=50)
    for i in range(10):
        t5.delete(-1)
        assert len(t5) == 50 - (i + 1)
    assert len(t5) == 40


def test_delete_list_of_positions():
    """Scattered, consecutive, even, odd, and slice-equivalent list deletions."""
    data = generate_test_data(50)

    # Scattered
    t = CTable(RowModel, new_data=data, expected_size=50)
    t.delete([0, 10, 20, 30, 40])
    assert len(t) == 45

    # Consecutive block
    t2 = CTable(RowModel, new_data=data, expected_size=50)
    t2.delete([5, 6, 7, 8, 9])
    assert len(t2) == 45

    # All even positions
    t3 = CTable(RowModel, new_data=data, expected_size=50)
    t3.delete(list(range(0, 50, 2)))
    assert len(t3) == 25

    # All odd positions
    t4 = CTable(RowModel, new_data=data, expected_size=50)
    t4.delete(list(range(1, 50, 2)))
    assert len(t4) == 25

    # Slice-equivalent: range(10, 20)
    t5 = CTable(RowModel, new_data=data, expected_size=50)
    t5.delete(list(range(10, 20)))
    assert len(t5) == 40

    # Slice with step: range(0, 20, 2)
    t6 = CTable(RowModel, new_data=data, expected_size=50)
    t6.delete(list(range(0, 20, 2)))
    assert len(t6) == 40

    # First 10 rows
    t7 = CTable(RowModel, new_data=data, expected_size=50)
    t7.delete(list(range(0, 10)))
    assert len(t7) == 40

    # Last 10 rows
    t8 = CTable(RowModel, new_data=data, expected_size=50)
    t8.delete(list(range(40, 50)))
    assert len(t8) == 40


def test_delete_out_of_bounds():
    """All IndexError scenarios: full table, partial table, empty table, negative."""
    data = generate_test_data(50)

    # Beyond length on full table
    t = CTable(RowModel, new_data=data, expected_size=50)
    with pytest.raises(IndexError):
        t.delete(60)
    with pytest.raises(IndexError):
        t.delete(-60)

    # Beyond nrows on partial table (capacity 50, only 25 rows)
    t2 = CTable(RowModel, new_data=generate_test_data(25), expected_size=50)
    assert len(t2) == 25
    with pytest.raises(IndexError):
        t2.delete(35)

    # Empty table: positions 0, 25, -1 all raise
    for pos in [0, 25, -1]:
        empty = CTable(RowModel, expected_size=50)
        assert len(empty) == 0
        with pytest.raises(IndexError):
            empty.delete(pos)


def test_delete_edge_cases():
    """Same position twice, all rows front/back, negative and mixed indices."""
    data = generate_test_data(50)

    # Same logical position twice: second delete hits what was position 11
    t = CTable(RowModel, new_data=data, expected_size=50)
    t.delete(10)
    assert len(t) == 49
    t.delete(10)
    assert len(t) == 48

    # Delete all rows from the front one by one
    t2 = CTable(RowModel, new_data=data, expected_size=50)
    for _ in range(50):
        t2.delete(0)
    assert len(t2) == 0

    # Delete all rows from the back one by one
    t3 = CTable(RowModel, new_data=data, expected_size=50)
    for _ in range(50):
        t3.delete(-1)
    assert len(t3) == 0

    # Negative indices list
    t4 = CTable(RowModel, new_data=data, expected_size=50)
    t4.delete([-1, -5, -10])
    assert len(t4) == 47

    # Mixed positive and negative indices
    t5 = CTable(RowModel, new_data=data, expected_size=50)
    t5.delete([0, -1, 25])
    assert len(t5) == 47


def test_delete_invalid_types():
    """string, float, and list-with-strings all raise errors."""
    data = generate_test_data(50)

    t = CTable(RowModel, new_data=data, expected_size=50)
    with pytest.raises(TypeError):
        t.delete("invalid")
    with pytest.raises(TypeError):
        t.delete(10.5)
    with pytest.raises(TypeError):
        t.delete([0, "invalid", 10])


def test_delete_stress():
    """Large batch deletion and alternating multi-pass pattern."""
    data = generate_test_data(50)

    # Delete 40 out of 50 at once
    t = CTable(RowModel, new_data=data, expected_size=50)
    t.delete(list(range(0, 40)))
    assert len(t) == 10

    # Alternating two-pass deletion
    t2 = CTable(RowModel, new_data=data, expected_size=50)
    t2.delete(list(range(0, 50, 2)))  # delete all even -> 25 remain
    assert len(t2) == 25
    t2.delete(list(range(0, 25, 2)))  # delete every other of remaining -> ~12
    assert len(t2) == 12


if __name__ == "__main__":
    pytest.main(["-v", __file__])
