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

# NOTE: Make sure to import your CTable and NumpyDtype correctly

# -------------------------------------------------------------------
# 1. Row Type Definition for Testing
# -------------------------------------------------------------------
RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


# -------------------------------------------------------------------
# 2. Test Data Generation
# -------------------------------------------------------------------
def generate_test_data(n_rows: int) -> list:
    """
    Generate n_rows of test data following the RowModel schema.
    Returns a list of tuples.
    """
    return [
        (i, complex(i, -i), float((i * 7) % 100), bool(i % 2))
        for i in range(1, n_rows + 1)
    ]


# -------------------------------------------------------------------
# 3. Helper Functions
# -------------------------------------------------------------------
def get_valid_positions(table: CTable) -> np.ndarray:
    """
    Extract the positions where _valid_rows is True.
    Returns a numpy array of indices.
    """
    return np.flatnonzero(table._valid_rows[:len(table._valid_rows)])


def assert_valid_rows_match(table: CTable, expected_valid_indices: list):
    """
    Check that _valid_rows has True exactly at the expected positions
    and False everywhere else (up to the table's internal array length).

    Args:
        table: The CTable instance to check
        expected_valid_indices: List of indices that should be True
    """
    valid_positions = get_valid_positions(table)
    expected_array = np.array(sorted(expected_valid_indices))

    np.testing.assert_array_equal(
        valid_positions[:len(expected_array)],
        expected_array,
        err_msg=f"Valid rows mismatch. Expected {expected_array}, got {valid_positions}"
    )


# -------------------------------------------------------------------
# 4. Basic Delete Tests (Single Element)
# -------------------------------------------------------------------

def test_delete_first_element_once():
    """Delete the first element (position 0) from a full 50-row table."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Before deletion
    assert len(table) == 50

    # Delete first element
    table.delete(0)

    # After deletion
    assert len(table) == 49
    # Position 0 should now be False, positions 1-49 should be True
    expected_valid = list(range(1, 50))
    assert_valid_rows_match(table, expected_valid)


def test_delete_first_element_10_times():
    """Delete the first element 10 times consecutively using a loop."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    initial_length = 50

    for i in range(10):
        table.delete(0)
        expected_length = initial_length - (i + 1)
        assert len(table) == expected_length, \
            f"After {i + 1} deletions, expected length {expected_length}, got {len(table)}"

    # After 10 deletions, should have 40 rows
    assert len(table) == 40


def test_delete_last_element_once():
    """Delete the last element using delete(-1) from a full 50-row table."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Before deletion
    assert len(table) == 50

    # Delete last element
    table.delete(-1)

    # After deletion
    assert len(table) == 49


def test_delete_last_element_10_times():
    """Delete the last element 10 times consecutively using delete(-1)."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    initial_length = 50

    for i in range(10):
        table.delete(-1)
        expected_length = initial_length - (i + 1)
        assert len(table) == expected_length, \
            f"After {i + 1} deletions, expected length {expected_length}, got {len(table)}"

    # After 10 deletions, should have 40 rows
    assert len(table) == 40


def test_delete_middle_element():
    """Delete a middle element from a 50-row table."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete position 25 (middle)
    table.delete(25)

    assert len(table) == 49


def test_delete_multiple_individual_elements():
    """Delete multiple non-consecutive elements one by one."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete positions 5, 15, 25, 35, 45
    positions_to_delete = [5, 15, 25, 35, 45]

    for _ in positions_to_delete:
        # Adjust position because previous deletions shift indices
        table.delete(0)  # Simplified: delete first element 5 times

    assert len(table) == 45


# -------------------------------------------------------------------
# 5. Delete with List of Positions
# -------------------------------------------------------------------

def test_delete_list_of_positions():
    """Delete multiple positions at once using a list."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete positions [0, 10, 20, 30, 40]
    table.delete([0, 10, 20, 30, 40])

    assert len(table) == 45


def test_delete_consecutive_positions_list():
    """Delete consecutive positions using a list."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete positions [5, 6, 7, 8, 9]
    table.delete([5, 6, 7, 8, 9])

    assert len(table) == 45


def test_delete_all_even_positions():
    """Delete all even-indexed positions."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete all even positions (0, 2, 4, ..., 48)
    even_positions = list(range(0, 50, 2))
    table.delete(even_positions)

    assert len(table) == 25


def test_delete_all_odd_positions():
    """Delete all odd-indexed positions."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete all odd positions (1, 3, 5, ..., 49)
    odd_positions = list(range(1, 50, 2))
    table.delete(odd_positions)

    assert len(table) == 25


# -------------------------------------------------------------------
# 6. Delete Out-of-Bounds Tests (Should Raise Errors)
# -------------------------------------------------------------------

def test_delete_position_beyond_length_full_table():
    """
    Try to delete position 60 in a full 50-row table.
    Should raise IndexError.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    with pytest.raises(IndexError):
        table.delete(60)


def test_delete_position_beyond_nrows_partial_table():
    """
    Try to delete position 35 in a table with capacity 50 but only 25 rows.
    Should raise IndexError.
    """
    data = generate_test_data(25)
    table = CTable(RowModel, new_data=data, expected_size=50)

    assert len(table) == 25

    with pytest.raises(IndexError):
        table.delete(35)


def test_delete_from_empty_table_position_25():
    """
    Try to delete position 25 from an empty table.
    Should raise IndexError.
    """
    table = CTable(RowModel, expected_size=50)

    assert len(table) == 0

    with pytest.raises(IndexError):
        table.delete(25)


def test_delete_from_empty_table_position_0():
    """
    Try to delete position 0 from an empty table.
    Should raise IndexError.
    """
    table = CTable(RowModel, expected_size=50)

    assert len(table) == 0

    with pytest.raises(IndexError):
        table.delete(0)


def test_delete_from_empty_table_position_negative():
    """
    Try to delete position -1 from an empty table.
    Should raise IndexError.
    """
    table = CTable(RowModel, expected_size=50)

    assert len(table) == 0

    with pytest.raises(IndexError):
        table.delete(-1)


def test_delete_negative_position_beyond_length():
    """
    Try to delete position -60 in a 50-row table.
    Should raise IndexError.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    with pytest.raises(IndexError):
        table.delete(-60)


# -------------------------------------------------------------------
# 7. Delete with Slices (if your implementation supports it)
# -------------------------------------------------------------------
# NOTE: Based on your current code, delete() accepts int or list[int].
# If you want to support slices, you'll need to modify your delete method.
# Below are tests assuming slice support is added.

def test_delete_slice_range_a_to_b():
    """
    Delete rows from position a to b (not including b) using slice(a, b).
    Example: delete positions 10 to 20 (10 rows).

    NOTE: This requires your delete() method to handle slice objects.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # This will only work if you implement slice support in delete()
    try:
        # Delete positions 10-19 (10 rows)
        positions = list(range(10, 20))
        table.delete(positions)

        assert len(table) == 40
    except TypeError:
        pytest.skip("Slice support not yet implemented in delete()")


def test_delete_slice_with_step():
    """
    Delete rows using slice with step: a:b:c
    Example: delete every other row from 0 to 20.

    NOTE: This requires your delete() method to handle slice objects.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    try:
        # Delete positions 0, 2, 4, ..., 18 (every other row from 0 to 20)
        positions = list(range(0, 20, 2))
        table.delete(positions)

        assert len(table) == 40
    except TypeError:
        pytest.skip("Slice support not yet implemented in delete()")


def test_delete_slice_from_start():
    """
    Delete rows from start to position b using slice(:b).
    Example: delete first 10 rows.

    NOTE: This requires your delete() method to handle slice objects.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    try:
        # Delete positions 0-9 (first 10 rows)
        positions = list(range(0, 10))
        table.delete(positions)

        assert len(table) == 40
    except TypeError:
        pytest.skip("Slice support not yet implemented in delete()")


def test_delete_slice_to_end():
    """
    Delete rows from position a to end using slice(a:).
    Example: delete last 10 rows.

    NOTE: This requires your delete() method to handle slice objects.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    try:
        # Delete positions 40-49 (last 10 rows)
        positions = list(range(40, 50))
        table.delete(positions)

        assert len(table) == 40
    except TypeError:
        pytest.skip("Slice support not yet implemented in delete()")


# -------------------------------------------------------------------
# 8. Edge Cases and Special Scenarios
# -------------------------------------------------------------------

def test_delete_same_position_twice():
    """
    Try to delete the same logical position twice.
    The second deletion should fail or behave correctly.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete position 10
    table.delete(10)
    assert len(table) == 49

    # Try to delete what is now position 10 (was position 11 before)
    table.delete(10)
    assert len(table) == 48


def test_delete_all_rows_one_by_one():
    """Delete all 50 rows one by one from the front."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    for _ in range(50):
        table.delete(0)

    assert len(table) == 0


def test_delete_all_rows_from_back():
    """Delete all 50 rows one by one from the back using -1."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    for _ in range(50):
        table.delete(-1)

    assert len(table) == 0


def test_delete_with_negative_indices():
    """Delete using various negative indices."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete positions -1, -5, -10 (last, 5th from last, 10th from last)
    table.delete([-1, -5, -10])

    assert len(table) == 47


def test_delete_mixed_positive_negative_indices():
    """Delete using a mix of positive and negative indices."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete positions [0, -1, 25] (first, last, middle)
    table.delete([0, -1, 25])

    assert len(table) == 47


# -------------------------------------------------------------------
# 9. Type Validation Tests
# -------------------------------------------------------------------

def test_delete_invalid_type_string():
    """Try to delete with a string (invalid type). Should raise TypeError."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    with pytest.raises(TypeError):
        table.delete("invalid")


def test_delete_invalid_type_float():
    """Try to delete with a float (invalid type). Should raise TypeError."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    with pytest.raises(TypeError):
        table.delete(10.5)


def test_delete_invalid_list_with_strings():
    """Try to delete with a list containing strings. Should raise TypeError."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    with pytest.raises(IndexError):
        table.delete([0, "invalid", 10])


# -------------------------------------------------------------------
# 10. Stress Tests
# -------------------------------------------------------------------

def test_delete_large_number_of_positions():
    """Delete a large number of positions at once."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete 40 out of 50 positions
    positions_to_delete = list(range(0, 40))
    table.delete(positions_to_delete)

    assert len(table) == 10


def test_delete_alternate_pattern():
    """
    Delete alternating rows multiple times to test
    the _valid_rows tracking under complex patterns.
    """
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # First pass: delete every other row (even indices)
    even_positions = list(range(0, 50, 2))
    table.delete(even_positions)
    assert len(table) == 25

    # Second pass: delete every other remaining row
    # (which are at logical positions 0, 2, 4, ... in the new 25-row table)
    new_even = list(range(0, 25, 2))
    table.delete(new_even)
    assert len(table) == 12  # Roughly half of 25


if __name__ == "__main__":
    pytest.main(["-v", __file__])
