#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest
import numpy as np
import blosc2
from blosc2 import CTable
from pydantic import BaseModel, Field
from typing import Annotated, TypeVar

# NOTE: Make sure to import your CTable, NumpyDtype correctly


# -------------------------------------------------------------------
# 1. Row Type Definition for Testing
# -------------------------------------------------------------------
RowT = TypeVar("RowT", bound=BaseModel)


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class MaxLen:
    def __init__(self, length: int):
        self.length = int(length)

class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


# -------------------------------------------------------------------
# 2. Predefined Test Data
# -------------------------------------------------------------------
# Small data (5 rows)
SMALL_DATA = [
    (1, 1 + 2j, 95.5, True),
    (2, 3 - 4j, 80.0, False),
    (3, 0j, 50.2, True),
    (4, -1 + 1j, 12.3, False),
    (5, 5j, 99.9, True)
]

# Medium data (20 rows) - the limit you mentioned for manual testing
MEDIUM_DATA = [
    (i, complex(i, -i), float(i % 100), bool(i % 2))
    for i in range(1, 21)
]

# Large data (10,000 rows) - to test scalability
LARGE_DATA = [
    (i, complex(i % 100, -(i % 50)), float((i * 7) % 100), bool(i % 3))
    for i in range(1, 10_001)
]

# Small structured array
dtype_struct = [('id', 'i8'), ('c_val', 'c16'), ('score', 'f8'), ('active', '?')]
SMALL_STRUCT = np.array(SMALL_DATA, dtype=dtype_struct)
MEDIUM_STRUCT = np.array(MEDIUM_DATA, dtype=dtype_struct)
LARGE_STRUCT = np.array(LARGE_DATA, dtype=dtype_struct)


# -------------------------------------------------------------------
# 3. Validation Utilities
# -------------------------------------------------------------------
def assert_table_equals_data(table: CTable, expected_data: list):
    """
    Helper function to check that the table length and the data
    extracted via _RowIndexer are exactly as expected.
    """
    assert len(table) == len(expected_data), \
        f"Expected length {len(expected_data)}, got {len(table)}"

    # Get column names to map indices to attribute access
    col_names = table.col_names

    # Check row by row
    for i in range(len(expected_data)):
        # row_extracted is now a CTable view of 1 row
        row_extracted = table.row[i]
        expected_row = expected_data[i]

        for col_idx, expected_val in enumerate(expected_row):
            col_name = col_names[col_idx]

            # Access the column dynamically and get the 0-th element of the view
            # e.g., row_extracted.id[0]
            extracted_val = getattr(row_extracted, col_name)[0]

            # Compare floats and complex numbers with tolerance
            if isinstance(expected_val, (float, complex)):
                np.testing.assert_allclose(
                    extracted_val, expected_val,
                    err_msg=f"Discrepancy at row {i}, col {col_name} (idx {col_idx})"
                )
            else:
                assert extracted_val == expected_val, \
                    f"Row {i}, col {col_name}: expected {expected_val}, got {extracted_val}"


# -------------------------------------------------------------------
# 4. Basic Construction Tests
# -------------------------------------------------------------------

def test_create_empty_table():
    """Empty table with no initial data."""
    table = CTable(RowModel)

    assert len(table) == 0
    assert table.nrows == 0
    assert table.ncols == 4

    for col_name in ["id", "c_val", "score", "active"]:
        assert col_name in table._cols
        assert isinstance(table._cols[col_name], blosc2.NDArray)


def test_create_empty_table_with_expected_size():
    """Empty table specifying expected_size."""
    table = CTable(RowModel, expected_size=5000)

    assert len(table) == 0
    assert table.nrows == 0
    # Verify that the internal arrays have the correct size
    for col in table._cols.values():
        assert len(col) == 5000


def test_create_empty_table_with_compact():
    """Empty table with auto-compaction enabled."""
    table = CTable(RowModel, compact=True)

    assert len(table) == 0
    assert table.auto_compact == True


# -------------------------------------------------------------------
# 5. Empty List Tests
# -------------------------------------------------------------------

def test_create_from_empty_list():
    """Create table passing an empty list as new_data."""
    table = CTable(RowModel, new_data=[])

    assert len(table) == 0
    assert table.nrows == 0


def test_extend_empty_list():
    """Extend table with an empty list."""
    table = CTable(RowModel)
    table.extend([])

    assert len(table) == 0


def test_extend_after_empty_list():
    """Create with an empty list and then extend with data."""
    table = CTable(RowModel, new_data=[])
    table.extend(SMALL_DATA)

    assert_table_equals_data(table, SMALL_DATA)


# -------------------------------------------------------------------
# 6. Small Data Tests (5 rows)
# -------------------------------------------------------------------

def test_create_from_small_list():
    """Create from a small list of tuples."""
    table = CTable(RowModel, new_data=SMALL_DATA)
    assert_table_equals_data(table, SMALL_DATA)


def test_create_from_small_struct():
    """Create from a small structured array."""
    table = CTable(RowModel, new_data=SMALL_STRUCT)
    assert_table_equals_data(table, SMALL_DATA)


def test_append_small_one_by_one():
    """Manual construction row by row (5 rows)."""
    table = CTable(RowModel)

    for row in SMALL_DATA:
        table.append(row)

    assert_table_equals_data(table, SMALL_DATA)


# -------------------------------------------------------------------
# 7. Medium Data Tests (20 rows)
# -------------------------------------------------------------------

def test_create_from_medium_list():
    """Create from a medium list (20 rows)."""
    table = CTable(RowModel, new_data=MEDIUM_DATA)
    assert_table_equals_data(table, MEDIUM_DATA)


def test_create_from_medium_struct():
    """Create from a medium structured array."""
    table = CTable(RowModel, new_data=MEDIUM_STRUCT)
    assert_table_equals_data(table, MEDIUM_DATA)


def test_append_medium_one_by_one():
    """Manual append of 20 rows."""
    table = CTable(RowModel)

    for row in MEDIUM_DATA:
        table.append(row)

    assert_table_equals_data(table, MEDIUM_DATA)


def test_extend_medium():
    """Extend with 20 rows at once."""
    table = CTable(RowModel)
    table.extend(MEDIUM_DATA)
    assert_table_equals_data(table, MEDIUM_DATA)


# -------------------------------------------------------------------
# 8. Large Data Tests (10,000 rows)
# -------------------------------------------------------------------

def test_create_from_large_list():
    """Create from a large list (10k rows)."""
    table = CTable(RowModel, new_data=LARGE_DATA)
    assert len(table) == 10_000
    # Verify only a few rows to avoid saturation
    assert table.id[0] == 1  # first id
    assert table.id[9999] == 10_000  # last id


def test_create_from_large_struct():
    """Create from a large structured array."""
    table = CTable(RowModel, new_data=LARGE_STRUCT)
    assert len(table) == 10_000


def test_extend_large():
    """Extend with 10k rows."""
    table = CTable(RowModel)
    table.extend(LARGE_DATA)
    assert len(table) == 10_000


# -------------------------------------------------------------------
# 9. Expected Size Tests
# -------------------------------------------------------------------

def test_expected_size_smaller_than_data():
    """Expected size smaller than the data to be inserted (must resize)."""
    table = CTable(RowModel, new_data=MEDIUM_DATA, expected_size=10)
    assert_table_equals_data(table, MEDIUM_DATA)


def test_expected_size_larger_than_data():
    """Expected size larger than the data (plenty of space available)."""
    table = CTable(RowModel, new_data=SMALL_DATA, expected_size=1000)
    assert_table_equals_data(table, SMALL_DATA)
    # Internal arrays should have size 1000
    for col in table._cols.values():
        assert len(col) == 1000


def test_expected_size_exact():
    """Expected size exact to the number of data rows."""
    table = CTable(RowModel, new_data=SMALL_DATA, expected_size=5)
    assert_table_equals_data(table, SMALL_DATA)


# -------------------------------------------------------------------
# 10. Auto-Compact Tests
# -------------------------------------------------------------------

def test_create_with_compact_false():
    """Create with compact=False."""
    table = CTable(RowModel, new_data=MEDIUM_DATA, compact=False)
    assert table.auto_compact == False
    assert_table_equals_data(table, MEDIUM_DATA)


def test_create_with_compact_true():
    """Create with compact=True."""
    table = CTable(RowModel, new_data=MEDIUM_DATA, compact=True)
    assert table.auto_compact == True
    assert_table_equals_data(table, MEDIUM_DATA)


# -------------------------------------------------------------------
# 11. Construction from Another CTable Tests
# -------------------------------------------------------------------

def test_create_from_another_ctable():
    """Create CTable from another CTable."""
    base_table = CTable(RowModel, new_data=SMALL_DATA)
    new_table = CTable(RowModel, new_data=base_table)

    assert_table_equals_data(new_table, SMALL_DATA)
    assert base_table is not new_table


def test_create_from_filtered_ctable():
    """Create CTable from another filtered CTable."""
    base_table = CTable(RowModel, new_data=MEDIUM_DATA)
    # Filter rows where active == True
    filtered = base_table.where(base_table.active == True)

    new_table = CTable(RowModel, new_data=filtered)
    assert len(new_table) == len(filtered)


# -------------------------------------------------------------------
# 12. Type Validation Tests
# -------------------------------------------------------------------

def test_invalid_data_length_append():
    """Append with an incorrect number of fields."""
    table = CTable(RowModel)
    invalid_row = [1, 1 + 2j, 95.5]  # Missing boolean

    with pytest.raises(ValueError, match="Expected 4 values"):
        table.append(invalid_row)


def test_invalid_data_type_append():
    """Append with an incompatible data type."""
    table = CTable(RowModel)
    invalid_row = ["invalid_text", 1 + 2j, 95.5, True]

    with pytest.raises((TypeError, ValueError)):
        table.append(invalid_row)


def test_invalid_data_dict_append():
    """Append with a dictionary (not supported)."""
    table = CTable(RowModel)
    row_dict = {"id": 1, "c_val": 1 + 2j, "score": 95.5, "active": True}

    with pytest.raises(TypeError, match="Dictionaries are not supported"):
        table.append(row_dict)


# -------------------------------------------------------------------
# 13. Default Values Tests
# -------------------------------------------------------------------

def test_default_values():
    """Check that model default values are respected."""
    # If append receives only some values, defaults should apply
    # (this depends on your exact implementation, adjust as needed)
    table = CTable(RowModel)
    # Your current implementation requires all values in append
    # This test is more conceptual - you can adjust it


# -------------------------------------------------------------------
# 14. Extreme Values Tests
# -------------------------------------------------------------------

def test_extreme_values_complex():
    """Extreme complex values."""
    extreme_data = [
        (1, complex(1e308, -1e308), 50.0, True),
        (2, complex(0, 0), 0.0, False),
        (3, complex(-1e308, 1e308), 100.0, True),
    ]
    table = CTable(RowModel, new_data=extreme_data)
    assert_table_equals_data(table, extreme_data)


def test_extreme_values_float():
    """Float values at allowed limits (0-100)."""
    extreme_data = [
        (1, 0j, 0.0, True),
        (2, 0j, 100.0, False),
        (3, 0j, 0.0001, True),
        (4, 0j, 99.9999, False),
    ]
    table = CTable(RowModel, new_data=extreme_data)
    assert_table_equals_data(table, extreme_data)


def test_extreme_values_int():
    """Large integer values."""
    extreme_data = [
        (1, 0j, 50.0, True),
        (2 ** 32, 0j, 50.0, False),
        (2 ** 60, 0j, 50.0, True),
    ]
    table = CTable(RowModel, new_data=extreme_data)
    assert_table_equals_data(table, extreme_data)


# -------------------------------------------------------------------
# 15. Combination Tests
# -------------------------------------------------------------------

def test_append_then_extend():
    """Individual append followed by a massive extend."""
    table = CTable(RowModel)
    table.append(SMALL_DATA[0])
    assert len(table) == 1

    table.extend(SMALL_DATA[1:])
    assert_table_equals_data(table, SMALL_DATA)


def test_extend_then_append():
    """Massive extend followed by an individual append."""
    table = CTable(RowModel)
    table.extend(SMALL_DATA[:4])
    assert len(table) == 4

    table.append(SMALL_DATA[4])
    assert_table_equals_data(table, SMALL_DATA)


def test_multiple_extends():
    """Multiple consecutive extends."""
    table = CTable(RowModel)
    table.extend(SMALL_DATA[:2])
    table.extend(SMALL_DATA[2:4])
    table.extend(SMALL_DATA[4:])
    assert_table_equals_data(table, SMALL_DATA)


# -------------------------------------------------------------------
# 16. Auto-Resize Tests
# -------------------------------------------------------------------

def test_auto_resize_on_append():
    """Verify that append auto-resizes when full."""
    table = CTable(RowModel, expected_size=2)

    # Fill beyond expected_size
    for row in SMALL_DATA:
        table.append(row)

    assert_table_equals_data(table, SMALL_DATA)
    # Internal arrays should have been resized
    assert all(len(col) >= 5 for col in table._cols.values())


def test_auto_resize_on_extend():
    """Verify that extend auto-resizes."""
    table = CTable(RowModel, expected_size=3)
    table.extend(MEDIUM_DATA)

    assert_table_equals_data(table, MEDIUM_DATA)


# -------------------------------------------------------------------
# 17. Column Integrity Tests
# -------------------------------------------------------------------

def test_column_access():
    """Verify direct access to columns."""
    table = CTable(RowModel, new_data=SMALL_DATA)

    # Access via __getitem__
    id_col = table["id"]
    assert isinstance(id_col, blosc2.ctable.Column)

    # Access via __getattr__
    score_col = table.score
    assert isinstance(score_col, blosc2.ctable.Column)


def test_column_types():
    """Verify that columns have the correct types."""
    table = CTable(RowModel, new_data=SMALL_DATA)

    assert table._cols["id"].dtype == np.int64
    assert table._cols["c_val"].dtype == np.complex128
    assert table._cols["score"].dtype == np.float64
    assert table._cols["active"].dtype == np.bool_


# -------------------------------------------------------------------
# 18. Valid Rows Tests
# -------------------------------------------------------------------

def test_valid_rows_initialization():
    """Verify that _valid_rows initializes correctly."""
    table = CTable(RowModel, new_data=SMALL_DATA)

    # The first 5 positions must be True
    valid_count = blosc2.count_nonzero(table._valid_rows)
    assert valid_count == 5


def test_valid_rows_after_extend():
    """Verify _valid_rows after extend."""
    table = CTable(RowModel)
    table.extend(MEDIUM_DATA)

    valid_count = blosc2.count_nonzero(table._valid_rows)
    assert valid_count == 20


if __name__ == "__main__":
    pytest.main(["-v", __file__])
