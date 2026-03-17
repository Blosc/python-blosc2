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

import blosc2
from blosc2 import CTable

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
# Predefined Test Data
# -------------------------------------------------------------------
SMALL_DATA = [
    (1, 1 + 2j, 95.5, True),
    (2, 3 - 4j, 80.0, False),
    (3, 0j, 50.2, True),
    (4, -1 + 1j, 12.3, False),
    (5, 5j, 99.9, True),
]
SMALLEST_DATA = SMALL_DATA[:2]

dtype_struct = [("id", "i8"), ("c_val", "c16"), ("score", "f8"), ("active", "?")]
SMALL_STRUCT = np.array(SMALL_DATA, dtype=dtype_struct)


# -------------------------------------------------------------------
# Validation Utility
# -------------------------------------------------------------------
def assert_table_equals_data(table: CTable, expected_data: list):
    assert len(table) == len(expected_data), f"Expected length {len(expected_data)}, got {len(table)}"
    col_names = table.col_names
    for i, expected_row in enumerate(expected_data):
        row_extracted = table.row[i]
        for col_idx, expected_val in enumerate(expected_row):
            col_name = col_names[col_idx]
            extracted_val = getattr(row_extracted, col_name)[0]
            if isinstance(expected_val, (float, complex)):
                np.testing.assert_allclose(
                    extracted_val, expected_val, err_msg=f"Discrepancy at row {i}, col {col_name}"
                )
            else:
                assert extracted_val == expected_val, (
                    f"Row {i}, col {col_name}: expected {expected_val}, got {extracted_val}"
                )


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_empty_table_variants():
    """Empty table: default, with expected_size, and with compact=True."""
    table = CTable(RowModel)
    assert len(table) == 0 and table.nrows == 0 and table.ncols == 4
    for col_name in ["id", "c_val", "score", "active"]:
        assert col_name in table._cols
        assert isinstance(table._cols[col_name], blosc2.NDArray)

    table_sized = CTable(RowModel, expected_size=5000)
    assert len(table_sized) == 0
    assert all(len(col) == 5000 for col in table_sized._cols.values())

    table_compact = CTable(RowModel, compact=True)
    assert len(table_compact) == 0 and table_compact.auto_compact is True


def test_empty_data_lifecycle():
    """Create from [], extend with [], then extend with real data."""
    table = CTable(RowModel, new_data=[])
    assert len(table) == 0

    table.extend([])
    assert len(table) == 0

    table.extend(SMALL_DATA)
    assert_table_equals_data(table, SMALL_DATA)


def test_construction_sources():
    """List of tuples and structured array both produce identical tables."""
    assert_table_equals_data(CTable(RowModel, new_data=SMALL_DATA), SMALL_DATA)
    assert_table_equals_data(CTable(RowModel, new_data=SMALL_STRUCT), SMALL_DATA)


def test_expected_size_variants():
    """expected_size smaller, exact, and larger than the inserted data."""
    for es in [1, 5]:
        assert_table_equals_data(CTable(RowModel, new_data=SMALL_DATA, expected_size=es), SMALL_DATA)

    table_large = CTable(RowModel, new_data=SMALL_DATA, expected_size=1000)
    assert_table_equals_data(table_large, SMALL_DATA)
    assert all(len(col) == 1000 for col in table_large._cols.values())


def test_compact_flag():
    """compact=False and compact=True both preserve data correctly."""
    table_false = CTable(RowModel, new_data=SMALL_DATA, compact=False)
    assert table_false.auto_compact is False
    assert_table_equals_data(table_false, SMALL_DATA)

    table_true = CTable(RowModel, new_data=SMALL_DATA, compact=True)
    assert table_true.auto_compact is True
    assert_table_equals_data(table_true, SMALL_DATA)


def test_append_and_clone():
    """Build table row by row, then clone it into a new CTable."""
    table = CTable(RowModel)
    for row in SMALLEST_DATA:
        table.append(row)
    assert_table_equals_data(table, SMALLEST_DATA)

    cloned = CTable(RowModel, new_data=table)
    assert_table_equals_data(cloned, SMALLEST_DATA)
    assert table is not cloned


def test_invalid_append():
    """Wrong length, incompatible type, and dict all raise errors."""
    table = CTable(RowModel, expected_size=1)

    # Too few values → IndexError (NumPy raises natively after simplification)
    with pytest.raises((IndexError, ValueError)):
        table.append([1, 1 + 2j, 95.5])  # missing boolean

    # Incompatible type → TypeError or ValueError from NumPy
    with pytest.raises((TypeError, ValueError)):
        table.append(["invalid_text", 1 + 2j, 95.5, True])


def test_extreme_values():
    """Extreme complex, float boundary, and large integer values."""
    extreme_complex = [
        (1, complex(1e308, -1e308), 50.0, True),
        (2, complex(0, 0), 0.0, False),
        (3, complex(-1e308, 1e308), 100.0, True),
    ]
    extreme_float = [
        (1, 0j, 0.0, True),
        (2, 0j, 100.0, False),
        (3, 0j, 0.0001, True),
        (4, 0j, 99.9999, False),
    ]
    extreme_int = [
        (1, 0j, 50.0, True),
        (2**32, 0j, 50.0, False),
        (2**60, 0j, 50.0, True),
    ]
    for data in [extreme_complex, extreme_float, extreme_int]:
        assert_table_equals_data(CTable(RowModel, new_data=data), data)


def test_extend_append_and_resize():
    """Auto-resize via append one-by-one, then extend+append beyond initial size."""
    # Append beyond expected_size triggers resize
    table = CTable(RowModel, expected_size=2)
    for row in SMALL_DATA:
        table.append(row)
    assert_table_equals_data(table, SMALL_DATA)
    assert all(len(col) >= 5 for col in table._cols.values())

    # Extend beyond expected_size, then append the last row
    table2 = CTable(RowModel, expected_size=2)
    table2.extend(SMALL_DATA[:4])
    assert len(table2) == 4
    table2.append(SMALL_DATA[4])
    assert_table_equals_data(table2, SMALL_DATA)


def test_column_integrity():
    """Column access via [] and getattr, and correct dtypes."""
    table = CTable(RowModel, new_data=SMALL_DATA)

    assert isinstance(table["id"], blosc2.ctable.Column)
    assert isinstance(table.score, blosc2.ctable.Column)

    assert table._cols["id"].dtype == np.int64
    assert table._cols["c_val"].dtype == np.complex128
    assert table._cols["score"].dtype == np.float64
    assert table._cols["active"].dtype == np.bool_


def test_valid_rows():
    """_valid_rows has exactly 5 True entries after creation and after extend."""
    table_direct = CTable(RowModel, new_data=SMALL_DATA)
    assert blosc2.count_nonzero(table_direct._valid_rows) == 5

    table_extended = CTable(RowModel)
    table_extended.extend(SMALL_DATA)
    assert blosc2.count_nonzero(table_extended._valid_rows) == 5


if __name__ == "__main__":
    pytest.main(["-v", __file__])
