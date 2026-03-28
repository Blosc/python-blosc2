#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

from blosc2 import CTable


# --- Basic model setup for tests ---
class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)


def generate_test_data(n_rows: int) -> list:
    return [(i, float(i)) for i in range(n_rows)]


def test_compact_empty_table():
    """Test compact() on a completely empty table (no data)."""
    table = CTable(RowModel, expected_size=100)

    assert len(table) == 0

    # Should not raise any error
    table.compact()

    # Capacity might have drastically reduced, but the logical table must remain empty
    assert len(table) == 0
    # Verify that if data is added later, it works correctly
    table.append((1, 10.0))
    assert len(table) == 1
    assert table.id[0] == 1


def test_compact_full_table():
    """Test compact() on a completely full table (no holes or free space)."""
    data = generate_test_data(50)
    table = CTable(RowModel, new_data=data, expected_size=50)

    assert len(table) == 50
    initial_capacity = len(table._valid_rows)

    # Should not raise any error or change the logical state
    table.compact()

    assert len(table) == 50
    # Capacity should not have changed because it was already full
    assert len(table._valid_rows) == initial_capacity

    # Verify data integrity
    assert table.id[0] == 0
    assert table.id[-1] == 49


def test_compact_already_compacted_table():
    """Test compact() on a table that has free space but no holes (contiguous data)."""
    data = generate_test_data(20)
    # Large expected_size to ensure free space at the end
    table = CTable(RowModel, new_data=data, expected_size=100)

    assert len(table) == 20

    # Execute compact. Since data is already contiguous, the table might reduce
    # its size due to the < len//2 while loop, but it shouldn't fail.
    table.compact()

    assert len(table) == 20

    # Verify that data remains in place
    for i in range(20):
        assert table.id[i] == i

    # Validate that all True values are consecutive at the beginning
    mask = table._valid_rows[: len(table._valid_rows)]
    assert np.all(mask[:20])
    if len(mask) > 20:
        assert not np.any(mask[20:])


def test_compact_with_holes():
    """Test compact() on a table with high fragmentation (holes)."""
    data = generate_test_data(30)
    table = CTable(RowModel, new_data=data, expected_size=50)

    # Delete sparsely: leave only [0, 5, 10, 15, 20, 25]
    to_delete = [i for i in range(30) if i % 5 != 0]
    table.delete(to_delete)

    assert len(table) == 6

    # Execute compact
    table.compact()

    assert len(table) == 6

    # Verify that the correct data survived and moved to the beginning
    expected_ids = [0, 5, 10, 15, 20, 25]
    for i, exp_id in enumerate(expected_ids):
        # Through the logical view (Column wrapper)
        assert table.id[i] == exp_id
        # Through the physical blosc2 array (to ensure compact worked)
        assert table._cols["id"][i] == exp_id

    # Verify physical mask: first 6 must be True, the rest False
    mask = table._valid_rows[: len(table._valid_rows)]
    assert np.all(mask[:6])
    if len(mask) > 6:
        assert not np.any(mask[6:])


def test_compact_all_deleted():
    """Test compact() on a table where absolutely all rows have been deleted."""
    data = generate_test_data(20)
    table = CTable(RowModel, new_data=data, expected_size=20)

    # Delete everything
    table.delete(list(range(20)))
    assert len(table) == 0

    # Should handle empty arrays correctly
    table.compact()

    assert len(table) == 0

    # Check that we can write to it again
    table.append((99, 99.0))
    assert len(table) == 1
    assert table.id[0] == 99


def test_compact_multiple_times():
    """Calling compact() multiple times in a row must not corrupt data or crash."""
    data = generate_test_data(10)
    table = CTable(RowModel, new_data=data, expected_size=20)

    table.delete([1, 3, 5, 7, 9])  # 5 elements remaining

    # Compact 3 times in a row
    table.compact()
    table.compact()
    table.compact()

    assert len(table) == 5
    assert list(table.id) == [0, 2, 4, 6, 8]
