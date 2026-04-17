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


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


def generate_test_data(n_rows: int, start_id: int = 1) -> list:
    return [(start_id + i, complex(i, -i), float((i * 7) % 100), bool(i % 2)) for i in range(n_rows)]


def get_valid_mask(table: CTable) -> np.ndarray:
    return np.array(table._valid_rows[: len(table._valid_rows)], dtype=bool)


def assert_mask_matches(table: CTable, expected_mask: list):
    actual = get_valid_mask(table)[: len(expected_mask)]
    np.testing.assert_array_equal(
        actual,
        np.array(expected_mask, dtype=bool),
        err_msg=f"Mask mismatch.\nExpected: {expected_mask}\nGot: {actual}",
    )


def assert_data_at_positions(table: CTable, positions: list, expected_ids: list):
    for pos, expected_id in zip(positions, expected_ids, strict=False):
        actual_id = int(table._cols["id"][pos])
        assert actual_id == expected_id, f"Position {pos}: expected ID {expected_id}, got {actual_id}"


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_gap_fill_mask_and_positions():
    """extend and append fill from last valid position; mask is updated correctly."""
    # extend after deletions: mask and physical positions
    t = CTable(Row, new_data=generate_test_data(7, 1), expected_size=10)
    t.delete([0, 2, 4, 6])
    assert_mask_matches(t, [False, True, False, True, False, True, False])
    assert len(t) == 3
    t.extend(generate_test_data(3, 8))
    assert_mask_matches(t, [False, True, False, True, False, True, True, True, True])
    assert len(t) == 6
    assert_data_at_positions(t, [6, 7, 8], [8, 9, 10])

    # append fills from last valid position, not into holes
    t2 = CTable(Row, new_data=generate_test_data(5, 1), expected_size=10)
    t2.delete([1, 3])
    assert_mask_matches(t2, [True, False, True, False, True])
    t2.append((6, 1j, 50.0, True))
    assert_mask_matches(t2, [True, False, True, False, True, True])
    t2.append((7, 2j, 60.0, False))
    assert_mask_matches(t2, [True, False, True, False, True, True, True])

    # extend fills from last valid position when there's enough capacity
    t3 = CTable(Row, new_data=generate_test_data(10, 1), expected_size=15)
    t3.delete([2, 4, 6])
    t3.extend(generate_test_data(3, 20))
    assert_data_at_positions(t3, [10, 11, 12], [20, 21, 22])


def test_resize_behavior():
    """Resize triggered when capacity is full; compact=True avoids massive resize."""
    # compact=False: append beyond capacity must resize
    t = CTable(Row, new_data=generate_test_data(10, 1), expected_size=10, compact=False)
    t.delete(list(range(9)))
    assert len(t) == 1
    initial_cap = len(t._valid_rows)
    t.append((11, 5j, 75.0, True))
    assert len(t._valid_rows) > initial_cap

    # compact=True: no massive resize after deletions + extend
    t2 = CTable(Row, new_data=generate_test_data(10, 1), expected_size=10, compact=True)
    t2.delete(list(range(9)))
    assert len(t2) == 1
    initial_cap2 = len(t2._valid_rows)
    t2.extend(generate_test_data(3, 11))
    assert len(t2._valid_rows) <= initial_cap2 * 2

    # extend exceeding capacity always resizes regardless of compact
    t3 = CTable(Row, new_data=generate_test_data(5, 1), expected_size=10, compact=False)
    t3.delete([0, 2, 4])
    initial_cap3 = len(t3._valid_rows)
    t3.extend(generate_test_data(20, 100))
    assert len(t3._valid_rows) > initial_cap3


def test_mixed_append_extend_with_gaps():
    """Multiple extends, appends, and deletes interleaved; lengths stay correct."""
    # Multiple extends with intermediate deletions
    t = CTable(Row, expected_size=20)
    t.extend(generate_test_data(5, 1))
    t.extend(generate_test_data(3, 10))
    assert len(t) == 8
    t.delete([2, 4, 6])
    assert len(t) == 5
    t.extend(generate_test_data(2, 20))
    assert len(t) == 7
    t.delete([0, 1])
    assert len(t) == 5
    t.extend(generate_test_data(4, 30))
    assert len(t) == 9

    # append + extend mixed, delete all then re-extend
    t2 = CTable(Row, expected_size=20)
    for i in range(5):
        t2.append((i + 1, complex(i), float(i * 10), True))
    assert len(t2) == 5
    t2.extend(generate_test_data(5, 10))
    assert len(t2) == 10
    t2.delete([1, 3, 5, 7, 9])
    assert len(t2) == 5
    t2.append((100, 0j, 50.0, False))
    assert len(t2) == 6
    t2.extend(generate_test_data(3, 200))
    assert len(t2) == 9

    # Fill all gaps then extend; delete all then extend from scratch
    t3 = CTable(Row, new_data=generate_test_data(10, 1), expected_size=15)
    t3.delete(list(range(0, 10, 2)))
    assert len(t3) == 5
    t3.extend(generate_test_data(5, 20))
    assert len(t3) == 10

    t4 = CTable(Row, new_data=generate_test_data(10, 1), expected_size=15)
    t4.delete(list(range(10)))
    assert len(t4) == 0
    t4.extend(generate_test_data(5, 100))
    assert len(t4) == 5


def test_compact_behavior():
    """Manual compact consolidates mask; auto-compact keeps data correct after extend."""
    # Manual compact: valid rows packed to front, extend fills after them
    t = CTable(Row, new_data=generate_test_data(10, 1), expected_size=15, compact=False)
    t.delete([1, 3, 5, 7, 9])
    assert len(t) == 5
    t.compact()
    assert_mask_matches(t, [True] * 5 + [False] * 10)
    t.extend(generate_test_data(3, 20))
    assert len(t) == 8

    # Auto-compact: table stays consistent after heavy deletions + extend
    t2 = CTable(Row, new_data=generate_test_data(10, 1), expected_size=15, compact=True)
    t2.delete(list(range(0, 8)))
    assert len(t2) == 2
    t2.extend(generate_test_data(10, 100))
    assert len(t2) == 12


def test_complex_scenarios():
    """Sparse gaps, alternating cycles, data integrity, and full workflow."""
    # Sparse table: many scattered deletions then bulk extend
    t = CTable(Row, new_data=generate_test_data(20, 1), expected_size=30)
    t.delete([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18])
    assert len(t) == 5
    t.extend(generate_test_data(10, 100))
    assert len(t) == 15

    # Alternating extend/delete cycles
    t2 = CTable(Row, expected_size=50)
    for cycle in range(5):
        t2.extend(generate_test_data(10, cycle * 100))
        current_len = len(t2)
        if current_len >= 5:
            t2.delete(list(range(0, min(5, current_len))))

    # Data integrity: correct row values survive delete + extend
    t3 = CTable(
        Row, new_data=[(1, 1j, 10.0, True), (2, 2j, 20.0, False), (3, 3j, 30.0, True)], expected_size=10
    )
    t3.delete(1)
    assert t3.row[0].id[0] == 1
    assert t3.row[1].id[0] == 3
    t3.extend([(10, 10j, 100.0, True), (11, 11j, 100.0, False)])
    assert t3.row[0].id[0] == 1
    assert t3.row[1].id[0] == 3
    assert t3.row[2].id[0] == 10
    assert t3.row[3].id[0] == 11

    # Full workflow
    t4 = CTable(Row, expected_size=20, compact=False)
    t4.extend(generate_test_data(10, 1))
    assert len(t4) == 10
    t4.delete([0, 2, 4, 6, 8])
    assert len(t4) == 5
    t4.append((100, 0j, 50.0, True))
    t4.append((101, 1j, 60.0, False))
    assert len(t4) == 7
    t4.extend(generate_test_data(5, 200))
    assert len(t4) == 12
    t4.delete([3, 7, 10])
    assert len(t4) == 9
    t4.extend(generate_test_data(3, 300))
    assert len(t4) == 12
    assert t4.nrows == 12
    assert t4.ncols == 4


if __name__ == "__main__":
    pytest.main(["-v", __file__])
