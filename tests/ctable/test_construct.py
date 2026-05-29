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
    if not expected_data:
        return
    col_names = table.col_names
    # Transpose: expected_data is list-of-rows → list-of-columns
    expected_cols = list(zip(*expected_data, strict=False))
    for col_idx, col_name in enumerate(col_names):
        actual = table[col_name][:]
        expected = expected_cols[col_idx]
        if isinstance(expected[0], (float, complex)):
            np.testing.assert_allclose(actual, expected, err_msg=f"col {col_name}")
        else:
            np.testing.assert_array_equal(actual, expected, err_msg=f"col {col_name}")


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------


def test_empty_table_variants():
    """Empty table: default, with expected_size, and with compact=True."""
    table = CTable(Row)
    assert len(table) == 0
    assert table.nrows == 0
    assert table.ncols == 4
    for col_name in ["id", "c_val", "score", "active"]:
        assert col_name in table._cols
        assert isinstance(table._cols[col_name], blosc2.NDArray)

    table_sized = CTable(Row, expected_size=5000)
    assert len(table_sized) == 0
    assert all(len(col) == 5000 for col in table_sized._cols.values())

    table_compact = CTable(Row, compact=True)
    assert len(table_compact) == 0
    assert table_compact.auto_compact is True


def test_empty_data_lifecycle():
    """Create from [], extend with [], then extend with real data."""
    table = CTable(Row, new_data=[])
    assert len(table) == 0

    table.extend([])
    assert len(table) == 0

    table.extend(SMALL_DATA)
    assert_table_equals_data(table, SMALL_DATA)


def test_construction_variants():
    """Sources (list, structured array), expected_size, and compact flag."""
    # list of tuples and structured array produce identical tables
    assert_table_equals_data(CTable(Row, new_data=SMALL_DATA), SMALL_DATA)
    assert_table_equals_data(CTable(Row, new_data=SMALL_STRUCT), SMALL_DATA)

    # expected_size smaller than data → resize; larger → preallocated
    for es in [1, 5]:
        assert_table_equals_data(CTable(Row, new_data=SMALL_DATA, expected_size=es), SMALL_DATA)
    table_large = CTable(Row, new_data=SMALL_DATA, expected_size=1000)
    assert_table_equals_data(table_large, SMALL_DATA)
    assert all(len(col) == 1000 for col in table_large._cols.values())

    # compact flag is stored and data is intact
    table_false = CTable(Row, new_data=SMALL_DATA, compact=False)
    assert table_false.auto_compact is False
    assert_table_equals_data(table_false, SMALL_DATA)

    table_true = CTable(Row, new_data=SMALL_DATA, compact=True)
    assert table_true.auto_compact is True
    assert_table_equals_data(table_true, SMALL_DATA)


def test_append_and_clone():
    """Build table row by row, then clone it into a new CTable."""
    table = CTable(Row)
    for row in SMALLEST_DATA:
        table.append(row)
    assert_table_equals_data(table, SMALLEST_DATA)

    cloned = CTable(Row, new_data=table)
    assert_table_equals_data(cloned, SMALLEST_DATA)
    assert table is not cloned


def test_invalid_append():
    """Constraint violation and incompatible type both raise errors."""
    table = CTable(Row, expected_size=1)

    # Constraint violation: id must be >= 0
    with pytest.raises(ValueError):
        table.append((-1, 1 + 2j, 95.5, True))

    # Constraint violation: score must be <= 100
    with pytest.raises(ValueError):
        table.append((1, 1 + 2j, 150.0, True))

    # Incompatible type for id: string cannot be coerced to int
    with pytest.raises((TypeError, ValueError)):
        table.append(["invalid_text", 1 + 2j, 95.5, True])


def test_extreme_values():
    """Extreme complex, float boundary, and large integer values in one table."""
    # Combine all extremes into one table to avoid repeated CTable construction
    extreme_data = [
        (1, complex(1e308, -1e308), 0.0, True),
        (2**32, 0j, 100.0, False),
        (2**60, complex(-1e308, 1e308), 0.0001, True),
        (4, 0j, 99.9999, False),
    ]
    assert_table_equals_data(CTable(Row, new_data=extreme_data), extreme_data)


def test_extend_append_and_resize():
    """Auto-resize via append one-by-one, then extend+append beyond initial size."""
    # Append beyond expected_size triggers resize
    table = CTable(Row, expected_size=2)
    for row in SMALL_DATA:
        table.append(row)
    assert_table_equals_data(table, SMALL_DATA)
    assert all(len(col) >= 5 for col in table._cols.values())

    # Extend beyond expected_size, then append the last row
    table2 = CTable(Row, expected_size=2)
    table2.extend(SMALL_DATA[:4])
    assert len(table2) == 4
    table2.append(SMALL_DATA[4])
    assert_table_equals_data(table2, SMALL_DATA)


def test_column_integrity():
    """Column access via [] and getattr, and correct dtypes."""
    table = CTable(Row, new_data=SMALL_DATA)

    assert isinstance(table["id"], blosc2.ctable.Column)
    assert isinstance(table.score, blosc2.ctable.Column)

    assert table._cols["id"].dtype == np.int64
    assert table._cols["c_val"].dtype == np.complex128
    assert table._cols["score"].dtype == np.float64
    assert table._cols["active"].dtype == np.bool_


def test_valid_rows():
    """_valid_rows has exactly 5 True entries after creation and after extend."""
    table_direct = CTable(Row, new_data=SMALL_DATA)
    assert blosc2.count_nonzero(table_direct._valid_rows) == 5

    table_extended = CTable(Row)
    table_extended.extend(SMALL_DATA)
    assert blosc2.count_nonzero(table_extended._valid_rows) == 5


def test_fixed_columns_share_aligned_grid():
    """Fixed-size scalar columns share one chunk/block grid so lazy
    expressions over mixed dtypes can take the fast_eval path."""

    @dataclass
    class MixedRow:
        a: float = blosc2.field(blosc2.float32(), default=0.0)
        b: float = blosc2.field(blosc2.float32(), default=0.0)
        c: float = blosc2.field(blosc2.float32(), default=0.0)
        d: float = blosc2.field(blosc2.float64(), default=0.0)
        n: int = blosc2.field(blosc2.int32(), default=0)

    table = CTable(MixedRow, expected_size=4_000_000)
    grids = {(table._cols[name].chunks, table._cols[name].blocks) for name in table.col_names}
    assert len(grids) == 1, f"columns are not aligned: {grids}"

    # The _valid_rows mask shares the same grid so where() keeps the fast path.
    assert table._valid_rows.chunks == table._cols["a"].chunks
    assert table._valid_rows.blocks == table._cols["a"].blocks

    # The table exposes the shared grid via .chunks / .blocks.
    assert table.chunks == table._cols["a"].chunks
    assert table.blocks == table._cols["a"].blocks
    assert table.chunks is not None
    assert len(table.chunks) == 1


def test_wide_string_column_excluded_from_aligned_grid():
    """Very wide fixed-width string columns keep per-dtype sizing instead of
    inheriting the shared grid (which would produce huge chunks)."""

    @dataclass
    class WideRow:
        a: float = blosc2.field(blosc2.float32(), default=0.0)
        d: float = blosc2.field(blosc2.float64(), default=0.0)
        s: str = blosc2.field(blosc2.string(max_length=50000), default="")

    table = CTable(WideRow, expected_size=4_000_000)
    # Numeric columns share the aligned grid...
    assert table._cols["a"].chunks == table._cols["d"].chunks
    assert table._cols["a"].blocks == table._cols["d"].blocks
    # ...but the wide string column does not (it would blow up the chunk size).
    assert table._cols["s"].chunks != table._cols["a"].chunks
    # The reported grid reflects the aligned (numeric) set.
    assert table.chunks == table._cols["a"].chunks


def test_from_arrow_aligns_columns_and_mask():
    """The Arrow-import path (used by parquet-to-blosc2) aligns fixed-size
    columns and the _valid_rows mask on a single shared grid."""
    pa = pytest.importorskip("pyarrow")

    n = 200_000
    rng = np.random.default_rng(0)
    tbl = pa.table(
        {
            "tips": pa.array(rng.random(n).astype("f4")),
            "km": pa.array(rng.random(n).astype("f4")),
            "lon": pa.array(-rng.random(n)),  # float64: previously misaligned
        }
    )
    table = CTable.from_arrow(tbl.schema, tbl.to_batches(), capacity_hint=n)
    grids = {(table._cols[name].chunks, table._cols[name].blocks) for name in table.col_names}
    grids.add((table._valid_rows.chunks, table._valid_rows.blocks))
    assert len(grids) == 1, f"from_arrow did not align grids: {grids}"


def test_empty_table_grid_properties():
    """A table with no fixed-size scalar columns reports None for chunks/blocks
    only when there is nothing to align."""

    @dataclass
    class ScalarRow:
        x: int = blosc2.field(blosc2.int64(), default=0)

    table = CTable(ScalarRow, expected_size=1000)
    assert table.chunks is not None
    assert table.blocks is not None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
