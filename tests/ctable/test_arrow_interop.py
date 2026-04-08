#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.to_arrow() and CTable.from_arrow()."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable

pa = pytest.importorskip("pyarrow")


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)
    label: str = blosc2.field(blosc2.string(max_length=16), default="")


DATA10 = [(i, float(i * 10 % 100), i % 2 == 0, f"r{i}") for i in range(10)]


# ===========================================================================
# to_arrow()
# ===========================================================================


def test_to_arrow_returns_pyarrow_table():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert isinstance(at, pa.Table)


def test_to_arrow_column_names():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert at.column_names == ["id", "score", "active", "label"]


def test_to_arrow_row_count():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert len(at) == 10


def test_to_arrow_int_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    np.testing.assert_array_equal(at["id"].to_pylist(), [r[0] for r in DATA10])


def test_to_arrow_float_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    np.testing.assert_allclose(at["score"].to_pylist(), [r[1] for r in DATA10])


def test_to_arrow_bool_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert at["active"].to_pylist() == [r[2] for r in DATA10]


def test_to_arrow_string_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert at["label"].to_pylist() == [r[3] for r in DATA10]


def test_to_arrow_string_type():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    assert at.schema.field("label").type == pa.string()


def test_to_arrow_skips_deleted_rows():
    t = CTable(Row, new_data=DATA10)
    t.delete([0, 1])
    at = t.to_arrow()
    assert len(at) == 8
    assert at["id"].to_pylist() == list(range(2, 10))


def test_to_arrow_empty_table():
    t = CTable(Row)
    at = t.to_arrow()
    assert len(at) == 0
    assert at.column_names == ["id", "score", "active", "label"]


def test_to_arrow_select_view():
    t = CTable(Row, new_data=DATA10)
    at = t.select(["id", "score"]).to_arrow()
    assert at.column_names == ["id", "score"]
    assert len(at) == 10


def test_to_arrow_where_view():
    t = CTable(Row, new_data=DATA10)
    at = t.where(t["id"] > 4).to_arrow()
    assert len(at) == 5


# ===========================================================================
# from_arrow()
# ===========================================================================


def test_from_arrow_returns_ctable():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    assert isinstance(t2, CTable)


def test_from_arrow_row_count():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    assert len(t2) == 10


def test_from_arrow_column_names():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    assert t2.col_names == ["id", "score", "active", "label"]


def test_from_arrow_int_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    np.testing.assert_array_equal(t2["id"].to_numpy(), t["id"].to_numpy())


def test_from_arrow_float_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    np.testing.assert_allclose(t2["score"].to_numpy(), t["score"].to_numpy())


def test_from_arrow_bool_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    np.testing.assert_array_equal(t2["active"].to_numpy(), t["active"].to_numpy())


def test_from_arrow_string_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at)
    assert t2["label"].to_numpy().tolist() == t["label"].to_numpy().tolist()


def test_from_arrow_empty_table():
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("val", pa.float64()),
        ]
    )
    at = pa.table({"id": pa.array([], type=pa.int64()), "val": pa.array([], type=pa.float64())})
    t = CTable.from_arrow(at)
    assert len(t) == 0
    assert t.col_names == ["id", "val"]


def test_from_arrow_roundtrip():
    """to_arrow then from_arrow preserves all values."""
    t = CTable(Row, new_data=DATA10)
    t2 = CTable.from_arrow(t.to_arrow())
    for name in ["id", "score", "active"]:
        np.testing.assert_array_equal(t2[name].to_numpy(), t[name].to_numpy())
    assert t2["label"].to_numpy().tolist() == t["label"].to_numpy().tolist()


def test_from_arrow_all_numeric_types():
    """All integer and float Arrow types map to correct blosc2 specs."""
    at = pa.table(
        {
            "i8": pa.array([1, 2, 3], type=pa.int8()),
            "i16": pa.array([1, 2, 3], type=pa.int16()),
            "i32": pa.array([1, 2, 3], type=pa.int32()),
            "i64": pa.array([1, 2, 3], type=pa.int64()),
            "u8": pa.array([1, 2, 3], type=pa.uint8()),
            "u16": pa.array([1, 2, 3], type=pa.uint16()),
            "u32": pa.array([1, 2, 3], type=pa.uint32()),
            "u64": pa.array([1, 2, 3], type=pa.uint64()),
            "f32": pa.array([1.0, 2.0, 3.0], type=pa.float32()),
            "f64": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        }
    )
    t = CTable.from_arrow(at)
    assert len(t) == 3
    assert t.col_names == list(at.column_names)


def test_from_arrow_string_max_length():
    """String max_length is set from the longest value in the data."""
    at = pa.table({"name": pa.array(["hi", "hello world", "!"], type=pa.string())})
    t = CTable.from_arrow(at)
    # "hello world" is 11 chars — stored dtype must accommodate it
    assert t["name"].dtype.itemsize // 4 >= 11


def test_from_arrow_unsupported_type_raises():
    at = pa.table({"ts": pa.array([1, 2, 3], type=pa.timestamp("s"))})
    with pytest.raises(TypeError, match="No blosc2 spec"):
        CTable.from_arrow(at)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
