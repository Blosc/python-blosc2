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
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    assert isinstance(t2, CTable)


def test_from_arrow_row_count():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    assert len(t2) == 10


def test_from_arrow_column_names():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    assert t2.col_names == ["id", "score", "active", "label"]


def test_from_arrow_int_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    np.testing.assert_array_equal(t2["id"][:], t["id"][:])


def test_from_arrow_float_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    np.testing.assert_allclose(t2["score"][:], t["score"][:])


def test_from_arrow_bool_values():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    np.testing.assert_array_equal(t2["active"][:], t["active"][:])


def test_from_arrow_string_values():
    # Without string_max_length, scalar strings become vlstring columns.
    # Accessing [:] on a vlstring column returns a Python list, not an ndarray.
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    assert list(t2["label"][:]) == t["label"][:].tolist()


def test_from_arrow_empty_table():
    schema = pa.schema(
        [
            pa.field("id", pa.int64()),
            pa.field("val", pa.float64()),
        ]
    )
    at = pa.table({"id": pa.array([], type=pa.int64()), "val": pa.array([], type=pa.float64())})
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert len(t) == 0
    assert t.col_names == ["id", "val"]


def test_from_arrow_timestamp_roundtrip_and_query():
    arr = pa.array(
        [np.datetime64("2025-01-01T00:00:00", "us"), None, np.datetime64("2025-01-02T00:00:00", "us")],
        type=pa.timestamp("us"),
    )
    at = pa.Table.from_arrays([arr], names=["ts"])

    t = CTable.from_arrow(at.schema, at.to_batches())

    assert isinstance(t._schema.columns_by_name["ts"].spec, blosc2.schema.timestamp)
    assert t.ts[0] == np.datetime64("2025-01-01T00:00:00", "us")
    np.testing.assert_array_equal(
        t.ts[:],
        np.array(["2025-01-01T00:00:00", "NaT", "2025-01-02T00:00:00"], dtype="datetime64[us]"),
    )
    assert len(t[t.ts >= np.datetime64("2025-01-02", "us")]) == 1

    out = t.to_arrow()
    assert out.schema.field("ts").type == pa.timestamp("us")
    assert out.column("ts").null_count == 1
    assert out.column("ts").to_pylist()[0] == arr.to_pylist()[0]


def test_from_arrow_roundtrip():
    """to_arrow then from_arrow preserves all values."""
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    t2 = CTable.from_arrow(at.schema, at.to_batches())
    for name in ["id", "score", "active"]:
        np.testing.assert_array_equal(t2[name][:], t[name][:])
    # label is re-imported as vlstring (no string_max_length given) → compare as lists
    assert list(t2["label"][:]) == t["label"][:].tolist()


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
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert len(t) == 3
    assert t.col_names == list(at.column_names)


def test_from_arrow_string_default_is_vlstring():
    """Without string_max_length, scalar string columns become vlstring (variable-length)."""
    at = pa.table({"name": pa.array(["hi", "hello world", "!"], type=pa.string())})
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert t["name"].is_varlen_scalar
    assert t["name"].dtype is None
    assert list(t["name"][:]) == ["hi", "hello world", "!"]


def test_from_arrow_string_fixed_width_with_max_length():
    """Passing string_max_length gives a fixed-width NDArray string column."""
    at = pa.table({"name": pa.array(["hi", "hello world", "!"], type=pa.string())})
    t = CTable.from_arrow(at.schema, at.to_batches(), string_max_length=32)
    # "hello world" is 11 chars — stored dtype must accommodate string_max_length
    assert t["name"].dtype.itemsize // 4 >= 11
    assert not t["name"].is_varlen_scalar
    assert t["name"][:].tolist() == ["hi", "hello world", "!"]


def test_from_arrow_list_struct_nullable_values_roundtrip():
    nutrient_type = pa.struct(
        [
            pa.field("name", pa.string()),
            pa.field("value", pa.float64()),
        ]
    )
    at = pa.table(
        {
            "id": pa.array([1, 2, 3], type=pa.int64()),
            "nutriments": pa.array(
                [
                    [{"name": "fat", "value": 1.5}, {"name": "salt", "value": 0.2}],
                    None,
                    [{"name": "energy", "value": 42.0}],
                ],
                type=pa.list_(nutrient_type),
            ),
        }
    )
    t = CTable.from_arrow(at.schema, at.to_batches())
    assert t[0].nutriments == [{"name": "fat", "value": 1.5}, {"name": "salt", "value": 0.2}]
    assert t[1].nutriments is None
    assert t[2].nutriments == [{"name": "energy", "value": 42.0}]


def test_from_arrow_unsupported_type_raises():
    at = pa.table({"duration": pa.array([1, 2, 3], type=pa.duration("s"))})
    with pytest.raises(TypeError, match="No blosc2 spec"):
        CTable.from_arrow(at.schema, at.to_batches())


if __name__ == "__main__":
    pytest.main(["-v", __file__])
