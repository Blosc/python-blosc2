#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.to_arrow() and CTable.from_arrow()."""

import datetime
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


def test_from_arrow_column_cparams(tmp_path):
    at = pa.table(
        {
            "x": pa.array([1.1, 2.2, 3.3], type=pa.float64()),
            "y": pa.array([1, 2, 3], type=pa.int64()),
        }
    )
    urlpath = tmp_path / "trunc.b2d"
    t = CTable.from_arrow(
        at.schema,
        at.to_batches(),
        urlpath=str(urlpath),
        column_cparams={
            "x": {
                "codec": blosc2.Codec.ZSTD.value,
                "clevel": 5,
                "filters": [blosc2.Filter.TRUNC_PREC.value, blosc2.Filter.SHUFFLE.value],
                "filters_meta": [32, 0],
            }
        },
    )

    assert t._cols["x"].cparams.filters[:2] == [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
    assert t._cols["x"].cparams.filters_meta[:2] == [32, 0]
    assert t._cols["y"].cparams.filters[-1] == blosc2.Filter.SHUFFLE
    t.close()

    reopened = CTable.open(str(urlpath), mode="r")
    assert reopened._cols["x"].cparams.filters[:2] == [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]


def test_from_arrow_column_cparams_nested_struct(tmp_path):
    # Regression: column_cparams with TRUNC_PREC must be applied to leaf columns
    # produced by struct flattening, not silently dropped.
    struct_type = pa.struct(
        [
            pa.field("lon", pa.float64()),
            pa.field("lat", pa.float64()),
        ]
    )
    at = pa.table(
        {
            "pos": pa.array(
                [{"lon": -87.6, "lat": 41.8}, {"lon": -87.7, "lat": 41.9}],
                type=struct_type,
            ),
            "fare": pa.array([10.0, 20.0], type=pa.float32()),
        }
    )
    trunc_cparams = {
        "codec": blosc2.Codec.ZSTD.value,
        "clevel": 5,
        "typesize": 8,
        "filters": [blosc2.Filter.TRUNC_PREC.value, blosc2.Filter.SHUFFLE.value],
        "filters_meta": [22, 0],
    }
    t = CTable.from_arrow(
        at.schema,
        at.to_batches(),
        urlpath=str(tmp_path / "trunc_nested.b2d"),
        column_cparams={"pos.lon": trunc_cparams, "pos.lat": trunc_cparams},
    )
    assert t.col_names == ["pos.lon", "pos.lat", "fare"]
    assert t._cols["pos.lon"].cparams.filters[:2] == [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
    assert t._cols["pos.lon"].cparams.filters_meta[:2] == [22, 0]
    assert t._cols["pos.lat"].cparams.filters[:2] == [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
    # fare is float32, no TRUNC_PREC requested
    assert blosc2.Filter.TRUNC_PREC not in t._cols["fare"].cparams.filters
    t.close()


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


def test_from_arrow_list_struct_timestamp_roundtrip():
    event_type = pa.struct(
        [
            pa.field("when", pa.timestamp("ms")),
            pa.field("value", pa.float64()),
        ]
    )
    at = pa.table(
        {
            "events": pa.array(
                [
                    [{"when": datetime.datetime(2020, 1, 1), "value": 1.5}],
                    None,
                ],
                type=pa.list_(event_type),
            )
        }
    )

    t = CTable.from_arrow(at.schema, at.to_batches())
    assert t[0].events == [{"when": 1577836800000, "value": 1.5}]
    assert t[1].events is None

    out = t.to_arrow()
    assert out.schema.field("events").type == pa.list_(event_type)
    assert out.column("events").to_pylist()[0][0]["when"].isoformat() == "2020-01-01T00:00:00"


def test_from_arrow_unsupported_type_raises():
    at = pa.table({"duration": pa.array([1, 2, 3], type=pa.duration("s"))})
    with pytest.raises(TypeError, match="No blosc2 spec"):
        CTable.from_arrow(at.schema, at.to_batches())


@pytest.mark.parametrize("batch_size", [1, 7, 100, 333, 1000, 1500])
def test_chunk_aligned_writer_matches_direct_write(batch_size):
    """The import-time buffered writer reproduces a plain element-by-element
    write regardless of how appends straddle chunk boundaries."""
    from blosc2.ctable import _ChunkAlignedWriter

    n = 4321
    chunk_len = 1000
    data = np.arange(n, dtype=np.float64)

    arr = blosc2.empty((n,), dtype=np.float64, chunks=(chunk_len,))
    writer = _ChunkAlignedWriter(arr, chunk_len)
    for start in range(0, n, batch_size):
        writer.append(data[start : start + batch_size])
    writer.flush()

    np.testing.assert_array_equal(arr[:], data)


def test_from_arrow_dictionary_codes_use_aligned_grid():
    """Imported dictionary columns create their int32 codes at full capacity
    on the aligned grid, not the tiny 4096-row default (which caused a
    create-then-resize and thousands of micro-chunks)."""
    n = 500_000
    rng = np.random.default_rng(0)
    labels = np.array(["alpha", "beta", "gamma", "delta", "epsilon"])
    schema = pa.schema(
        [
            pa.field("a", pa.float32()),
            pa.field("c", pa.dictionary(pa.int32(), pa.string())),
        ]
    )
    a = pa.array(rng.random(n).astype("f4"))
    c = pa.array(labels[rng.integers(0, len(labels), n)]).dictionary_encode()
    t = CTable.from_arrow(schema, [pa.record_batch([a, c], schema=schema)], capacity_hint=n)

    codes = t._cols["c"].codes
    # Codes share the numeric column's (aligned) grid and are not micro-chunked.
    assert codes.chunks == t._cols["a"].chunks
    assert codes.schunk.nchunks < 10
    assert list(t["c"][:5]) == c.to_pylist()[:5]


def test_from_arrow_variable_batches_roundtrip():
    """Variable-sized Arrow batches that straddle the column chunk grid import
    losslessly (exercises the chunk-aligned write buffer)."""

    @dataclass
    class Row:
        a: float = blosc2.field(blosc2.float32(), default=0.0)
        d: float = blosc2.field(blosc2.float64(), default=0.0)

    rng = np.random.default_rng(0)
    sizes = [854_973, 996_662, 1_002_093, 145_272]  # uneven, cross 1.25M chunks
    n = sum(sizes)
    a_all = rng.random(n).astype("f4")
    d_all = -rng.random(n)

    schema = pa.schema([pa.field("a", pa.float32()), pa.field("d", pa.float64())])
    batches, off = [], 0
    for s in sizes:
        batches.append(
            pa.record_batch([pa.array(a_all[off : off + s]), pa.array(d_all[off : off + s])], schema=schema)
        )
        off += s

    t = CTable.from_arrow(schema, batches, capacity_hint=n)
    assert len(t) == n
    np.testing.assert_array_equal(t._cols["a"][:], a_all)
    np.testing.assert_array_equal(t._cols["d"][:], d_all)
    # All rows marked valid by the single end-of-import write.
    assert int(blosc2.count_nonzero(t._valid_rows[:n])) == n


# ===========================================================================
# __arrow_c_stream__ (Arrow PyCapsule protocol)
# ===========================================================================


@dataclass
class MixedRow:
    id: int = blosc2.field(blosc2.int64(null_value=np.iinfo(np.int64).min))
    value: float = blosc2.field(blosc2.float64(null_value=float("nan")))
    label: str = blosc2.field(blosc2.string(max_length=8))
    active: bool = blosc2.field(blosc2.bool())
    kind: str = blosc2.field(blosc2.dictionary())


def _mixed_table():
    null_id = np.iinfo(np.int64).min
    data = [
        (0, 0.0, "r0", True, "a"),
        (null_id, 1.5, "r1", False, "b"),
        (2, float("nan"), "r2", True, "a"),
        (3, 3.5, "r3", False, "c"),
    ]
    return CTable(MixedRow, new_data=data)


def test_arrow_c_stream_matches_to_arrow():
    t = _mixed_table()
    via_capsule = pa.table(t)
    assert via_capsule.equals(t.to_arrow())


def test_arrow_c_stream_filtered_view():
    t = _mixed_table()
    view = t[t.id != t["id"].null_value]
    assert pa.table(view).equals(view.to_arrow())


def test_arrow_c_stream_nulls_survive():
    t = _mixed_table()
    at = pa.table(t)
    assert at["id"][1].as_py() is None
    assert at["value"][2].as_py() is None


def test_from_arrow_accepts_capsule_producer():
    t = CTable(Row, new_data=DATA10)
    at = t.to_arrow()
    via_capsule = CTable.from_arrow(at)
    via_batches = CTable.from_arrow(at.schema, at.to_batches())
    assert via_capsule.to_arrow().equals(via_batches.to_arrow())


def test_from_arrow_rejects_capsule_plus_batches():
    at = pa.table({"id": [1, 2]})
    with pytest.raises(TypeError, match="not both"):
        CTable.from_arrow(at, at.to_batches())


def test_from_arrow_requires_batches_for_plain_schema():
    at = pa.table({"id": [1, 2]})
    with pytest.raises(TypeError, match="requires batches"):
        CTable.from_arrow(at.schema)


def test_duckdb_reads_ctable_directly():
    duckdb = pytest.importorskip("duckdb")
    t = CTable(Row, new_data=DATA10)
    result = duckdb.sql("SELECT count(*) AS n FROM t").fetchone()
    assert result[0] == len(DATA10)


def test_polars_reads_ctable_directly():
    pl = pytest.importorskip("polars")
    t = CTable(Row, new_data=DATA10)
    df = pl.DataFrame(t)
    assert df.shape == (len(DATA10), 4)
    assert df.columns == ["id", "score", "active", "label"]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
