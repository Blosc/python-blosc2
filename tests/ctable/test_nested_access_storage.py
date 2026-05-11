from dataclasses import dataclass

import pytest

import blosc2

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - optional dependency
    pa = None
    pq = None

pytestmark = pytest.mark.skipif(pa is None, reason="pyarrow is required for nested Arrow/Parquet tests")


@dataclass
class AccessRow:
    trip_begin_lon: float
    payment_fare: float


@dataclass
class PersistRow:
    a: int


def test_dotted_column_attribute_namespace_and_where_string():
    t = blosc2.CTable(AccessRow)
    t.append((1.0, 10.0))
    t.append((2.0, 30.0))
    t.append((3.0, 40.0))

    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("payment_fare", "payment.fare")

    assert t["trip.begin.lon"].sum() == 6.0
    assert t.trip.begin.lon.max() == 3.0

    view1 = t.where("payment.fare > 20")
    assert view1.nrows == 2

    view2 = t.where(t.payment.fare > 20)
    assert view2.nrows == 2


def test_dotted_column_persists_under_hierarchical_cols(tmp_path):
    t = blosc2.CTable(PersistRow)
    t.append((1,))
    t.rename_column("a", "trip.begin.lon")

    path = tmp_path / "nested.b2d"
    t.save(str(path), overwrite=True)

    leaf = path / "_cols" / "trip" / "begin" / "lon.b2nd"
    assert leaf.exists()

    opened = blosc2.CTable.open(str(path))
    assert opened["trip.begin.lon"][0] == 1


def test_select_struct_prefix_expands_descendants():
    t = blosc2.CTable(AccessRow)
    t.append((1.0, 10.0))
    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("payment_fare", "payment.fare")

    s = t.select(["trip"])
    assert s.col_names == ["trip.begin.lon"]


def test_from_arrow_flattens_struct_columns_to_dotted_leaves():
    trip_type = pa.struct([("begin", pa.struct([("lon", pa.float64()), ("lat", pa.float64())]))])
    schema = pa.schema([pa.field("trip", trip_type)])
    batch = pa.record_batch(
        [
            pa.array(
                [
                    {"begin": {"lon": 1.1, "lat": 2.2}},
                    {"begin": {"lon": 3.3, "lat": 4.4}},
                ],
                type=trip_type,
            )
        ],
        schema=schema,
    )

    t = blosc2.CTable.from_arrow(schema, [batch])
    assert "trip.begin.lon" in t.col_names
    assert "trip.begin.lat" in t.col_names
    assert t["trip.begin.lon"][1] == 3.3

    row0 = t[0]
    assert isinstance(row0.trip, dict)
    assert row0.trip["begin"]["lon"] == 1.1
    assert row0.trip["begin"]["lat"] == 2.2


def test_nested_struct_parquet_roundtrip(tmp_path):
    trip_type = pa.struct([("begin", pa.struct([("lon", pa.float64()), ("lat", pa.float64())]))])
    schema = pa.schema([pa.field("trip", trip_type)])
    table = pa.table(
        {
            "trip": pa.array(
                [
                    {"begin": {"lon": 1.1, "lat": 2.2}},
                    {"begin": {"lon": 3.3, "lat": 4.4}},
                    {"begin": {"lon": 5.5, "lat": 6.6}},
                ],
                type=trip_type,
            )
        },
        schema=schema,
    )

    src = tmp_path / "src.parquet"
    pq.write_table(table, src)

    t = blosc2.CTable.from_parquet(src)
    assert t.col_names == ["trip.begin.lon", "trip.begin.lat"]
    assert t[2].trip["begin"]["lon"] == 5.5

    dst = tmp_path / "dst.parquet"
    t.to_parquet(dst)
    out = pq.read_table(dst)
    assert out.num_rows == 3
    assert out.schema.names == ["trip"]
    assert out.column("trip").to_pylist()[0]["begin"]["lon"] == 1.1
