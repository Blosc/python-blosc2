import pytest

import blosc2
from blosc2.schema_compiler import schema_from_dict, schema_to_dict

try:
    import pyarrow as pa
except ImportError:  # pragma: no cover - optional dependency
    pa = None

pytestmark = pytest.mark.skipif(pa is None, reason="pyarrow is required for nested Arrow/Parquet tests")


def _table_with_empty_root_alias():
    md = {b"blosc2_empty_root_physical": b"root"}
    schema = pa.schema([pa.field("root", pa.float64())]).with_metadata(md)
    batch = pa.record_batch([pa.array([1.0, 2.0, 3.0])], schema=schema)
    return blosc2.CTable.from_arrow(schema, [batch])


def test_schema_version_2_with_nested_metadata_roundtrip():
    schema = pa.schema([pa.field("x.y", pa.float64())])
    batch = pa.record_batch([pa.array([1.0, 2.0])], schema=schema)
    t = blosc2.CTable.from_arrow(schema, [batch])

    d = schema_to_dict(t._schema)
    assert d["version"] == 2
    assert "nested" in d["metadata"]

    restored = schema_from_dict(d)
    assert restored.metadata["nested"]["physical_to_storage"]["x.y"] == "_cols/x/y"


def test_empty_root_metadata_exports_back_to_empty_arrow_name():
    t = _table_with_empty_root_alias()
    out = t.to_arrow()
    assert out.schema.names == [""]


def test_empty_root_logical_alias_getitem_select_and_index():
    t = _table_with_empty_root_alias()
    assert t[""][0] == 1.0
    s = t.select([""])
    assert s.col_names == ["root"]

    ix = t.create_index(col_name="")
    assert ix is not None

    # index management should accept logical alias too
    t.rebuild_index(col_name="")
    t.drop_index(col_name="")


def test_sort_by_nested_prefix_requires_leaf_column():
    schema = pa.schema([pa.field("trip.begin.lon", pa.float64()), pa.field("trip.begin.lat", pa.float64())])
    batch = pa.record_batch([pa.array([2.0, 1.0]), pa.array([20.0, 10.0])], schema=schema)
    t = blosc2.CTable.from_arrow(schema, [batch])

    with pytest.raises(ValueError):
        t.sort_by("trip")

    s = t.sort_by("trip.begin.lon")
    assert s["trip.begin.lon"][0] == 1.0


def test_nested_ops_compat_matrix_smoke():
    n = 20_000
    lon = pa.array([float(i % 1000) for i in range(n)], type=pa.float64())
    lat = pa.array([float((i * 2) % 1000) for i in range(n)], type=pa.float64())
    fare = pa.array([float(i % 50) for i in range(n)], type=pa.float64())
    schema = pa.schema(
        [
            pa.field("trip.begin.lon", pa.float64()),
            pa.field("trip.begin.lat", pa.float64()),
            pa.field("payment.fare", pa.float64()),
        ]
    )
    batch = pa.record_batch([lon, lat, fare], schema=schema)

    t = blosc2.CTable.from_arrow(schema, [batch])

    view = t.where("payment.fare > 25")
    assert 0 < view.nrows < n

    t.create_index(col_name="payment.fare")
    t.rebuild_index(col_name="payment.fare")

    sorted_t = t.sort_by("trip.begin.lon")
    assert sorted_t["trip.begin.lon"][0] <= sorted_t["trip.begin.lon"][1]

    proj = t.select(["trip"])
    assert proj.col_names == ["trip.begin.lon", "trip.begin.lat"]
