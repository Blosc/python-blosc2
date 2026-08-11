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


def test_schema_v2_nested_metadata_roundtrip():
    """Nested metadata alone asks for version 2, not 3.

    The version is a feature max, so the field is declared non-nullable to
    isolate it: an Arrow field is nullable by default, and a nullable column now
    resolves to mask storage, which is what raises the version to 3.
    """
    schema = pa.schema([pa.field("x.y", pa.float64(), nullable=False)])
    batch = pa.record_batch([pa.array([1.0, 2.0])], schema=schema)
    t = blosc2.CTable.from_arrow(schema, [batch])

    d = schema_to_dict(t._schema)
    assert d["version"] == 2
    assert "nested" in d["metadata"]

    restored = schema_from_dict(d)
    assert restored.metadata["nested"]["logical_to_physical"]["x.y"] == "x.y"


def test_a_mask_column_raises_the_nested_schema_to_v3():
    """The version is the max of the features present, not a per-feature tag."""
    schema = pa.schema([pa.field("x.y", pa.float64())])  # nullable by default
    batch = pa.record_batch([pa.array([1.0, 2.0])], schema=schema)
    t = blosc2.CTable.from_arrow(schema, [batch])

    d = schema_to_dict(t._schema)
    assert d["version"] == 3
    assert "nested" in d["metadata"]
    assert t["x.y"].null_storage == "mask"


def test_empty_root_exports_empty_arrow_name():
    t = _table_with_empty_root_alias()
    out = t.to_arrow()
    assert out.schema.names == [""]


def test_empty_root_alias_getitem_and_select():
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


@pytest.mark.heavy
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


def _struct_table(urlpath=None):
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3], pa.int64()),
            "trip": pa.array([{"lon": 1.0, "lat": 2.0}] * 3),
        }
    )
    kwargs = {"urlpath": str(urlpath), "mode": "w"} if urlpath is not None else {}
    return blosc2.CTable.from_arrow(tbl, **kwargs)


def test_a_struct_column_survives_save_and_reopen(tmp_path):
    """The logical parent of a nested group has to be serialized too.

    ``columns_by_name`` carries a ``StructSpec`` for ``trip`` beside its
    physical ``trip.lon``/``trip.lat``, and it is *not* one of ``columns`` --
    which is exactly why ``_export_arrow_names`` looks it up there.  Nothing
    wrote it out, so a reopened table exported the leaves flat while the table
    it was saved from exported the struct.
    """
    path = tmp_path / "nested.b2t"
    t = _struct_table(path)
    assert t.to_arrow().schema.names == ["a", "trip"]
    del t

    reopened = blosc2.CTable.open(str(path))
    assert reopened.to_arrow().schema.names == ["a", "trip"]
    assert reopened.to_arrow().column("trip").to_pylist()[0] == {"lon": 1.0, "lat": 2.0}


def test_the_struct_parent_survives_a_second_round_trip(tmp_path):
    """Restored on load and re-derived on save, so it does not decay."""
    first = tmp_path / "one.b2t"
    second = tmp_path / "two.b2t"
    t = _struct_table(first)
    del t
    blosc2.CTable.open(str(first)).copy(urlpath=str(second))
    assert blosc2.CTable.open(str(second)).to_arrow().schema.names == ["a", "trip"]


def test_the_parent_specs_do_not_leak_into_user_metadata(tmp_path):
    """Carried inside the metadata blob, but consumed on the way back in."""
    path = tmp_path / "meta.b2t"
    t = _struct_table(path)
    del t
    reopened = blosc2.CTable.open(str(path))
    assert "struct_parents" in schema_to_dict(reopened._schema).get("metadata", {})
    assert "struct_parents" not in reopened._schema.metadata


def test_a_file_saved_before_struct_parents_still_loads():
    """Older tables simply keep their flat export rather than failing to open."""
    data = schema_to_dict(_struct_table()._schema)
    data["metadata"].pop("struct_parents")
    schema = schema_from_dict(data)
    assert sorted(schema.columns_by_name) == ["a", "trip.lat", "trip.lon"]
