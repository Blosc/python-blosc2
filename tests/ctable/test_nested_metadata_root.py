import pyarrow as pa

import blosc2
from blosc2.schema_compiler import schema_from_dict, schema_to_dict


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
