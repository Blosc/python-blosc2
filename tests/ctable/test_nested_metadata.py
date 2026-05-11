import pyarrow as pa

import blosc2
from blosc2.schema_compiler import schema_from_dict, schema_to_dict


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
    md = {b"blosc2_empty_root_physical": b"root"}
    schema = pa.schema([pa.field("root", pa.float64())]).with_metadata(md)
    batch = pa.record_batch([pa.array([1.0, 2.0])], schema=schema)

    t = blosc2.CTable.from_arrow(schema, [batch])
    out = t.to_arrow()
    assert out.schema.names == [""]
