from __future__ import annotations

from dataclasses import dataclass

import pytest

import blosc2
from blosc2.schema import ListSpec
from blosc2.schema_compiler import compile_schema, schema_from_dict, schema_to_dict


@dataclass
class Product:
    code: str = blosc2.field(blosc2.string(max_length=8))
    tags: list[str] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.string(max_length=16), nullable=True, batch_rows=32)
    )


def test_list_builder_and_compile_schema():
    spec = blosc2.list(blosc2.string(max_length=10), nullable=True, storage="batch", serializer="msgpack")
    assert isinstance(spec, ListSpec)
    assert spec.nullable is True
    assert spec.display_label() == "list[string]"

    schema = compile_schema(Product)
    assert isinstance(schema.columns_by_name["tags"].spec, ListSpec)
    assert schema.columns_by_name["tags"].dtype is None


def test_list_schema_roundtrip():
    schema = compile_schema(Product)
    d = schema_to_dict(schema)
    tags = next(c for c in d["columns"] if c["name"] == "tags")
    assert tags["kind"] == "list"
    assert tags["item"]["kind"] == "string"
    restored = schema_from_dict(d)
    assert isinstance(restored.columns_by_name["tags"].spec, ListSpec)
    assert restored.columns_by_name["tags"].spec.batch_rows == 32


def test_list_annotation_mismatch_rejected():
    @dataclass
    class Bad:
        tags: str = blosc2.field(blosc2.list(blosc2.string()))

    with pytest.raises(TypeError, match="list spec"):
        compile_schema(Bad)
