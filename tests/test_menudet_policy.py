"""Metadata policy/preflight checks independent of the installed native version."""

from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.b2objects import decode_b2object_payload
from blosc2.dsl_kernel import SourceKernel


@pytest.mark.parametrize("policy", ["safe", "full"])
@pytest.mark.parametrize("expression", ["x + 1", "sqrt(x)"])
def test_nested_legacy_effective_policy(tmp_path, policy, expression):
    fixture = Path(__file__).parent / "data" / "legacy_lazyudf_v1" / "expr.b2nd"
    carrier = blosc2.empty((5,), dtype="float64", meta={"b2o": {"kind": "lazyexpr", "version": 1}})
    carrier.schunk.vlmeta["b2o"] = {
        "kind": "lazyexpr",
        "version": 1,
        "expression": expression,
        "operands": {"x": {"kind": "urlpath", "version": 1, "urlpath": str(fixture)}},
    }
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        blosc2.from_cframe(carrier.to_cframe(), deserialize=policy)


@pytest.mark.parametrize("policy", ["safe", "full"])
def test_structured_legacy_rejects_before_python(monkeypatch, policy):
    from msgpack import ExtType, packb

    from blosc2.msgpack_utils import msgpack_unpackb

    record = {
        "kind": "lazyudf",
        "version": 1,
        "function_kind": "dsl",
        "dsl_version": 1,
        "udf_source": "def k():\n    return 1\n",
        "name": "k",
        "dtype": "int64",
        "shape": [2],
        "operands": {},
        "kwargs": {},
    }
    legacy = ExtType(43, packb(record))
    nested = ExtType(46, packb({"values": [legacy], "shape": [1]}))
    monkeypatch.setattr(SourceKernel, "__init__", lambda *a, **k: pytest.fail("Python reconstruction"))
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        decode_b2object_payload({"kind": "lazyudf", "version": 1}, deserialize=policy)
    for payload in (packb(legacy), packb(nested)):
        with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
            msgpack_unpackb(payload, deserialize=policy)


@pytest.mark.parametrize("route", ["embed", "dict", "tree"])
def test_rejected_kernel_does_not_replace_store_value(tmp_path, route):
    cls = {"embed": blosc2.EmbedStore, "dict": blosc2.DictStore, "tree": blosc2.TreeStore}[route]
    udf = blosc2.lazyudf(lambda inputs, output, offset: None, (np.arange(4),), dtype="int64")
    with cls(str(tmp_path / "store.b2z"), mode="w") as store:
        store["/value"] = np.arange(4)
        with pytest.raises(TypeError, match="MenudetKernel"):
            store["/value"] = udf
        np.testing.assert_array_equal(store["/value"][:], np.arange(4))
        operand = blosc2.asarray(np.arange(4), urlpath=tmp_path / "input.b2nd", mode="w")
        for body in ("print(x)\n    return x", "return callback(x)"):
            kernel = SourceKernel.from_source(f"def invalid(x):\n    {body}\n")
            invalid = blosc2.lazyudf(kernel, (operand,), dtype="int64")
            with pytest.raises(blosc2.MenudetError):
                store["/value"] = invalid
            np.testing.assert_array_equal(store["/value"][:], np.arange(4))
        valid = SourceKernel.from_source("def valid(x):\n    return x + 2\n")
        store["/recipe"] = blosc2.lazyudf(valid, (operand,), dtype="int64")
        np.testing.assert_array_equal(store["/recipe"][:], np.arange(4) + 2)


def test_ctable_legacy_metadata_policy_before_reconstruction(monkeypatch):
    from types import SimpleNamespace

    table = object.__new__(blosc2.CTable)
    table._storage = SimpleNamespace(_deserialize_mode="safe")
    schema = {
        "computed_columns": [
            {"name": "bad", "col_deps": [], "dtype": "int64", "kind": "dsl", "dsl_source": "invalid"}
        ]
    }
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError):
        table._load_computed_cols_from_schema(schema)
    table._computed_cols = {"x": {"kind": "dsl"}}
    table._materialized_cols = {}
    with pytest.raises(TypeError, match="row-domain"):
        table._preflight_menudet_persistence()


@pytest.mark.parametrize("generated", [False, True])
@pytest.mark.parametrize("policy", ["safe", "full"])
def test_legacy_table_recipes_are_unsupported(tmp_path, generated, policy):
    from dataclasses import dataclass

    @dataclass
    class Row:
        x: float = 0.0

    path = str(tmp_path / "legacy.b2d")
    table = blosc2.CTable(
        Row, urlpath=path, mode="w", new_data={"x": [2.0, 4.0]}, create_summary_index=False
    )
    if generated:
        table.add_column("old", blosc2.float64(), values=np.array([4.0, 6.0]))
    schema = table._schema_dict_with_computed()
    recipe = {
        "name": "old",
        "dsl_source": "def old(x):\n    return x + sum(x)\n",
        "col_deps": ["x"],
        "dtype": "float64",
    }
    if generated:
        recipe.update(transformer_kind="dsl", stale=False, computed_column=None, expression=None)
        schema["materialized_columns"] = [recipe]
    else:
        recipe["kind"] = "dsl"
        schema["computed_columns"] = [recipe]
    table._storage.save_schema(schema)
    table.close()
    with pytest.raises(blosc2.UnsupportedLegacyRecipeError, match="previous release"):
        blosc2.CTable.open(path, mode="a", deserialize=policy)
