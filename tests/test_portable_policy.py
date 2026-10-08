"""Metadata policy/preflight checks independent of the installed native version."""

from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.b2objects import decode_b2object_payload


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
    if policy == "safe":
        with pytest.raises(blosc2.UnsafeDeserializationError):
            blosc2.from_cframe(carrier.to_cframe(), deserialize=policy)
    else:
        opened = blosc2.from_cframe(carrier.to_cframe(), deserialize=policy)
        nested = next(iter(opened.operands.values()))
        assert isinstance(nested, blosc2.LazyUDF)
        np.testing.assert_allclose(nested.compute()[:], (np.arange(5) * 3) ** 2)


def test_structured_legacy_rejects_before_python(monkeypatch):
    from msgpack import ExtType, packb

    import blosc2.b2objects as b2objects
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
    for payload in (packb(legacy), packb(nested)):
        trusted = msgpack_unpackb(payload, deserialize="full")
        assert isinstance(trusted if isinstance(trusted, blosc2.LazyUDF) else trusted[0], blosc2.LazyUDF)
        legacy_recipe = trusted if isinstance(trusted, blosc2.LazyUDF) else trusted[0]
        with pytest.raises(TypeError, match="legacy"):
            legacy_recipe.to_cframe()

    monkeypatch.setattr(b2objects, "kernel_from_source", lambda *a, **k: pytest.fail("Policy bypass"))
    with pytest.raises(blosc2.UnsafeDeserializationError):
        decode_b2object_payload({"kind": "lazyudf", "version": 1}, deserialize="safe")
    for payload in (packb(legacy), packb(nested)):
        with pytest.raises(blosc2.UnsafeDeserializationError):
            msgpack_unpackb(payload)
        with pytest.raises(blosc2.UnsafeDeserializationError):
            msgpack_unpackb(payload, deserialize="safe")


@pytest.mark.parametrize("route", ["embed", "dict", "tree"])
def test_rejected_kernel_does_not_replace_store_value(tmp_path, route):
    cls = {"embed": blosc2.EmbedStore, "dict": blosc2.DictStore, "tree": blosc2.TreeStore}[route]
    udf = blosc2.lazyudf(lambda inputs, output, offset: None, (np.arange(4),), dtype="int64")
    with cls(str(tmp_path / "store.b2z"), mode="w") as store:
        store["/value"] = np.arange(4)
        with pytest.raises(TypeError, match="PortableKernel"):
            store["/value"] = udf
        np.testing.assert_array_equal(store["/value"][:], np.arange(4))
        operand = blosc2.asarray(np.arange(4), urlpath=tmp_path / "input.b2nd", mode="w")
        for body in ("print(x)\n    return x", "return callback(x)"):
            kernel = blosc2.DSLKernel.from_source(f"def invalid(x):\n    {body}\n")
            invalid = blosc2.lazyudf(kernel, (operand,), dtype="int64")
            with pytest.raises(blosc2.PortableArtifactError):
                store["/value"] = invalid
            np.testing.assert_array_equal(store["/value"][:], np.arange(4))
        valid = blosc2.DSLKernel.from_source("def valid(x):\n    return x + 2\n")
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
    with pytest.raises(
        blosc2.UnsafeDeserializationError, match=r"block/batch-dependent.*deserialize='full'"
    ):
        table._load_computed_cols_from_schema(schema)
    table._computed_cols = {"x": {"kind": "dsl"}}
    table._materialized_cols = {}
    with pytest.raises(TypeError, match="row-domain"):
        table._preflight_portable_persistence()


@pytest.mark.parametrize("generated", [False, True])
def test_legacy_table_recipes_retain_full_opt_in(tmp_path, generated):
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
    with pytest.raises(
        blosc2.UnsafeDeserializationError, match=r"block/batch-dependent.*deserialize='full'"
    ):
        blosc2.CTable.open(path)
    trusted = blosc2.CTable.open(path, mode="a", deserialize="full")
    if generated:
        expected_batch = trusted._evaluate_dsl_materialized_batch(
            trusted._materialized_cols["old"], {"x": np.array([8.0])}
        )
        trusted.append({"x": 8.0})
        # Appended batches retain their old groups rather than adopting the new API contract.
        np.testing.assert_array_equal(trusted["old"][:], [4.0, 6.0, expected_batch[0]])
    else:
        # The original two-row execution block, not independent one-lane rows.
        np.testing.assert_array_equal(trusted["old"][:], [8.0, 10.0])
    trusted.close()
