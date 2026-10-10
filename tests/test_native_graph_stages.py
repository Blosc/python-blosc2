"""Separately declared staged subset: materialization, lifetime and budgets."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


@pytest.mark.parametrize(
    "expression", ["x - sum(x, axis=0)", "sum(sum(x, axis=0))", "sum(x, axis=0) + max(x, axis=0)"]
)
@pytest.mark.parametrize("dtype", ["int8", "float32", "float64"])
def test_automatic_native_stages(expression, dtype):
    x = np.arange(1, 13, dtype=dtype).reshape(3, 4)
    plan = blosc2.NativeGraph.from_expression(expression, {"x": dtype})
    assert plan.info()["stages"] >= 2
    imported = blosc2.NativeGraph.from_json(plan.to_json())
    assert json.loads(imported.to_json())["format"] == "menudet-staged-graph-1"
    expected = (
        x - np.sum(x, axis=0)
        if expression.startswith("x -")
        else (
            np.sum(np.sum(x, axis=0))
            if expression.startswith("sum(sum")
            else np.sum(x, axis=0) + np.max(x, axis=0)
        )
    )
    for tile in [1, 5]:
        result, report = imported.evaluate({"x": x}, tile_items=tile, return_report=True)
        assert result.dtype == expected.dtype
        assert result.shape == expected.shape
        assert result.tobytes() == expected.tobytes()
        assert report["stages"] == plan.info()["stages"]
        assert report["intermediate_bytes"] > 0


def test_native_required_staged_adapter():
    x = blosc2.asarray(np.arange(12, dtype="float32").reshape(3, 4))
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x - sum(x, axis=0)", {"x": x})
    result = expr.compute(_require_native=True)
    np.testing.assert_array_equal(result[:], x[:] - np.sum(x[:], axis=0))
    assert expr._native_execution_report["stages"] == 2
    with pytest.raises(ValueError, match="partial"):
        expr.compute(slice(1, None), _require_native=True)


def test_shared_stage_value_and_budget():
    graph = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-1.1",
        "requires": ["numeric", "staged"],
        "nodes": [
            {"id": 0, "op": "input", "name": "x", "dtype": "float32"},
            {"id": 1, "op": "function", "name": "sqrt", "args": [0]},
            {"id": 2, "op": "add", "args": [1, 1]},
        ],
        "root": 2,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    plan = blosc2.NativeGraph.from_json(graph)
    with pytest.raises(blosc2.PortableArtifactError, match="budget"):
        plan.specialize({"x": ("float32", (3,))}, intermediate_budget=11)
    schedule = plan.specialize({"x": ("float32", (3,))}, intermediate_budget=12)
    assert schedule.stage_info(0) == {
        "shape": (3,),
        "dtype": np.dtype("float32"),
        "bytes": 12,
        "last_consumer": 1,
    }
    assert schedule.info()["intermediate_bytes"] == 12
    result, report = schedule.execute({"x": np.array([1, 4, 9], dtype="float32")})
    np.testing.assert_array_equal(result, [2, 4, 6])
    assert report["evaluated_tiles"] == 2
    del plan
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(
            executor.map(
                lambda value: schedule.execute({"x": np.full(3, value, dtype="float32")}), range(1, 9)
            )
        )


@pytest.mark.parametrize(
    "expression", ["where(x > 0, sum(x), x)", "x and sum(x)", "sum(x - sum(x), where=mask)"]
)
def test_conditional_stage_domains_reject(expression):
    inputs = {"x": "float32"}
    if "mask" in expression:
        inputs["mask"] = "bool"
    with pytest.raises(blosc2.PortableArtifactError, match=r"conditional|masked"):
        blosc2.NativeGraph.from_expression(expression, inputs)


def trusted_document():
    artifact = json.loads(blosc2.NativeGraph.from_expression("x * 2", {"x": "float32"}).map_json())
    graph = json.loads(blosc2.NativeGraph.from_expression("sum(y)", {"y": "float32"}).to_json())
    return {
        "format": "menudet-staged-graph-1",
        "semantics": "menudet-numpy-1.1",
        "requires": ["numeric", "staged"],
        "inputs": [{"name": "x", "dtype": "float32"}],
        "root": 1,
        "stages": [
            {
                "id": 0,
                "kind": "portable",
                "inputs": {"x": {"input": "x"}},
                "artifact": artifact,
                "contract": {
                    "cardinality": "elementwise",
                    "context": "none",
                    "effects": "ordered-lazy",
                    "mask": "none",
                },
            },
            {"id": 1, "kind": "graph", "inputs": {"y": {"stage": 0}}, "graph": graph},
        ],
    }


def test_trusted_portable_stage_and_roundtrip():
    plan = blosc2.NativeGraph(trusted_document())
    imported = blosc2.NativeGraph(plan.to_json())
    x = np.arange(5, dtype="float32")
    result = imported.evaluate({"x": x})
    assert result == np.sum(x * 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("mask", "external"), ("effects", "pure"), ("context", "nd"), ("cardinality", "block-scalar")],
)
def test_unqualified_trusted_contract_rejects(field, value):
    document = trusted_document()
    document["stages"][0]["contract"][field] = value
    with pytest.raises(blosc2.PortableArtifactError, match="contract"):
        blosc2.NativeGraph(document)


def test_late_stage_status_recovery():
    plan = blosc2.NativeGraph.from_expression("sqrt(x) + sum(y)", {"x": "float64", "y": "float64"})
    schedule = plan.specialize({"x": ("float64", (3,)), "y": ("float64", (3,))})
    with pytest.raises(blosc2.PortableArtifactError) as failure:
        schedule.execute({"x": np.array([-1.0, 1.0, 4.0]), "y": np.ones(3)}, 1)
    assert failure.value.fp_status["flags"] & 1
    result, report = schedule.execute({"x": np.array([0.0, 1.0, 4.0]), "y": np.ones(3)}, 1)
    np.testing.assert_array_equal(result, [3, 4, 5])
    assert not report["fp_flags"]


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "step", "swapped"])
def test_staged_reducers_and_layouts(name, layout):
    x = np.arange(12, dtype="float32").reshape(3, 4)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1]
    elif layout == "step":
        x = x[:, ::2]
    elif layout == "swapped":
        x = x.astype(">f4")
    expression = f"{name}(x, axis=-1, keepdims=True) + x"
    plan = blosc2.NativeGraph.from_expression(expression, {"x": "float32"})
    expected = getattr(np, name)(x.astype("float32"), axis=-1, keepdims=True) + x
    for tile in [1, 7]:
        result = plan.evaluate({"x": x}, tile_items=tile)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("shape", [(0, 4), (2, 0), ()])
def test_empty_and_rank_zero_stages(shape):
    x = np.zeros(shape, dtype="float64")
    plan = blosc2.NativeGraph.from_expression("sum(x) + x", {"x": "float64"})
    result = plan.evaluate({"x": x})
    np.testing.assert_array_equal(result, np.sum(x) + x)
    assert result.shape == shape


@pytest.mark.parametrize(
    "mutation", ["cycle", "missing", "signature", "unreachable", "root", "capability", "nested"]
)
def test_staged_dependency_validation(mutation):
    document = trusted_document()
    if mutation == "cycle":
        document["stages"][0]["inputs"]["x"] = {"stage": 1}
    elif mutation == "missing":
        document["stages"][1]["inputs"] = {}
    elif mutation == "signature":
        document["inputs"][0]["dtype"] = "float64"
    elif mutation == "unreachable":
        document["stages"][1]["inputs"]["y"] = {"input": "x"}
    elif mutation == "root":
        document["root"] = 0
    elif mutation == "capability":
        document["requires"] = ["numeric"]
    else:
        document["stages"][1]["graph"] = trusted_document()
    with pytest.raises(blosc2.PortableArtifactError):
        blosc2.NativeGraph(document)


def test_staged_map_export_and_required_jit():
    expression = "x - sum(x)"
    plan = blosc2.NativeGraph.from_expression(expression, {"x": "float32"}, jit=True)
    if plan.has_jit:
        plan = blosc2.NativeGraph.from_expression(expression, {"x": "float32"}, require_jit=True)
        assert plan.has_jit
    else:
        with pytest.raises(blosc2.PortableArtifactError, match="required map JIT"):
            blosc2.NativeGraph.from_expression(expression, {"x": "float32"}, require_jit=True)
    with pytest.raises(ValueError, match="elementwise"):
        plan.map_json()
    result, report = plan.evaluate({"x": np.arange(5, dtype="float32")}, return_report=True)
    np.testing.assert_array_equal(result, [-10, -9, -8, -7, -6])
    assert report["jit_stages"] + report["interpreter_stages"] == 2
    if plan.has_jit:
        # The direct-input sum bypasses its identity map; only subtraction
        # executes JIT code, even if both prepared maps are JIT-capable.
        assert report["jit_stages"] == 1
    else:
        assert report["interpreter_stages"] >= 1
