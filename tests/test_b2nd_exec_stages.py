"""Typed stage ownership, bindings, diagnostics and adapter differential gates."""

import gc
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph
from blosc2.typed_exec import lower_typed_graph, prepare_typed_plan

pytestmark = pytest.mark.skipif(
    not getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)(),
    reason="requires the opt-in C-Blosc2 typed executor",
)


@pytest.mark.parametrize(
    "expression",
    [
        "x - sum(x, axis=0)",
        "sum(sum(x, axis=0))",
        "sum(x, axis=0) + max(x, axis=0)",
        "max(sum(x, axis=0) + min(x, axis=0))",
        "float32(sum(float64(x), axis=0)) + x",
        "sum(x, axis=0, initial=-2, keepdims=True) + x",
    ],
)
@pytest.mark.parametrize("dtype", ["int8", "float32", "float64"])
def test_general_stages_adapter_parity_and_budget(expression, dtype, monkeypatch):
    values = np.arange(1, 13, dtype=dtype).reshape(3, 4)[:, ::-1]
    inputs = {"x": values}
    baseline = NativeGraph.from_expression(expression, {"x": dtype}, jit=False)
    expected, reference = baseline.evaluate(inputs, tile_items=3, return_report=True)
    native = NativeGraph.from_expression(expression, {"x": dtype}, jit=False, _executor="c-blosc2")
    assert native.stages == baseline.stages

    def forbidden(*args, **kwargs):
        pytest.fail("typed stage prepare/inspect/specialize/execute reached the adapter")

    monkeypatch.setattr("blosc2.graph_plan.NativeGraphHandle", forbidden)
    schedule = native.specialize({"x": (values.dtype, values.shape)}, tile_items=3)
    budget = schedule.info()["intermediate_bytes"]
    assert budget > 0
    if budget > 1:
        with pytest.raises(blosc2.MenudetError, match="budget"):
            native.specialize({"x": (values.dtype, values.shape)}, intermediate_budget=budget - 1)
    del native
    gc.collect()
    actual, report = schedule.execute(inputs)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert report["fp_flags"] == reference["fp_flags"]
    assert report["stages"] == baseline.stages
    assert report["intermediate_bytes"] == budget
    assert report["live_intermediate_bytes"] == 0
    assert report["input_materialization_bytes"] == 0
    assert len(report["stage_reports"]) == report["entered_stages"] == baseline.stages


def test_general_stages_concurrent_schedule_and_error_diagnostics():
    graph = NativeGraph.from_expression(
        "sum(y) + sqrt(x)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    schedule = graph.specialize({"x": ("float64", (4,)), "y": ("float64", (4,))}, tile_items=2)
    with pytest.raises(blosc2.MenudetError) as caught:
        schedule.execute({"x": np.array([-1.0, 1, 4, 9]), "y": np.ones(4)}, 1)
    assert caught.value.stage == 1
    assert caught.value.node >= 0
    assert caught.value.fp_status["flags"] & 1

    def run(value):
        inputs = {"x": np.full(4, value * value, dtype="float64"), "y": np.ones(4)}
        actual, report = schedule.execute(inputs)
        np.testing.assert_array_equal(actual, np.full(4, 4 + value))
        assert report["live_intermediate_bytes"] == 0
        assert not report["fp_flags"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(1, 9)))


def test_general_source_stage_preflight_and_no_replay():
    class Source:
        shape = (3, 4)
        dtype = np.dtype("float64")

        def __init__(self):
            self.calls = 0
            self.error = None

        def read(self, indices):
            self.calls += 1
            if self.error is not None and self.calls == 2:
                raise self.error
            return np.asarray(indices, dtype="float64") + 1

    source = Source()
    graph = NativeGraph.from_expression(
        "sum(x, axis=0) + max(x, axis=0)", {"x": "float64"}, jit=False, _executor="c-blosc2"
    )
    schedule = graph.specialize({"x": (source.dtype, source.shape)}, tile_items=2)
    with pytest.raises(blosc2.MenudetError, match="budget"):
        schedule.execute_sources({"x": source}, max_gather_bytes=0)
    assert source.calls == 0
    error = NotImplementedError("reader failure is not admission")
    source.error = error
    with pytest.raises(NotImplementedError) as caught:
        schedule.execute_sources({"x": source})
    assert caught.value is error
    assert source.calls == 2
    source.calls = 0
    source.error = None
    result, report = schedule.execute_sources({"x": source})
    raw = np.arange(1, 13, dtype="float64").reshape(3, 4)
    np.testing.assert_array_equal(result, raw.sum(axis=0) + raw.max(axis=0))
    assert report["stages"] == 3


def test_compressed_stage_capability_rejects_without_materialization(monkeypatch):
    graph = NativeGraph.from_expression("x - sum(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    array = blosc2.asarray(np.arange(8, dtype="float64"))

    def forbidden(*args, **kwargs):
        pytest.fail("staged capability rejection read a compressed operand")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    with pytest.raises(NotImplementedError, match="Compressed staged"):
        graph.evaluate_sources({"x": array})


@pytest.mark.parametrize("shape", [(0, 4), (2, 0), ()])
def test_empty_and_rank_zero_stages(shape):
    x = np.full(shape, 4.5, dtype="float64")
    baseline = NativeGraph.from_expression("sum(x) + x", {"x": "float64"}, jit=False)
    typed = NativeGraph.from_expression("sum(x) + x", {"x": "float64"}, jit=False, _executor="c-blosc2")
    expected = baseline.evaluate({"x": x})
    actual = typed.evaluate({"x": x})
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape == shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("name", ["sum", "prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("layout", ["c", "f", "reverse", "step", "swapped"])
def test_staged_reducers_and_layouts_parity(name, layout):
    x = np.arange(1, 13, dtype="float32").reshape(3, 4)
    if layout == "f":
        x = np.asfortranarray(x)
    elif layout == "reverse":
        x = x[::-1, ::-1]
    elif layout == "step":
        x = x[:, ::2]
    elif layout == "swapped":
        x = x.astype(">f4")
    expression = f"{name}(x, axis=-1, keepdims=True) + x"
    baseline = NativeGraph.from_expression(expression, {"x": "float32"}, jit=False)
    typed = NativeGraph.from_expression(expression, {"x": "float32"}, jit=False, _executor="c-blosc2")
    for tile in [1, 7]:
        expected = baseline.evaluate({"x": x}, tile_items=tile)
        actual = typed.evaluate({"x": x}, tile_items=tile)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    "expression",
    ["where(x > 0, sum(x), x)", "where(x > 0, x, y) + sum(x)", "x and sum(x)", "sum(x - sum(x), where=m)"],
)
def test_automatic_conditional_stage_boundary_preserves_restriction(expression):
    signature = {"x": "float64"}
    if "y" in expression:
        signature["y"] = "float64"
    if "where=m" in expression:
        signature["m"] = "bool"
    with pytest.raises((blosc2.MenudetError, NotImplementedError)):
        NativeGraph.from_expression(expression, signature, jit=False, _executor="c-blosc2")


def test_imported_scalar_shared_and_staged_output_reject_with_diagnostics():
    shared_scalar = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-0.1",
        "requires": ["numeric", "staged"],
        "nodes": [
            {"id": 0, "op": "input", "name": "x", "dtype": "float64"},
            {
                "id": 1,
                "op": "constant",
                "dtype": "int64",
                "encoding": "decimal",
                "value": "1",
                "category": "weak",
            },
            {
                "id": 2,
                "op": "constant",
                "dtype": "int64",
                "encoding": "decimal",
                "value": "2",
                "category": "weak",
            },
            {"id": 3, "op": "add", "args": [1, 2]},
            {"id": 4, "op": "add", "args": [3, 3]},
            {"id": 5, "op": "add", "args": [0, 4]},
        ],
        "root": 5,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    with pytest.raises(NotImplementedError) as caught:
        NativeGraph(shared_scalar, jit=False, _executor="c-blosc2")
    assert caught.value.node == 3

    staged_cast = json.loads(
        NativeGraph.from_expression("x - sum(x)", {"x": "float32"}, jit=False).to_json()
    )
    # Verify automatic staged menudet-graph-1 with non-auto output rejects (like adapter).
    staged_doc = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-0.1",
        "requires": ["numeric", "staged"],
        "nodes": [
            {"id": 0, "op": "input", "name": "x", "dtype": "float32"},
            {
                "id": 1,
                "op": "sum",
                "args": [0],
                "axes": None,
                "keepdims": False,
                "dtype": "auto",
                "initial": None,
                "where": None,
            },
            {"id": 2, "op": "sub", "args": [0, 1]},
        ],
        "root": 2,
        "output": {"dtype": "float64", "casting": "safe"},
    }
    assert json.loads(json.dumps(staged_cast))["format"] == "menudet-staged-graph-1"
    with pytest.raises(blosc2.MenudetError):
        NativeGraph(staged_doc, jit=False, _executor="c-blosc2")

    # Stage diagnostics propagate on prepare and specialize budget failures.
    from test_native_graph_stages import trusted_document

    bad_doc = trusted_document()
    bad_doc["stages"][1]["graph"]["output"] = {"dtype": "int32", "casting": "safe"}
    bad_doc["stages"][1]["graph"]["requires"] = ["numeric"]
    with pytest.raises(blosc2.MenudetError) as bad_import:
        NativeGraph(bad_doc, jit=False, _executor="c-blosc2")
    assert bad_import.value.stage == 1

    multi = NativeGraph.from_expression(
        "sum(x, axis=0) + max(x, axis=0)", {"x": "float64"}, jit=False, _executor="c-blosc2"
    )
    with pytest.raises(blosc2.MenudetError) as budget_err:
        multi.specialize({"x": ("float64", (3, 4))}, intermediate_budget=63)
    assert budget_err.value.stage == 1
    assert budget_err.value.node >= 0


@pytest.mark.parametrize(
    "expression",
    ["sum(sqrt(x), where=m)", "prod(sqrt(x), where=m)", "max(sqrt(x), where=m, initial=5)"],
)
def test_resident_masked_region_adapter_parity(expression):
    x = np.array([-9.0, 4, -1, 16])
    for m in (np.array([False, True, False, True]), np.zeros(4, dtype=bool)):
        inputs = {"x": x, "m": m}
        signature = {"x": "float64", "m": "bool"}
        baseline = NativeGraph.from_expression(expression, signature, jit=False)
        expected, reference = baseline.evaluate(inputs, tile_items=3, return_report=True)
        typed = NativeGraph.from_expression(expression, signature, jit=False, _executor="c-blosc2")
        actual, report = typed.evaluate(inputs, tile_items=3, raise_mask=1, return_report=True)
        assert actual.tobytes() == expected.tobytes()
        assert report["fp_flags"] == reference["fp_flags"] == 0
        assert typed.reduction_info()["where"]


def test_late_stage_fp_recovery_and_unary_pos():
    graph = NativeGraph.from_expression(
        "sum(y) + sqrt(x)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    schedule = graph.specialize({"x": ("float64", (3,)), "y": ("float64", (3,))})
    with pytest.raises(blosc2.MenudetError) as failure:
        schedule.execute({"x": np.array([-1.0, 1.0, 4.0]), "y": np.ones(3)}, 1)
    assert failure.value.stage == 1
    assert failure.value.fp_status["flags"] & 1
    result, report = schedule.execute({"x": np.array([0.0, 1.0, 4.0]), "y": np.ones(3)}, 1)
    np.testing.assert_array_equal(result, [3.0, 4.0, 5.0])
    assert report["fp_flags"] == 0

    x = np.arange(1, 6, dtype="float32")
    pos_base = NativeGraph.from_expression("+x - sum(x)", {"x": "float32"}, jit=False)
    pos_typed = NativeGraph.from_expression("+x - sum(x)", {"x": "float32"}, jit=False, _executor="c-blosc2")
    np.testing.assert_array_equal(pos_typed.evaluate({"x": x}), pos_base.evaluate({"x": x}))


def test_staged_map_export_and_jit_policy_parity():
    expression = "x - sum(x)"
    baseline = NativeGraph.from_expression(expression, {"x": "float32"}, jit=True)
    if baseline.has_jit:
        typed = NativeGraph.from_expression(
            expression, {"x": "float32"}, require_jit=True, _executor="c-blosc2"
        )
        assert typed.has_jit
    else:
        typed = NativeGraph.from_expression(expression, {"x": "float32"}, jit=True, _executor="c-blosc2")
        with pytest.raises(blosc2.MenudetError):
            NativeGraph.from_expression(expression, {"x": "float32"}, require_jit=True, _executor="c-blosc2")
    with pytest.raises(ValueError, match="elementwise"):
        typed.map_json()
    result, report = typed.evaluate({"x": np.arange(5, dtype="float32")}, return_report=True)
    np.testing.assert_array_equal(result, [-10, -9, -8, -7, -6])
    assert report["jit_stages"] + report["interpreter_stages"] == 2


def test_explicit_typed_masked_stage_binding_and_capability():
    def plan(expression, signature):
        nodes, names, root, _ = lower_typed_graph(expression, {}, input_dtypes=signature, allow_stages=True)
        return prepare_typed_plan(nodes, len(names), root), names

    mapped, names0 = plan("x * 2", {"x": "float64"})
    reduced, names1 = plan("sum(sqrt(v), where=m)", {"v": "float64", "m": "bool"})
    assert names0 == ("x",)
    bindings = [[-1], [0 if name == "v" else -2 for name in names1]]
    native = blosc2.blosc2_ext._TypedExecPlan.assemble([mapped, reduced], bindings, 2)
    x = np.array([-9.0, 4, -1, 16])
    m = np.array([False, True, False, True])
    schedule = native.specialize(
        [(x.dtype, x.shape), (m.dtype, m.shape)], tile_items=3, intermediate_budget=x.nbytes
    )
    del native, mapped, reduced
    prepare_typed_plan.cache_clear()
    gc.collect()
    actual, report = schedule.execute((x, m), 1)
    assert actual == np.sqrt(x[m] * 2).sum()
    assert report["stages"] == 2
    assert report["live_intermediate_bytes"] == 0
    assert not report["fp_flags"]

    class Source:
        shape = x.shape
        dtype = x.dtype
        calls = 0

        def read(self, indices):
            self.calls += 1
            pytest.fail("masked stage capability rejection reached a reader")

    source = Source()
    with pytest.raises(NotImplementedError, match="Reader-backed masked"):
        schedule.execute_sources((source, m))
    assert source.calls == 0


def test_imported_menudet_stages_without_native_artifact_parser(monkeypatch):
    from test_native_graph_stages import trusted_document

    document = trusted_document()
    values = {"x": np.arange(5, dtype="float32")}
    baseline = NativeGraph(document, jit=False)
    expected, reference = baseline.evaluate(values, return_report=True)

    def forbidden(*args, **kwargs):
        pytest.fail("typed imported stages consulted a native interchange/artifact parser")

    monkeypatch.setattr("blosc2.graph_plan.NativeGraphHandle", forbidden)
    typed = NativeGraph(document, jit=False, _executor="c-blosc2")
    typed = NativeGraph.from_json(typed.to_json(), jit=False, _executor="c-blosc2")
    assert typed.stages == baseline.stages
    schedule = typed.specialize({"x": ("float32", (5,))}, intermediate_budget=20)
    assert schedule.stage_info(0) == {
        "shape": (5,),
        "dtype": np.dtype("float32"),
        "bytes": 20,
        "last_consumer": 1,
    }
    with pytest.raises(blosc2.MenudetError, match="budget"):
        typed.specialize({"x": ("float32", (5,))}, intermediate_budget=19)
    with pytest.raises(ValueError, match="elementwise"):
        typed.map_json()
    del typed
    gc.collect()
    actual, report = schedule.execute(values)
    assert actual.tobytes() == expected.tobytes()
    assert report["fp_flags"] == reference["fp_flags"]


@pytest.mark.parametrize(
    "mutation", ["cycle", "missing", "signature", "unreachable", "root", "capability", "nested"]
)
def test_imported_stage_dependency_validation(mutation):
    from test_native_graph_stages import trusted_document

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
    with pytest.raises(blosc2.MenudetError):
        NativeGraph(document, jit=False, _executor="c-blosc2")


def test_imported_shared_computed_region_and_final_cast():
    document = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-0.1",
        "requires": ["numeric", "staged"],
        "nodes": [
            {"id": 0, "op": "input", "name": "x", "dtype": "float32"},
            {"id": 1, "op": "function", "name": "sqrt", "args": [0]},
            {"id": 2, "op": "add", "args": [1, 1]},
        ],
        "root": 2,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    typed = NativeGraph(document, jit=False, _executor="c-blosc2")
    baseline = NativeGraph(document, jit=False)
    values = {"x": np.array([1, 4, 9], dtype="float32")}
    actual, report = typed.evaluate(values, intermediate_budget=12, return_report=True)
    assert actual.tobytes() == baseline.evaluate(values).tobytes()
    assert report["stages"] == 2
    assert report["intermediate_bytes"] == 12
    cast_document = json.loads(NativeGraph.from_expression("x + 1", {"x": "int8"}, jit=False).to_json())
    cast_document["output"] = {"dtype": "int16", "casting": "safe"}
    typed = NativeGraph(cast_document, jit=False, _executor="c-blosc2")
    baseline = NativeGraph(cast_document, jit=False)
    values = {"x": np.array([1, 2, 3], dtype="int8")}
    assert typed.inferred_dtype == baseline.inferred_dtype
    actual, report = typed.evaluate(values, return_report=True)
    assert actual.dtype == np.dtype("int16")
    assert actual.tobytes() == baseline.evaluate(values).tobytes()
    assert report["stages"] == baseline.stages


def test_imported_masked_stage_adapter_parity():
    first = json.loads(NativeGraph.from_expression("x * 2", {"x": "float64"}, jit=False).to_json())
    second = json.loads(
        NativeGraph.from_expression(
            "sum(sqrt(v), where=m)", {"v": "float64", "m": "bool"}, jit=False
        ).to_json()
    )
    document = {
        "format": "menudet-staged-graph-1",
        "semantics": "menudet-numpy-0.1",
        "requires": ["numeric", "staged"],
        "inputs": [{"name": "x", "dtype": "float64"}, {"name": "m", "dtype": "bool"}],
        "root": 1,
        "stages": [
            {"id": 0, "kind": "graph", "inputs": {"x": {"input": "x"}}, "graph": first},
            {"id": 1, "kind": "graph", "inputs": {"v": {"stage": 0}, "m": {"input": "m"}}, "graph": second},
        ],
    }
    inputs = {"x": np.array([-9.0, 4, -1, 16]), "m": np.array([False, True, False, True])}
    baseline = NativeGraph(document, jit=False)
    expected, reference = baseline.evaluate(inputs, raise_mask=1, return_report=True)
    typed = NativeGraph(document, jit=False, _executor="c-blosc2")
    actual, report = typed.evaluate(inputs, raise_mask=1, return_report=True)
    assert actual.tobytes() == expected.tobytes()
    assert report["fp_flags"] == reference["fp_flags"] == 0
    assert report["intermediate_bytes"] == inputs["x"].nbytes


def test_trusted_source_features_reject_before_required_jit():
    with pytest.raises((blosc2.MenudetError, NotImplementedError), match=r"feature|control\.flow|permitted"):
        blosc2.blosc2_ext._TypedExecPlan.from_source(
            b"0.1",
            b"def k(x):\n    if x > 0:\n        return x\n    return -x\n",
            {"x": np.dtype("float64")},
            [],
            np.dtype("float64"),
            execution=2,
            compiler=2,
        )
