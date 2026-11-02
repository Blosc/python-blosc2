"""Resident prefilter binding, ownership and scheduling contracts, not native matrices."""

import gc
import json
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2
from blosc2.native_graph import prepare_resident


@pytest.fixture(autouse=True)
def resident_prefilter_runtime(native_graph_runtime):
    if not getattr(blosc2.blosc2_ext, "menudet_resident_prefilter_available", lambda: False)():
        pytest.skip("Installed extension lacks native resident prefilter scheduling")


def prepared(inputs, expression="x*x - 1"):
    with blosc2.jit_options(jit=False, jit_backend="cc", math_backend="system"):
        result = prepare_resident(expression, inputs, jit=False, optimization="aggressive")
    assert result is not None
    return result[2]


@pytest.mark.parametrize(
    ("dtype", "expression"), [("float64", "x*x - 1"), ("float32", "x < 2"), ("int64", "x + 1")]
)
def test_direct_resident_prefilter_tail_grid_and_rebinding(dtype, expression):
    x = (np.arange(471) % 3).astype(dtype).reshape(3, 157)
    schedule = prepared({"x": x}, expression)
    executor = schedule.kernel._handle.resident_prefilter(
        x.shape, workers=2, block_items=128, chunk_items=256, graph=schedule.plan._handle
    )
    first, report = executor.evaluate({"x": x})
    np.testing.assert_array_equal(first, eval(expression, {"x": x}))
    assert report["route"] == "resident-prefilter"
    assert report["chunk_calls"] == 2
    assert report["evaluated_blocks"] == 4
    assert report["driver_buffer_bytes"] == report["input_buffer_bytes"] == report["output_copy_bytes"] == 0
    assert not report["context_reused"]
    destination = np.empty_like(first)
    x[:] = 2
    second, report = executor.evaluate({"x": x}, out=destination)
    assert second is destination
    assert report["context_reused"]
    np.testing.assert_array_equal(second, eval(expression, {"x": x}))


def test_output_canaries_and_mapping_order():
    x = np.arange(517, dtype="float64")
    y = -x.copy()
    schedule = prepared({"x": x, "y": y}, "x < y")
    executor = schedule.kernel._handle.resident_prefilter(x.shape, workers=2, block_items=128)
    backing = np.ones(x.size + 2, dtype="bool")
    output = backing[1:-1]
    result, _ = executor.evaluate({"y": y, "x": x}, out=output)
    assert result is output
    np.testing.assert_array_equal(output, x < y)
    assert backing[0]
    assert backing[-1]


@pytest.mark.parametrize("invalid", ["shape", "dtype", "strides", "readonly-output", "alias", "names"])
def test_binding_rejections_do_not_poison_context(invalid):
    x = np.arange(257, dtype="float64")
    executor = prepared({"x": x}).kernel._handle.resident_prefilter(x.shape, workers=2, block_items=128)
    inputs = {"x": x}
    output = np.empty_like(x)
    if invalid == "shape":
        inputs = {"x": x[:-1]}
    elif invalid == "dtype":
        inputs = {"x": x.astype("float32")}
    elif invalid == "strides":
        inputs = {"x": x[::-1]}
    elif invalid == "readonly-output":
        output.flags.writeable = False
    elif invalid == "alias":
        output = x
    else:
        inputs = {"wrong": x}
    with pytest.raises(ValueError):
        executor.evaluate(inputs, out=output)
    result, _ = executor.evaluate({"x": x})
    np.testing.assert_array_equal(result, x * x - 1)


@pytest.mark.parametrize("integrated", [False, True])
def test_cached_context_does_not_retain_numeric_buffers(integrated):
    x = np.arange(300003 if integrated else 257, dtype="float64")
    schedule = prepared({"x": x})
    if integrated:
        output, _ = schedule.execute({"x": x}, workers=2)
    else:
        executor = schedule.kernel._handle.resident_prefilter(x.shape, workers=2, block_items=128)
        output, _ = executor.evaluate({"x": x})
    refs = weakref.ref(x), weakref.ref(output)
    del x, output
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_fp_status_is_merged_and_reset():
    x = np.full(1001, np.finfo("float64").max)
    executor = prepared({"x": x}).kernel._handle.resident_prefilter(x.shape, workers=2, block_items=128)
    result, status = executor.evaluate({"x": x})
    assert np.isposinf(result).all()
    if status["fp_supported"]:
        assert status["fp_flags"] & 4
    x[:] = 2
    result, status = executor.evaluate({"x": x})
    np.testing.assert_array_equal(result, 3)
    assert status["fp_flags"] == 0


def test_configured_resident_uses_no_python_tile_scheduler(monkeypatch):
    x = np.arange(300003, dtype="float64")
    schedule = prepared({"x": x})

    def forbidden(*args, **kwargs):
        pytest.fail("Configured resident map used Python tile scheduling")

    monkeypatch.setattr("blosc2.resident_execution.run_dynamic", forbidden)
    result, report = schedule.execute({"x": x}, workers=2)
    np.testing.assert_array_equal(result, x * x - 1)
    assert report["route"] == "resident-prefilter"
    assert report["tasks"] == 2
    assert report["interpreter_stages"] == 1


def test_contexts_are_per_calling_thread():
    shape = (300003,)
    schedule = prepared({"x": np.zeros(shape)})
    barrier = threading.Barrier(3)

    def run(value):
        x = np.full(shape, float(value))
        barrier.wait(timeout=10)
        first, cold = schedule.execute({"x": x}, workers=2)
        second, warm = schedule.execute({"x": x}, workers=2)
        np.testing.assert_array_equal(first, x * x - 1)
        np.testing.assert_array_equal(second, first)
        return cold["context_reused"], warm["context_reused"]

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(run, [2, 3, 4])) == [(False, True)] * 3


def test_native_workspaces_are_bounded_per_caller():
    import blosc2.native_graph as native

    native.clear_resident_prefilter_cache()
    x = np.arange(300003, dtype="float64")
    seen_contexts = set()
    for expression in ("x*x - 1", "x + 2", "x - 3"):
        schedule = prepared({"x": x}, expression)
        result, _ = schedule.execute({"x": x}, workers=2)
        np.testing.assert_array_equal(result, eval(expression, {"x": x}))
        assert len(native._resident_prefilter_contexts.contexts) <= 2
        seen_contexts.update(id(ctx) for ctx in native._resident_prefilter_contexts.contexts.values())
    # The third expression rebinds the evicted idle context in-place without
    # allocating a third blosc2_context/thread pool.
    assert len(seen_contexts) <= 2
    native.clear_resident_prefilter_cache()
    assert not native._resident_prefilter_contexts.contexts


@pytest.mark.parametrize("size", [260, 300])
def test_multi_chunk_tail_reuse_preserves_block_geometry(size):
    x = np.arange(size, dtype="float64")
    schedule = prepared({"x": x}, "x + 1")
    context = schedule.kernel._handle.resident_prefilter(
        x.shape, workers=2, block_items=128, chunk_items=256, graph=schedule.plan._handle
    )
    first, first_report = context.evaluate({"x": x})
    second, second_report = context.evaluate({"x": x + 10})
    np.testing.assert_array_equal(first, x + 1)
    np.testing.assert_array_equal(second, x + 11)
    assert first_report["chunk_calls"] == 2
    assert second_report["chunk_calls"] == 2
    assert first_report["evaluated_blocks"] == 3
    assert second_report["evaluated_blocks"] == 3


def test_configured_evaluate_explicit_output_fast_path(monkeypatch):
    import importlib

    lazy = importlib.import_module("blosc2.lazyexpr")
    x = np.linspace(-2.0, 2.0, 300_003, dtype="float64")
    out = np.empty_like(x)
    monkeypatch.setattr(lazy, "LazyExpr", lambda *a, **k: pytest.fail("Entered LazyExpr fallback"))
    with blosc2.jit_options(jit=True, cflags="-O2"):
        result = blosc2.evaluate("x*x - 1", local_dict={"x": x}, out=out)
    assert result is out
    np.testing.assert_allclose(out, x * x - 1)


def test_native_scheduler_errors_are_not_replayed(monkeypatch):
    schedule = prepared({"x": np.zeros(300003)})
    calls = []

    class FailedContext:
        def evaluate(self, *args, **kwargs):
            calls.append(1)
            raise ArithmeticError("native scheduling failed")

    class Handle:
        def resident_prefilter(self, *args, **kwargs):
            return FailedContext()

    monkeypatch.setattr(schedule.kernel, "_handle", Handle())
    monkeypatch.setattr(
        "blosc2.resident_execution.evaluate_resident", lambda *a, **k: pytest.fail("Replayed")
    )
    with pytest.raises(ArithmeticError, match="native scheduling failed"):
        schedule.execute({"x": np.zeros(300003)}, workers=2)
    assert calls == [1]


def test_graph_rejection_remains_authoritative(monkeypatch):
    schedule = prepared({"x": np.zeros(300003)})
    monkeypatch.setattr(schedule, "partitionable", False)
    with pytest.raises(ValueError, match="rejected graph"):
        schedule.execute({"x": np.zeros(300003)}, workers=2)


def test_graph_proof_cannot_attest_changed_captures():
    x = np.arange(257, dtype="int64")
    schedule = prepared({"x": x}, "x + 1")
    document = json.loads(schedule.plan.map_json())
    assert document["constants"]
    document["constants"][0]["value"] = "2"
    altered = blosc2.MenudetKernel.from_json(json.dumps(document), jit=False)
    with pytest.raises(ValueError, match="documents disagree"):
        altered._handle.resident_prefilter(x.shape, workers=2, graph=schedule.plan._handle)


def test_native_graph_rejection_overrides_artifact_proof():
    from blosc2.graph_plan import NativeGraph

    x = np.ones(257)
    schedule = prepared({"x": x})
    rejected = NativeGraph.from_expression("x + sum(x)", {"x": "float64"}, jit=False)
    with pytest.raises(NotImplementedError, match="graph rejected"):
        schedule.kernel._handle.resident_prefilter(x.shape, workers=2, graph=rejected._handle)
