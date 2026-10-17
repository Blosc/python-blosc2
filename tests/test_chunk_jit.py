"""Chunk fallback must retain native lowering and explicit execution policy."""

import importlib

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("jit", [False, True])
def test_chunk_lowering_respects_jit_policy(native_graph_runtime, monkeypatch, jit):
    native = importlib.import_module("blosc2.native_graph")
    lazy = importlib.import_module("blosc2.lazyexpr")
    original = native.lower_native_graph
    plans = []

    def lower(*args, **kwargs):
        assert kwargs["jit"] is jit
        result = original(*args, **kwargs)
        plans.append(result[0])
        return result

    monkeypatch.setattr(native, "lower_native_graph", lower)
    x = np.arange(100_003.0, dtype="float64")
    output = np.empty_like(x)
    with blosc2.jit_options(jit=jit):
        result = lazy.evaluate_chunk("x * 2 + 1", {"x": x}, out=output)
    assert result is output
    np.testing.assert_array_equal(result, x * 2 + 1)
    assert len(plans) == 1
    if not jit:
        assert not plans[0].has_jit


def test_chunk_execution_failure_is_not_replayed(native_graph_runtime, monkeypatch):
    native = importlib.import_module("blosc2.native_graph")
    lazy = importlib.import_module("blosc2.lazyexpr")

    class FailedSchedule:
        def info(self):
            return {"shape": (3,), "dtype": np.dtype("float64")}

        def execute(self, *args, **kwargs):
            raise ArithmeticError("native execution failed")

    class Plan:
        def info(self):
            return {}

        def specialize(self, *args, **kwargs):
            return FailedSchedule()

    monkeypatch.setattr(native, "lower_native_graph", lambda expr, **kwargs: (Plan(), expr.operands, False))
    with pytest.raises(ArithmeticError, match="native execution failed"):
        lazy.evaluate_chunk("x + 1", {"x": np.ones(3)})


@pytest.mark.parametrize("policy", [{"jit_backend": "cc"}, {"fp_accuracy": blosc2.FPAccuracy.HIGH}])
def test_chunk_does_not_compile_with_unsupported_graph_policy(native_graph_runtime, monkeypatch, policy):
    native = importlib.import_module("blosc2.native_graph")
    lazy = importlib.import_module("blosc2.lazyexpr")
    original = native.lower_native_graph
    calls = []

    def lower(*args, **kwargs):
        calls.append(kwargs["jit"])
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "lower_native_graph", lower)
    x = np.arange(17.0)
    with blosc2.jit_options(**policy):
        result = lazy.evaluate_chunk("x + 1", {"x": x})
    np.testing.assert_array_equal(result, x + 1)
    assert calls == [False]


def test_integral_math_uses_native_parallel_tasks(native_graph_runtime):
    x = np.arange(600_003, dtype="int64") % 97
    y = (x % 31) + 1
    for expression, expected in [("minimum(x,y)", np.minimum(x, y)), ("fmod(x,y)", np.fmod(x, y))]:
        plan = blosc2.NativeGraph.from_expression(expression, {"x": x.dtype, "y": y.dtype}, jit=True)
        assert plan.info()["partitionable"]
        out = np.empty_like(x)
        result, report = plan.evaluate({"x": x, "y": y}, out=out, workers=4, return_report=True)
        assert result is out
        assert report["tasks"] == 4
        np.testing.assert_array_equal(result, expected)


def test_small_transcendental_map_uses_caller_and_workers(native_graph_runtime, monkeypatch):
    import blosc2.resident_execution as resident

    pool = resident.resident_pool()
    submissions = []

    class RecordingPool:
        def submit(self, *args):
            future = pool.submit(*args)
            submissions.append(future)
            return future

    monkeypatch.setattr(resident, "resident_pool", RecordingPool)
    x = np.linspace(0.1, 1.0, 100_003)
    plan = blosc2.NativeGraph.from_expression("sin(x) + cos(x)", {"x": x.dtype}, jit=False)
    expected = plan.evaluate({"x": x})
    result, report = plan.evaluate({"x": x}, workers=8, return_report=True)
    assert report["tasks"] == 4
    assert len(submissions) == 3
    assert all(future.done() for future in submissions)
    assert report["jit_stages"] == 0
    np.testing.assert_array_equal(result, expected)


def test_native_executor_joins_after_interrupted_wait(native_graph_runtime, monkeypatch):
    import blosc2.resident_execution as resident

    pool = resident.resident_pool()
    submitted = []

    class InterruptedFuture:
        def __init__(self, future):
            self.future = future
            self.calls = 0

        def result(self):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt("interrupted join")
            return self.future.result()

        def done(self):
            return self.calls > 1 and self.future.done()

    class InterruptedPool:
        def submit(self, *args):
            future = pool.submit(*args)
            submitted.append(future)
            return InterruptedFuture(future)

    monkeypatch.setattr(resident, "resident_pool", InterruptedPool)
    x = np.arange(600_003.0)
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    with pytest.raises(KeyboardInterrupt, match="interrupted join"):
        plan.evaluate({"x": x}, workers=4)
    assert len(submitted) == 3
    assert all(future.done() for future in submitted)


def test_native_executor_rejects_late_job_after_enqueue_failure(native_graph_runtime, monkeypatch):
    from threading import Event

    import blosc2.resident_execution as resident

    pool = resident.resident_pool()
    release = Event()
    submitted = []

    def late_job(function, args):
        assert release.wait(timeout=10)
        function(*args)

    class EnqueueThenFailPool:
        def submit(self, function, *args):
            submitted.append(pool.submit(late_job, function, args))
            raise RuntimeError("failed after enqueue")

    monkeypatch.setattr(resident, "resident_pool", EnqueueThenFailPool)
    x = np.arange(600_003.0)
    out = np.full_like(x, -999.0)
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    try:
        with pytest.raises(RuntimeError, match="failed after enqueue"):
            plan.evaluate({"x": x}, out=out, workers=4)
    finally:
        release.set()
        for future in submitted:
            future.result(timeout=10)
    # The late job must not dereference invocation pointers after the caller
    # returned, even though submit() never supplied its Future for joining.
    np.testing.assert_array_equal(out, -999.0)
