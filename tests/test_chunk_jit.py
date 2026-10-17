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
