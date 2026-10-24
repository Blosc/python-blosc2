"""Explicit reusable evaluator contracts, independent of NumExpr availability."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2


def test_rebind_shape_dtype_output_and_release():
    evaluator = blosc2.prepare_evaluate("x + y", order="F")
    assert evaluator.input_names == ("x", "y")
    for shape, dtype in [((3, 4), "float64"), ((17,), "int64")]:
        x = np.ones(shape, dtype=dtype)
        out = np.empty_like(x)
        refs = [weakref.ref(x), weakref.ref(out)]
        assert evaluator({"x": x, "y": x}, out=out) is out
        np.testing.assert_array_equal(out, 2)
        del x, out
        gc.collect()
        assert all(ref() is None for ref in refs)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Host thread pools are unavailable on WASM")
def test_concurrent_rebinding():
    evaluator = blosc2.prepare_evaluate("x * 2")

    def run(value):
        x = np.full(257, value, dtype="float64")
        result = evaluator.evaluate({"x": x})
        np.testing.assert_array_equal(result, x * 2)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(12)))


def test_missing_operand_does_not_capture_caller():
    evaluator = blosc2.prepare_evaluate("x + y")
    with pytest.raises(NameError, match="y"):
        evaluator({"x": np.ones(3)})


def test_options_are_validated_and_overrides_do_not_mutate():
    with pytest.raises(ValueError, match="optimization"):
        blosc2.prepare_evaluate("x", optimization="invalid")
    evaluator = blosc2.prepare_evaluate("x + 1", order="F")
    x = np.arange(12.0).reshape(3, 4)
    assert evaluator({"x": x}, order="C").flags.c_contiguous
    assert evaluator({"x": x}).flags.f_contiguous


@pytest.mark.parametrize("expression", ["x", "copy(x)", "x + 1", "where(x > 0, x, -x)"])
@pytest.mark.parametrize("direction", [1, -1])
def test_overlapping_destination(expression, direction):
    data = np.arange(34.0)
    x, out = (data[:-1], data[1:]) if direction == 1 else (data[1:], data[:-1])
    original = x.copy()
    expected = original + 1 if expression == "x + 1" else original
    assert blosc2.evaluate(expression, local_dict={"x": x}, out=out) is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize(
    "options",
    [
        {"optimization": "invalid"},
        {"truediv": "invalid"},
        {"sanitize": 1},
        {"disable_cache": 1},
        {"unknown": True},
    ],
)
def test_invalid_options_rejected_before_output_writes(order, options):
    x = np.arange(12.0).reshape(3, 4)
    out = np.full(x.shape, -999.0, order=order)
    with pytest.raises((ValueError, TypeError)):
        blosc2.evaluate("x + 1", local_dict={"x": x}, out=out, **options)
    np.testing.assert_array_equal(out, -999)


@pytest.mark.parametrize(("reducer", "expected"), [("sum", 0), ("prod", 1)])
def test_empty_reduction_identity(reducer, expected):
    out = np.empty(())
    result = blosc2.evaluate(f"{reducer}(x)", local_dict={"x": np.empty(0)}, out=out)
    assert result is out
    assert result == expected


def test_wrong_output_shape_rejected():
    with pytest.raises(ValueError):
        blosc2.evaluate("x + 1", local_dict={"x": np.ones((2, 3))}, out=np.empty((3, 2)))


@pytest.mark.parametrize("mode", ["safe", "full"])
@pytest.mark.parametrize("layout", ["C", "F"])
@pytest.mark.parametrize(
    "options",
    [
        {"optimization": "none"},
        {"optimization": "moderate"},
        {"optimization": "aggressive"},
    ],
)
def test_keyword_behavior_across_layouts(mode, layout, options):
    x = np.arange(12.0).reshape(3, 4)
    out = np.empty(x.shape, order=layout)
    with blosc2.expression_evaluation(mode):
        result = blosc2.evaluate("x / 2 + 1", local_dict={"x": x}, out=out, **options)
    assert result is out
    np.testing.assert_array_equal(result, x / 2 + 1)


def test_safe_mode_rejects_unsafe_expression():
    with blosc2.expression_evaluation("safe"), pytest.raises(blosc2.exceptions.UnsafeDeserializationError):
        blosc2.evaluate("__import__('os')", local_dict={})


@pytest.mark.parametrize("out_provided", [False, True])
def test_empty_axis_reduction(out_provided):
    x = np.empty((2, 0, 3))
    out = np.empty((2, 3)) if out_provided else None
    result = blosc2.evaluate("sum(x, axis=1)", local_dict={"x": x}, out=out)
    if out_provided:
        assert result is out
    np.testing.assert_array_equal(result, x.sum(axis=1))


@pytest.mark.parametrize("reducer", ["min", "max"])
def test_empty_extrema_raise(reducer):
    with pytest.raises(ValueError):
        blosc2.evaluate(f"{reducer}(x)", local_dict={"x": np.empty(0)})


def test_prepared_broadcast_scalar_and_compressed_rebinding():
    evaluator = blosc2.prepare_evaluate("x + y")
    x = np.arange(12.0).reshape(3, 4)
    y = np.arange(4.0)
    np.testing.assert_array_equal(evaluator({"x": x, "y": y}), x + y)
    stored = blosc2.asarray(x)
    np.testing.assert_array_equal(evaluator({"x": stored, "y": 2}), x + 2)
    assert blosc2.prepare_evaluate("3.0")({}) == 3.0


def test_numpy_fallback_options_without_numexpr(monkeypatch):
    import importlib

    module = importlib.import_module("blosc2.lazyexpr")
    monkeypatch.setattr(module, "numexpr", None)
    x = np.arange(12.0).reshape(3, 4)
    evaluator = blosc2.prepare_evaluate("x / 2", order="F")
    result = evaluator({"x": x})
    assert result.flags.f_contiguous
    np.testing.assert_array_equal(result, x / 2)
    with pytest.raises(ValueError, match="optimization"):
        evaluator({"x": x}, optimization="invalid")


def test_prepared_safe_policy_is_preserved(monkeypatch):
    import importlib

    module = importlib.import_module("blosc2.lazyexpr")
    original = module.evaluate
    seen = []

    def record(*args, **kwargs):
        seen.append(module.evaluation_mode())
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate", record)
    with blosc2.expression_evaluation("safe"):
        evaluator = blosc2.prepare_evaluate("x + 1")
    with blosc2.expression_evaluation("full"):
        evaluator({"x": np.arange(3.0)})
    assert seen == ["safe"]
