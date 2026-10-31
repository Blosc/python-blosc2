"""Aggressive powers default only in evaluate, with cache-isolated opt-outs."""

import math

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph


@pytest.mark.parametrize("expression", ["x**3", "pow(x, 3)"])
@pytest.mark.parametrize("compressed", [False, True])
def test_aggressive_power_is_default(expression, compressed):
    x = np.full(257, 1.515049231134367)
    inputs = {"x": blosc2.asarray(x) if compressed else x}
    strict = blosc2.evaluate(expression, local_dict=inputs, optimization="moderate")
    if not compressed:
        np.testing.assert_array_equal(strict, np.full_like(x, math.pow(float(x[0]), 3)))
    chain = (x * x) * x
    # Alternate policies with the same expression/signature to catch stale plans
    # and cached recipe metadata leaking a relaxed policy into explicit opt-outs.
    for options, expected in [
        ({}, chain),
        ({"optimization": "aggressive"}, chain),
        ({"optimization": "moderate"}, strict),
        ({"optimization": "none"}, strict),
        ({"optimization": "aggressive"}, chain),
        ({}, chain),
    ]:
        actual = blosc2.evaluate(expression, local_dict=inputs, **options)
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(blosc2.lazyexpr(expression, inputs)[:], strict)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "complex128"])
def test_aggressive_power_dtype_and_destination(dtype):
    x = np.linspace(0.25, 1.25, 257).astype(dtype)
    if dtype == "int32":
        x = np.arange(257, dtype=dtype)
    out = np.empty_like(x)
    result = blosc2.evaluate("(x + 1)**3", local_dict={"x": x}, out=out, optimization="aggressive")
    assert result is out
    assert result.dtype == x.dtype
    np.testing.assert_allclose(result, ((x + 1) * (x + 1)) * (x + 1), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("expression", ["x**2", "pow(x, 2)", "x**2.0", "pow(x, 2.0)"])
def test_default_square(expression):
    x = np.arange(257, dtype="float64") * 0.1
    np.testing.assert_array_equal(blosc2.evaluate(expression, local_dict={"x": x}), x * x)


def test_aggressive_does_not_enable_global_fast_math():
    x = np.full(257, 2.0**53)
    result = blosc2.evaluate("(x + 1) - x", local_dict={"x": x})
    np.testing.assert_array_equal(result, np.zeros_like(x))


def test_aggressive_nonconstant_exponent_retains_power():
    x = np.full(257, 1.515049231134367)
    y = np.full_like(x, 3)
    expected = np.full_like(x, math.pow(float(x[0]), 3))
    result = blosc2.evaluate("x**y", local_dict={"x": x, "y": y}, optimization="aggressive")
    np.testing.assert_array_equal(result, expected)


def test_aggressive_negative_power_and_compressed_output():
    x = np.linspace(0.5, 1.5, 65537)
    compressed = blosc2.asarray(x)
    out = blosc2.empty_like(compressed)
    actual = blosc2.evaluate("x**-3", local_dict={"x": compressed}, out=out, optimization="aggressive")
    assert actual is out
    inverse = 1 / x
    np.testing.assert_array_equal(actual[:], inverse * inverse * inverse)


def test_square_keeps_strict_rounding_with_cc():
    x = np.full(257, 1 + 2.0**-27)
    with blosc2.jit_options(jit=True, jit_backend="cc"):
        actual = blosc2.evaluate("x**2 - 1", local_dict={"x": x})
    np.testing.assert_array_equal(actual, x * x - 1)


def test_aggressive_caller_resolution_and_staged_policy():
    from blosc2.native_graph import compile_plan

    x = np.linspace(0.5, 1.5, 257)
    np.testing.assert_array_equal(blosc2.evaluate("x**3"), x * x * x)
    plan = compile_plan(
        "(x - sum(x**3))**3", False, input_dtypes=(("x", x.dtype),), optimization="aggressive"
    )
    document = plan.to_json()
    assert document.count('"optimization":"aggressive"') >= 2
    restored = NativeGraph(document, jit=False)
    actual, _ = restored.specialize({"x": (x.dtype, x.shape)}).execute({"x": x})
    base = x - np.sum(x * x * x)
    np.testing.assert_allclose(actual, base * base * base, rtol=1e-12)


def test_aggressive_power_rebinding_and_serialization():
    from blosc2.native_graph import compile_plan

    plan = compile_plan("x**3", False, input_dtypes=(("x", np.dtype("float64")),), optimization="aggressive")
    document = plan.to_json()
    assert '"optimization":"aggressive"' in document
    restored = NativeGraph(document, jit=False)
    x = np.full(257, 1.515049231134367)
    result, _ = restored.specialize({"x": (x.dtype, x.shape)}).execute({"x": x})
    np.testing.assert_array_equal(result, x * x * x)
    kernel = blosc2.MenudetKernel.from_json(plan.map_json(), jit=False)
    np.testing.assert_array_equal(kernel.evaluate({"x": x}), x * x * x)
    for value in [1.515049231134367, 1.25]:
        current = np.full_like(x, value)
        result = blosc2.evaluate("x**3", local_dict={"x": current}, optimization="aggressive")
        np.testing.assert_array_equal(result, current * current * current)


def test_optimization_validation():
    x = np.arange(33, dtype="float64")
    with pytest.raises(ValueError, match="optimization"):
        blosc2.evaluate("x**3", local_dict={"x": x}, optimization="fast")
