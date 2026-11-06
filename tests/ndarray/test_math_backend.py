"""Per-call math backend selection through the public evaluation APIs."""

import json
import sys

import numpy as np
import pytest

import blosc2

TOLERANCE = {"rtol": 1e-12, "atol": 1e-12}


@pytest.fixture(autouse=True)
def restore_options():
    previous = blosc2.get_jit_options()
    yield
    blosc2.set_jit_options(**previous)


def test_math_backend_invalid_value():
    with pytest.raises(ValueError, match="math_backend"):
        blosc2.evaluate("x + 1", local_dict={"x": np.arange(4.0)}, math_backend="automatic")


def test_math_backend_is_per_call_and_does_not_leak():
    data = np.arange(8.0)
    reference = blosc2.evaluate("sin(x) + cos(x)", local_dict={"x": data})
    assert blosc2.get_jit_options()["math_backend"] is None
    result = blosc2.evaluate("sin(x) + cos(x)", local_dict={"x": data}, math_backend="system")
    np.testing.assert_allclose(result, reference, **TOLERANCE)
    assert blosc2.get_jit_options()["math_backend"] is None


def test_lazyexpr_compute_and_prefilter_follow_math_backend():
    data = np.arange(32.0).reshape(8, 4)
    expr = blosc2.lazyexpr("sin(x) + cos(x)", {"x": blosc2.asarray(data)})
    expected = np.sin(data) + np.cos(data)
    np.testing.assert_allclose(expr.compute(math_backend="system")[:], expected, **TOLERANCE)
    np.testing.assert_allclose(expr.compute(_getitem=True, math_backend="system"), expected, **TOLERANCE)


def test_reduction_accepts_math_backend():
    data = np.arange(16.0)
    expr = blosc2.lazyexpr("sin(x)", {"x": data})
    result = expr.sum(math_backend="system")
    np.testing.assert_allclose(result, np.sin(data).sum(), **TOLERANCE)


def test_small_compressed_explicit_backend_overrides_environment(monkeypatch):
    monkeypatch.setenv("ME_SIMD_MATH_BACKEND", "bogus")
    data = np.arange(32.0).reshape(8, 4)
    operand = blosc2.asarray(data)
    expected = np.sin(data)
    result = blosc2.evaluate("sin(x)", local_dict={"x": operand}, math_backend="system")
    np.testing.assert_allclose(result, expected, **TOLERANCE)
    expr = blosc2.lazyexpr("sin(x)", {"x": operand})
    np.testing.assert_allclose(expr.compute(_getitem=True, math_backend="system"), expected, **TOLERANCE)


def test_context_math_backend_reaches_native_artifact():
    if blosc2.IS_WASM:
        pytest.skip("Native graph artifacts are unavailable under WebAssembly")
    expr = blosc2.lazyexpr("sin(x)", {"x": blosc2.asarray(np.arange(8.0))})

    with blosc2.jit_options(math_backend="system"):
        semantics = json.loads(expr.native_kernel().to_json())["semantics"]
    assert semantics["math"] == "system"


@pytest.mark.skipif(
    sys.platform != "darwin" or blosc2.IS_WASM,
    reason="Accelerate requires a macOS native build",
)
def test_accelerate_backend_matches_system():
    data = np.linspace(-3.0, 3.0, 1024)
    expected = blosc2.evaluate("sin(x) + exp(x)", local_dict={"x": data}, math_backend="system")
    result = blosc2.evaluate("sin(x) + exp(x)", local_dict={"x": data}, math_backend="accelerate")
    np.testing.assert_allclose(result, expected, **TOLERANCE)
    with blosc2.jit_options(math_backend="accelerate"):
        semantics = json.loads(
            blosc2.lazyexpr("sin(x)", {"x": blosc2.asarray(data)}).native_kernel().to_json()
        )["semantics"]
    assert semantics["math"] == "accelerate"
