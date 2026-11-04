"""NumPy 1 automatic evaluation is bounded NumPy, not native NumPy-2 semantics."""

import importlib

import numpy as np
import pytest

import blosc2

lazy = importlib.import_module("blosc2.lazyexpr")
native = importlib.import_module("blosc2.native_graph")
evaluator = importlib.import_module("blosc2.expression_evaluator")


@pytest.fixture
def value_promotion_policy(monkeypatch):
    # Exercise admission on modern hosts too; numerical expectations always use
    # the installed NumPy. Actual NumPy 1.26 CI checks its promotion semantics.
    for module in (lazy, native, evaluator):
        monkeypatch.setattr(module, "_NUMPY_VALUE_PROMOTION", True)


@pytest.mark.parametrize("mode", ["legacy", "typed"])
@pytest.mark.parametrize("persisted", [False, True])
def test_automatic_evaluation_uses_bounded_numpy(
    value_promotion_policy, monkeypatch, tmp_path, mode, persisted
):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", mode)
    values = np.arange(96, dtype=np.float32).reshape(12, 8)
    options = {"urlpath": str(tmp_path / "a.b2nd")} if persisted else {}
    a = blosc2.asarray(values, chunks=(3, 8), blocks=(1, 4), **options)

    def forbidden(*args, **kwargs):
        pytest.fail("NumPy value-promotion policy reached native numerical execution")

    monkeypatch.setattr(native, "lower_native_graph", forbidden)
    monkeypatch.setattr(native, "compile_plan", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    original = lazy.evaluate_chunk
    sizes = []

    def record(expression, operands=None, **kwargs):
        sizes.extend(value.size for value in (operands or {}).values() if type(value) is np.ndarray)
        return original(expression, operands, **kwargs)

    monkeypatch.setattr(lazy, "evaluate_chunk", record)
    expr = blosc2.lazyexpr("a + 0.1", {"a": a})
    expected = values + 0.1
    assert expr.dtype == expected.dtype
    assert expr.shape == expected.shape
    assert not hasattr(expr, "_metadata_native")
    np.testing.assert_array_equal(expr[:], expected)
    np.testing.assert_array_equal(expr.compute()[:], expected)
    np.testing.assert_allclose(expr.sum(), expected.sum(), rtol=1e-6)
    np.testing.assert_allclose(expr.sum(axis=0), expected.sum(axis=0), rtol=1e-6)
    np.testing.assert_array_equal(expr[2:5], expected[2:5])
    out = np.empty((12, 16), dtype=expected.dtype)[:, ::2]
    assert expr.compute(out=out) is out
    np.testing.assert_array_equal(out, expected)
    assert sizes
    assert max(sizes) <= np.prod(a.chunks)


def test_resident_and_cached_evaluate_decline_native(value_promotion_policy, monkeypatch):
    x = np.arange(24, dtype=np.float32)

    def forbidden(*args, **kwargs):
        pytest.fail("Automatic resident preparation reached the native compiler")

    monkeypatch.setattr(native, "compile_plan", forbidden)
    monkeypatch.setattr(native, "lower_native_graph", forbidden)
    for _ in range(2):
        expr = blosc2.lazyexpr("x + 1", {"x": x})
        assert not hasattr(expr, "_resident_native")
        assert not hasattr(expr, "_prepared_native")
        np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}), x + 1)
        np.testing.assert_array_equal(expr[:], x + 1)
    assert native.prepare_resident("x + 1", {"x": x}) is None
    assert native.compute_resident_if_eligible(expr, slice(None)) is None
    assert native.prepare_native_first(expr, slice(None)) is None
    assert native.compute_native_prefilter_if_eligible(expr, slice(None)) is None
    assert evaluator._prepare_typed_map_kernel("x + 1", {"x": x}) is None


def test_explicit_native_apis_remain_available(value_promotion_policy, native_graph_runtime):
    x = np.arange(24, dtype=np.float64)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expr.compute(_require_native=True)[:], x + 1)
    kernel = expr.native_kernel()
    np.testing.assert_array_equal(kernel.evaluate_block({"o0": x}, block_shape=x.shape), x + 1)


@pytest.mark.parametrize("scalar", [0.1, np.float64(0.1), np.array(0.1, dtype=np.float64)])
def test_fallback_metadata_preserves_scalar_promotion(value_promotion_policy, monkeypatch, scalar):
    values = np.full(24, 0.1, dtype=np.float32)
    a = blosc2.asarray(values, chunks=(6,), blocks=(3,))

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata preparation read compressed input values")

    with monkeypatch.context() as patch:
        patch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        expr = blosc2.lazyexpr("a + scalar", {"a": a, "scalar": scalar})
        assert expr.dtype == (values + scalar).dtype
        assert expr.shape == values.shape
    np.testing.assert_array_equal(expr[:], values + scalar)
    comparison = blosc2.lazyexpr("a > scalar", {"a": a, "scalar": scalar})
    np.testing.assert_array_equal(comparison[:], values > scalar)
    assert comparison.sum() == (values > scalar).sum()
