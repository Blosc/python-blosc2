"""NumExpr API compatibility for resident inputs and an explicit NumPy output.

Numerical/backend cross-products belong in miniexpr. These representative
cases exercise missing expression forms, shape contracts and output policies.
"""

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
@pytest.mark.parametrize("expression", ["x", "x + 1", "copy(x)", "ones_like(x)", "where(x > 0, x, -x)"])
def test_empty_destination_identity(shape, expression):
    x = np.empty(shape)
    output = np.empty(shape)
    assert blosc2.evaluate(expression, local_dict={"x": x}, out=output) is output


@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
@pytest.mark.parametrize("layout", ["C", "F", "permuted"])
def test_numpy_output_order(order, layout):
    x = np.arange(24.0).reshape(2, 3, 4)
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "permuted":
        x = x.transpose(1, 2, 0)
    expected = np.add(x, 1, order=order)
    result = blosc2.evaluate("x + 1", local_dict={"x": x}, order=order)
    np.testing.assert_array_equal(result, expected)
    assert result.strides == expected.strides
    assert not np.shares_memory(result, x)


def test_default_order_preserves_fortran_layout():
    x = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    result = blosc2.evaluate("x + 1", local_dict={"x": x})
    assert result.flags.f_contiguous
    np.testing.assert_array_equal(result, x + 1)


@pytest.mark.parametrize("mode", ["full", "safe"])
@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
def test_layout_does_not_select_numexpr(mode, order, monkeypatch):
    import importlib

    module = importlib.import_module("blosc2.lazyexpr")
    x = np.asfortranarray(np.arange(24.0).reshape(2, 3, 4))

    def forbidden(*args, **kwargs):
        pytest.fail("Output order selected the NumExpr backend")

    if module.numexpr is not None:
        for name in ("evaluate", "validate", "NumExpr"):
            monkeypatch.setattr(module.numexpr, name, forbidden)
    with blosc2.expression_evaluation(mode):
        result = blosc2.evaluate("x + 1", local_dict={"x": x}, order=order)
    expected = np.add(x, 1, order=order)
    np.testing.assert_array_equal(result, expected)
    assert result.strides == expected.strides


def test_explicit_c_order_retains_native_admission(monkeypatch, native_graph_runtime):
    import blosc2.native_graph as native

    if blosc2.IS_WASM:
        pytest.skip("Native-first dispatch requires host JIT")
    probe = blosc2.NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True)
    if not probe.has_jit:
        pytest.skip("Native-first dispatch requires JIT")
    original = native.compute_native_graph
    calls = []

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "compute_native_graph", record)
    x = np.arange(24.0).reshape(2, 3, 4)
    result = blosc2.evaluate("x + 1", local_dict={"x": x}, order="C")
    np.testing.assert_array_equal(result, x + 1)
    assert calls


@pytest.mark.parametrize("order", ["C", "F", "A", "K"])
def test_broadcast_output_order(order):
    x = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    y = np.arange(3.0).reshape(3, 1)
    expected = np.add(x, y, order=order)
    result = blosc2.evaluate("x + y", local_dict={"x": x, "y": y}, order=order)
    np.testing.assert_array_equal(result, expected)
    assert result.strides == expected.strides


def test_fortran_reduction_and_explicit_destination():
    x = np.arange(24.0).reshape(2, 3, 4)
    result = blosc2.evaluate("sum(x, axis=0)", local_dict={"x": x}, order="F")
    assert result.flags.f_contiguous
    np.testing.assert_array_equal(result, x.sum(axis=0))
    output = np.empty_like(x, order="C")
    assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=output, order="F") is output
    assert output.flags.c_contiguous
    np.testing.assert_array_equal(output, x + 1)


def test_order_with_nonbroadcastable_reduction_inputs():
    x = np.arange(12.0).reshape(3, 4)
    y = np.arange(10.0).reshape(2, 5)
    result = blosc2.evaluate("sum(x) + y", local_dict={"x": x, "y": y}, order="F")
    assert result.flags.f_contiguous
    np.testing.assert_array_equal(result, x.sum() + y)


@pytest.mark.parametrize("mode", ["full", "safe"])
@pytest.mark.parametrize("option", ["truediv", "sanitize", "disable_cache"])
def test_removed_options_rejected_before_preparation(mode, option, monkeypatch):
    import importlib

    module = importlib.import_module("blosc2.lazyexpr")

    def prepare(*args, **kwargs):
        pytest.fail("Removed options must fail before preparing an expression")

    monkeypatch.setattr(module, "lazyexpr", prepare)
    x = np.arange(17.0)
    output = np.full_like(x, -999)
    with blosc2.expression_evaluation(mode):
        with pytest.raises(ValueError, match=option):
            blosc2.evaluate("x + 1", local_dict={"x": x}, out=output, **{option: True})
    np.testing.assert_array_equal(output, -999)


@pytest.mark.parametrize("order", ["invalid", "", None, 1])
def test_invalid_output_order(order):
    x = np.arange(12.0)
    with pytest.raises(ValueError, match="order"):
        blosc2.evaluate("x + 1", local_dict={"x": x}, order=order)


@pytest.fixture
def numexpr():
    return pytest.importorskip("numexpr")


@pytest.mark.parametrize("mode", ["full", "safe"])
@pytest.mark.parametrize(
    "expression",
    [
        "copy(x)",
        "fmod(x, y)",
        "where(x > 0.3, x, y)",
        "complex(x, y)",
        "sum(x)",
        "prod(x)",
        "min(x)",
        "max(x)",
        "3.0",
        "sin(0.5)",
    ],
)
def test_numexpr_expression_forms(numexpr, mode, expression):
    inputs = {"x": np.linspace(0.1, 0.6, 17), "y": np.linspace(1.1, 1.6, 17)}
    expected = numexpr.evaluate(expression, local_dict=inputs)
    output = np.empty_like(expected)
    with blosc2.expression_evaluation(mode):
        result = blosc2.evaluate(expression, local_dict=inputs, out=output)
    assert result is output
    np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("casting", ["no", "safe", "same_kind", "unsafe"])
def test_numexpr_output_casting(numexpr, casting):
    inputs = {"x": np.arange(17.0), "y": np.ones(17)}
    output = np.full(17, -999, dtype="float32")
    if casting in {"no", "safe"}:
        with pytest.raises(TypeError):
            numexpr.evaluate("x + y", local_dict=inputs, out=output, casting=casting)
        with pytest.raises(TypeError):
            blosc2.evaluate("x + y", local_dict=inputs, out=output, casting=casting)
        np.testing.assert_array_equal(output, -999)
    else:
        assert blosc2.evaluate("x + y", local_dict=inputs, out=output, casting=casting) is output
        np.testing.assert_array_equal(output, inputs["x"] + inputs["y"])


def test_complex_constructor_preserves_infinite_components():
    real = np.array([1.0, np.inf, -np.inf])
    imag = np.array([np.inf, 2.0, -np.inf])
    output = np.empty(3, dtype="complex128")
    with np.errstate(all="raise"):
        result = blosc2.evaluate("complex(x, y)", local_dict={"x": real, "y": imag}, out=output)
    assert result is output
    np.testing.assert_array_equal(result.real, real)
    np.testing.assert_array_equal(result.imag, imag)


def test_output_cache_does_not_retain_destination(monkeypatch):
    import gc
    import importlib
    import weakref

    module = importlib.import_module("blosc2.lazyexpr")
    module._evaluate_recipe_cache.clear()
    x = np.arange(17.0)
    output = np.empty_like(x)
    reference = weakref.ref(output)
    assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=output) is output
    assert module._evaluate_recipe_cache
    del output
    gc.collect()
    assert reference() is None

    def forbidden(*args, **kwargs):
        pytest.fail("Cached explicit-output evaluation rebuilt its LazyExpr")

    monkeypatch.setattr(blosc2.LazyExpr, "_new_expr", forbidden)
    rebound = np.arange(17.0) + 10
    output = np.empty_like(rebound)
    assert blosc2.evaluate("x + 1", local_dict={"x": rebound}, out=output) is output
    np.testing.assert_array_equal(output, rebound + 1)


@pytest.mark.parametrize("expression", ["sqrt(x * x)", "hypot(x, y)", "where(x > y, x, y)"])
def test_root_math_call_is_not_a_native_reduction(expression):
    from blosc2.native_graph import _native_text_expression

    source, reduction = _native_text_expression(expression)
    assert source
    assert not reduction


@pytest.mark.skipif(blosc2.IS_WASM, reason="Host thread pools are unavailable on WASM")
def test_numpy_fallback_rebinds_without_retaining_arrays():
    import gc
    import importlib
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    module = importlib.import_module("blosc2.lazyexpr")

    def run(index):
        x = np.full(257, index + 0.5)
        y = np.full(257, index + 1.5)
        output = np.empty(257, dtype="complex128")
        refs = [weakref.ref(value) for value in (x, y, output)]
        result = blosc2.evaluate("complex(x, y)", local_dict={"x": x, "y": y}, out=output)
        assert result is output
        np.testing.assert_array_equal(result.real, x)
        np.testing.assert_array_equal(result.imag, y)
        return refs

    with ThreadPoolExecutor(max_workers=4) as pool:
        references = list(pool.map(run, range(12)))
    gc.collect()
    assert module._evaluate_recipe_cache
    assert all(reference() is None for group in references for reference in group)


def test_readonly_output_is_rejected_before_writes():
    x = np.arange(17.0)
    output = np.full_like(x, -999)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        blosc2.evaluate("x + 1", local_dict={"x": x}, out=output)
    np.testing.assert_array_equal(output, -999)
