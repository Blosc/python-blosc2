"""NumExpr API compatibility for resident inputs and an explicit NumPy output.

Numerical/backend cross-products belong in miniexpr. These representative
cases exercise missing expression forms, shape contracts and output policies.
"""

import numpy as np
import pytest

import blosc2


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


def test_resident_compatibility_program_rebinds_without_retaining_arrays(numexpr):
    import gc
    import importlib
    import weakref
    from concurrent.futures import ThreadPoolExecutor

    module = importlib.import_module("blosc2.lazyexpr")
    module._resident_numexpr_program.cache_clear()

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
    assert module._resident_numexpr_program.cache_info().currsize
    assert all(reference() is None for group in references for reference in group)


def test_readonly_output_is_rejected_before_writes():
    x = np.arange(17.0)
    output = np.full_like(x, -999)
    output.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        blosc2.evaluate("x + 1", local_dict={"x": x}, out=output)
    np.testing.assert_array_equal(output, -999)
