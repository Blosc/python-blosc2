"""Public reduction results retain array semantics, including full reductions."""

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    "op", ["sum", "prod", "min", "max", "mean", "std", "var", "any", "all", "argmin", "argmax"]
)
@pytest.mark.parametrize("frontend", ["function", "method", "expression"])
@pytest.mark.parametrize(("axis", "keepdims"), [(None, False), (None, True), (0, False), (1, True)])
def test_reduction_array_result(op, frontend, axis, keepdims):
    values = np.linspace(0.5, 1.5, 24).reshape(4, 6)
    if op in ("any", "all"):
        values = values > 1
    arr = blosc2.asarray(values, chunks=(2, 3), blocks=(1, 3))
    operand = blosc2.lazyexpr("a", {"a": arr}) if frontend == "expression" else arr
    if frontend == "function":
        result = getattr(blosc2, op)(operand, axis=axis, keepdims=keepdims)
    else:
        result = getattr(operand, op)(axis=axis, keepdims=keepdims)
    expected = getattr(np, op)(values, axis=axis, keepdims=keepdims)
    assert not np.isscalar(result)
    assert result.shape == np.shape(expected)
    assert result.dtype == np.asarray(expected).dtype
    np.testing.assert_allclose(np.asarray(result), expected)


@pytest.mark.parametrize("op", ["sum", "prod", "any", "all"])
def test_empty_reduction_array_result(op):
    dtype = np.bool_ if op in ("any", "all") else np.float64
    values = np.empty((0,), dtype=dtype)
    result = getattr(blosc2.asarray(values), op)()
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    np.testing.assert_equal(result, getattr(np, op)(values))


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "mean", "std", "var"])
def test_masked_reduction_array_result(op):
    values = np.linspace(0.5, 1.5, 24)
    arr = blosc2.asarray(values, chunks=(6,), blocks=(3,))
    mask = blosc2.asarray(values > 1)
    result = getattr(arr, op)(where=mask)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    np.testing.assert_allclose(result, getattr(np, op)(values[values > 1]))


def test_count_nonzero_array_result():
    arr = blosc2.asarray([0, 1, 2])
    result = blosc2.count_nonzero(arr)
    assert isinstance(result, np.ndarray)
    assert result.shape == ()
    assert result[()] == 2


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "mean", "std", "var"])
@pytest.mark.parametrize("axis", [None, (0, 1), ()])
def test_interpreter_reduction_array_result(op, axis):
    values = np.linspace(0.9, 1.1, 120).reshape(10, 12)
    arr = blosc2.asarray(values, chunks=(5, 6), blocks=(1, 3))
    with blosc2.jit_options(jit=False):
        result = getattr(arr, op)(axis=axis)
    expected = getattr(np, op)(values, axis=axis)
    assert not np.isscalar(result)
    assert result.shape == np.shape(expected)
    np.testing.assert_allclose(result, expected)
