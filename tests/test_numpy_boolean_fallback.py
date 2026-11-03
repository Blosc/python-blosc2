"""Array-aware fallback boolean operations retain Python scalar control flow."""

import importlib

import numpy as np
import pytest

import blosc2
from blosc2.expression_graph import expression_evaluation


@pytest.fixture(params=["legacy", "typed"], autouse=True)
def evaluator_mode(monkeypatch, request):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", request.param)


@pytest.mark.parametrize("evaluation", ["full", "safe"])
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("True or (a + 256)", True),
        ("False and (a + 256)", False),
        ("2 or a", 2),
        ("0 and a", 0),
    ],
)
def test_scalar_boolean_short_circuit(expression, expected, evaluation):
    module = importlib.import_module("blosc2.lazyexpr")
    a = np.arange(4, dtype=np.uint8)
    with expression_evaluation(evaluation):
        result = module.evaluate_chunk(expression, {"a": a})
        public_result = blosc2.evaluate(expression, local_dict={"a": a})
    assert result.shape == ()
    assert result.item() == expected
    np.testing.assert_array_equal(public_result, np.full(public_result.shape, expected))


@pytest.mark.parametrize("evaluation", ["full", "safe"])
@pytest.mark.parametrize("expression", ["0 or a", "2 and a"])
def test_scalar_boolean_returns_array_operand(expression, evaluation):
    module = importlib.import_module("blosc2.lazyexpr")
    a = np.arange(4, dtype=np.int32)
    with expression_evaluation(evaluation):
        result = module.evaluate_chunk(expression, {"a": a})
    assert result.dtype == a.dtype
    np.testing.assert_array_equal(result, a)


@pytest.mark.parametrize("evaluation", ["full", "safe"])
@pytest.mark.parametrize("expression", ["False or (a + 256)", "True and (a + 256)"])
def test_scalar_boolean_evaluates_participating_branch(expression, evaluation):
    if np.__version__.startswith("1."):
        pytest.skip("NumPy 1 promotes this weak scalar instead of reporting overflow")
    module = importlib.import_module("blosc2.lazyexpr")
    with expression_evaluation(evaluation), pytest.raises(OverflowError):
        module.evaluate_chunk(expression, {"a": np.arange(4, dtype=np.uint8)})


@pytest.mark.parametrize("evaluation", ["full", "safe"])
@pytest.mark.parametrize("expression", ["a and logical_and", "a or logical_and", "not a"])
def test_array_boolean_operations_do_not_shadow_operands(expression, evaluation):
    module = importlib.import_module("blosc2.lazyexpr")
    a = np.array([True, True, False, False])
    other = np.array([True, False, True, False])
    expected = (
        np.logical_and(a, other)
        if " and " in expression
        else np.logical_or(a, other)
        if " or " in expression
        else np.logical_not(a)
    )
    with expression_evaluation(evaluation):
        result = module.evaluate_chunk(expression, {"a": a, "logical_and": other})
    np.testing.assert_array_equal(result, expected)


def test_zero_dimensional_boolean_short_circuit():
    module = importlib.import_module("blosc2.lazyexpr")
    a = np.arange(4, dtype=np.uint8)
    result = module.evaluate_chunk("left or (a + 256)", {"left": np.array(True), "a": a})
    assert result.shape == ()
    assert result.item() is True


def test_trusted_scalar_short_circuit_does_not_resolve_unvisited_name():
    module = importlib.import_module("blosc2.lazyexpr")
    with expression_evaluation("full"):
        result = module.evaluate_chunk("True or unbound_name", {})
    assert result.shape == ()
    assert result.item() is True
