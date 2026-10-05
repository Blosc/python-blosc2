"""Comparison-chain semantics across lowering, interpreter and native backends."""

import ast

import numpy as np
import pytest

import blosc2
from blosc2.dsl_compare import lower_chained_comparisons
from blosc2.dsl_kernel import kernel_from_source


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize(
    "expression",
    [
        "0 <= x < 5",
        "5 > x >= 0",
        "x == 2 != 3",
        "x != 2 == 2",
        "-2 < x <= 4 != 3",
        "(0 < x < 5) + (2 < x < 7)",
        "not (0 < x < 5)",
        "(x < 0) or (0 < x < 5)",
        "(x > 0) and (0 < x < 5)",
        "where(0 < x < 5, 1, 2)",
    ],
)
def test_chained_comparisons(expression, jit):
    source = f"def k(x):\n    return {expression}\n"
    kernel = kernel_from_source(source, "k")
    assert "b2_chain_" not in kernel.dsl_source
    assert expression in kernel.dsl_source
    assert blosc2.validate_dsl(kernel)["valid"]
    x = np.arange(-3, 8, dtype=np.float64)
    namespace = {"where": np.where}
    exec(source, namespace)
    expected = np.array([namespace["k"](float(v)) for v in x], dtype=np.float64)
    result = blosc2.lazyudf(kernel, (x,), dtype=np.float64, jit=jit)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize("jit", [False, True])
def test_chain_short_circuit_division(jit):
    kernel = kernel_from_source("def k(x):\n    return 0 < x < 12 / x\n", "k")
    x = np.arange(-3, 8, dtype=np.int64)
    expected = np.array([0 < int(v) < 12 / int(v) for v in x])
    result = blosc2.lazyudf(kernel, (x,), dtype=np.bool_, jit=jit)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize("jit", [False, True])
def test_chain_control_flow_and_continue(jit):
    source = (
        "def k(x):\n"
        "    y = x\n"
        "    while 0 <= y < 3:\n"
        "        y += 1\n"
        "        continue\n"
        "    if 3 <= y < 5:\n"
        "        y += 10\n"
        "    elif -5 <= y < 0:\n"
        "        y -= 10\n"
        "    else:\n"
        "        y += 0 < y < 10\n"
        "    return y\n"
    )
    kernel = kernel_from_source(source, "k")
    namespace = {}
    exec(source, namespace)
    x = np.arange(-3, 8, dtype=np.float64)
    expected = np.array([namespace["k"](float(v)) for v in x])
    result = blosc2.lazyudf(kernel, (x,), dtype=np.float64, jit=jit)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize(
    ("values", "expected_calls"),
    [
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        ([2, 1, 2, 3], [2, 1]),
        ([0, 2, 1, 3], [0, 2, 1]),
    ],
)
def test_lowering_evaluates_operands_once(values, expected_calls):
    func = ast.parse("def k():\n    return probe(0) < probe(1) < probe(2) < probe(3)\n").body[0]
    lowered = lower_chained_comparisons(func)
    calls = []

    def probe(index):
        calls.append(values[index])
        return values[index]

    namespace = {"probe": probe}
    exec(compile(ast.Module(body=[lowered], type_ignores=[]), "<chain>", "exec"), namespace)
    assert namespace["k"]() == (values[0] < values[1] < values[2] < values[3])
    assert calls == expected_calls


def test_temporary_names_do_not_collide():
    kernel = kernel_from_source(
        "def k(b2_chain_0):\n    b2_chain_1 = 7\n    return 0 < b2_chain_0 < b2_chain_1\n", "k"
    )
    x = np.arange(-1, 10, dtype=np.float64)
    np.testing.assert_array_equal(blosc2.lazyudf(kernel, (x,), dtype=np.bool_)[:], (x > 0) & (x < 7))


def test_chain_actual_native_jit():
    baseline = kernel_from_source("def k(x):\n    return x + 1\n", "k")
    available = blosc2.validate_dsl_jit(baseline, [np.float64], np.float64)
    kernel = kernel_from_source("def k(x):\n    return 0 < x < 5\n", "k")
    status = blosc2.validate_dsl_jit(kernel, [np.float64], np.float64)
    assert status["compiled"]
    if available["jit"]:
        assert status["jit"]


@pytest.mark.parametrize("jit", [False, True])
def test_chain_boolean_output_preserves_float_operands(jit):
    kernel = kernel_from_source("def k(x):\n    return 0.25 < x < 0.75\n", "k")
    x = np.array([np.nan, -np.inf, 0, 0.25, 0.5, 0.75, 1, np.inf])
    result = blosc2.lazyudf(kernel, (x,), dtype=np.bool_, jit=jit)
    np.testing.assert_array_equal(result[:], (x > 0.25) & (x < 0.75))


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("wasm_dispatch", [False, True])
def test_chain_float_output_preserves_large_integer_operands(jit, wasm_dispatch, monkeypatch):
    if wasm_dispatch:
        monkeypatch.setattr(blosc2, "IS_WASM", True)
    kernel = kernel_from_source("def k(x, lo, hi):\n    return lo < x < hi\n", "k")
    x = np.arange(6, dtype=np.int64) + 2**54
    lo = np.full(6, 2**54 + 1, dtype=np.int64)
    hi = np.full(6, 2**54 + 4, dtype=np.int64)
    result = blosc2.lazyudf(kernel, (x, lo, hi), dtype=np.float64, jit=jit)
    np.testing.assert_array_equal(result[:], (lo < x) & (x < hi))


@pytest.mark.parametrize("jit", [False, True])
def test_chain_in_range_arguments(jit):
    source = "def k(x):\n    y = 0\n    for i in range(int(0 < x < 5), 3):\n        y += 1\n    return y\n"
    kernel = kernel_from_source(source, "k")
    x = np.arange(-1, 7, dtype=np.float64)
    expected = np.where((x > 0) & (x < 5), 2, 3)
    np.testing.assert_array_equal(blosc2.lazyudf(kernel, (x,), dtype=np.float64, jit=jit)[:], expected)


def test_chain_while_iteration_cap_counts_body_only(monkeypatch):
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    kernel = kernel_from_source(
        "def k(x):\n    y = 0\n    while 0 <= y < 3:\n        y += 1\n    return x + y\n", "k"
    )
    x = np.arange(4, dtype=np.float64)
    np.testing.assert_array_equal(blosc2.lazyudf(kernel, (x,), dtype=np.float64, jit=False)[:], x + 3)


@pytest.mark.parametrize("jit", [False, True])
def test_chain_string_operands(jit):
    kernel = kernel_from_source("def k(x):\n    return 'b' == x != 'a'\n", "k")
    x = np.array(["a", "b", "c", "d", "e"])
    expected = (x == "b") & (x != "a")
    np.testing.assert_array_equal(blosc2.lazyudf(kernel, (x,), dtype=np.bool_, jit=jit)[:], expected)


@pytest.mark.parametrize(
    ("expression", "expected_calls"),
    [
        ("probe(0) + (probe(1) < probe(2) < probe(3))", [0, 1, 2, 3]),
        ("pair(probe(0), probe(1) < probe(2) < probe(3))", [0, 1, 2, 3]),
        ("False and (probe(0) < probe(1) < probe(2))", []),
        ("True or (probe(0) < probe(1) < probe(2))", []),
        ("probe(0) < (probe(1) < probe(2) < probe(3)) < probe(4)", [0, 1, 2, 3, 4]),
    ],
)
def test_lowering_nested_evaluation_order(expression, expected_calls):
    func = ast.parse(f"def k():\n    return {expression}\n").body[0]
    calls = []

    def probe(index):
        calls.append(index)
        return index

    namespace = {"probe": probe, "pair": lambda a, b: a + b}
    exec(
        compile(ast.Module(body=[lower_chained_comparisons(func)], type_ignores=[]), "<chain>", "exec"),
        namespace,
    )
    namespace["k"]()
    assert calls == expected_calls
