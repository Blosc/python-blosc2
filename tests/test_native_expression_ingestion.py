"""Automatic lowering uses native text only for a proven shared grammar subset."""

import numpy as np
import pytest

import blosc2
import blosc2.native_graph as native
from blosc2.graph_plan import NativeGraph


@pytest.mark.parametrize("dtype", ["int8", "uint64", "float32", "float64", "bool"])
@pytest.mark.parametrize("backend", ["tcc", "cc"])
def test_native_expression_ingestion_bypasses_adapter(native_graph_runtime, monkeypatch, dtype, backend):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    values = np.arange(12).reshape(3, 4).astype(dtype)
    y = np.ones((1, 4), dtype=dtype)
    source = "x == y" if dtype == "bool" else "x + (y - 1)"
    expr = blosc2.lazyexpr(source, {"x": values, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible native text expression constructed Python graph nodes")

    monkeypatch.setattr(native.GraphAdapter, "visit", forbidden)
    plan, arrays, reduction = native.lower_native_graph(expr, jit=True)
    assert plan._frontend == "native-expression"
    assert not reduction
    expected = values == y if dtype == "bool" else values + (y - 1)
    np.testing.assert_array_equal(plan.evaluate(arrays), expected)


@pytest.mark.parametrize(
    "source",
    [
        "x + 0x10",
        "x + 1_000",
        "x + (-0.0)",
        "x * -2 + 1",
        "x & 3 == 1",
        "sum(x + 1)",
        "sum(x + 1, axis=0, keepdims=True)",
        "max(x, initial=-1)",
    ],
)
def test_text_and_json_frontend_parity(native_graph_runtime, monkeypatch, source):
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    expr = blosc2.LazyExpr._new_expr(source, {"x": values}, guess=False)
    text, inputs, reduction = native.lower_native_graph(expr)
    assert text._frontend == "native-expression"
    monkeypatch.setattr(native, "_native_text_expression", lambda expression: None)
    declarative, old_inputs, old_reduction = native.lower_native_graph(expr)
    assert declarative._frontend == "declarative-json"
    assert reduction == old_reduction
    expected, old_report = declarative.evaluate(old_inputs, return_report=True)
    result, report = text.evaluate(inputs, return_report=True)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)
    assert report["fp_flags"] == old_report["fp_flags"]


@pytest.mark.parametrize("capture", [1, np.int32(1), np.float32(1)])
def test_scalar_captures_keep_declarative_frontend(native_graph_runtime, capture):
    expr = blosc2.lazyexpr("x + c", {"x": np.arange(6, dtype=np.float32), "c": capture})
    plan, arrays, reduction = native.lower_native_graph(expr)
    assert plan._frontend == "declarative-json"
    assert not reduction
    np.testing.assert_array_equal(plan.evaluate(arrays), np.arange(6, dtype=np.float32) + capture)


@pytest.mark.parametrize("source", ["x.sum()", "sum(x, dtype='int64')", "np.sum(x)", "sum(x) + x"])
def test_unqualified_syntax_keeps_declarative_frontend(native_graph_runtime, source):
    expr = blosc2.LazyExpr._new_expr(source, {"x": np.arange(6, dtype=np.int32)}, guess=False)
    plan, _, _ = native.lower_native_graph(expr)
    assert plan._frontend == "declarative-json"


def test_method_options_keep_declarative_frontend(native_graph_runtime):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(6, dtype=np.float32)})
    plan, arrays, reduction = native.lower_native_graph(
        expr, reduction_options={"reduction": "sum", "axis": 0}
    )
    assert plan._frontend == "declarative-json"
    assert reduction
    assert plan.evaluate(arrays) == 21


def test_automatic_execution_reports_native_frontend(native_graph_runtime):
    values = np.arange(100_000, dtype=np.float64)
    expr = blosc2.lazyexpr("x + 1", {"x": values})
    prepared = native.prepare_native_first(expr, ())
    if prepared is None:
        pytest.skip("Automatic native execution requires graph JIT")
    np.testing.assert_array_equal(expr[:], values + 1)
    assert expr._native_execution_report["frontend"] == "native-expression"


def test_native_text_ignores_unused_capture(native_graph_runtime):
    values = np.arange(6, dtype=np.float32)
    expr = blosc2.LazyExpr._new_expr("x + 1", {"x": values, "unused": np.int32(3)}, guess=False)
    plan, arrays, _ = native.lower_native_graph(expr)
    assert plan._frontend == "native-expression"
    assert tuple(arrays) == ("x",)
    np.testing.assert_array_equal(plan.evaluate(arrays), values + 1)


@pytest.mark.parametrize("name", ["true", "false", "null"])
def test_native_lexer_reserved_names_keep_adapter(native_graph_runtime, name):
    values = np.arange(6, dtype=np.float32)
    expr = blosc2.LazyExpr._new_expr(f"{name} + 1", {name: values}, guess=False)
    plan, _, _ = native.lower_native_graph(expr)
    assert plan._frontend == "declarative-json"


def test_native_text_rejects_unicode_binding_syntax():
    assert native._native_text_expression("α + 1") is None


def test_native_text_cache_uses_dtype_and_jit_configuration(native_graph_runtime, monkeypatch):
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(6, dtype=np.float32)})
    first, _, _ = native.lower_native_graph(expr)
    name = next(iter(expr.operands))
    expr.operands[name][:] = 0
    second, arrays, _ = native.lower_native_graph(expr)
    assert second is first
    np.testing.assert_array_equal(second.evaluate(arrays), 1)
    expr.operands[name] = np.arange(6, dtype=np.float64)
    different, _, _ = native.lower_native_graph(expr)
    assert different is not first
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc")
    cc, _, _ = native.lower_native_graph(expr, jit=True)
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "tcc")
    tcc, _, _ = native.lower_native_graph(expr, jit=True)
    assert tcc is not cc


def test_native_text_preparation_errors_are_not_replayed(native_graph_runtime, monkeypatch):
    native.compile_plan.cache_clear()
    expr = blosc2.lazyexpr("x + 1", {"x": np.arange(6, dtype=np.float32)})

    def failure(*args, **kwargs):
        raise RuntimeError("native text preparation failed")

    def forbidden(*args, **kwargs):
        pytest.fail("Unexpected JSON replay after native text preparation failed")

    monkeypatch.setattr(NativeGraph, "from_expression", failure)
    monkeypatch.setattr(native.GraphAdapter, "visit", forbidden)
    with pytest.raises(RuntimeError, match="native text preparation failed"):
        native.lower_native_graph(expr)
