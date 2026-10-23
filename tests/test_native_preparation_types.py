"""Shared native preparation binds numerical types without dummy-array evaluation."""

import importlib

import numpy as np
import pytest

import blosc2

lazy = importlib.import_module("blosc2.lazyexpr")
native = importlib.import_module("blosc2.native_graph")
NUMERIC_TYPES = [
    "bool",
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float32",
    "float64",
]


@pytest.fixture(autouse=True)
def runtime(native_graph_runtime):
    lazy._evaluate_recipe_cache.clear()
    yield
    lazy._evaluate_recipe_cache.clear()


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_numeric_resident_dispatch_and_output_type(dtype, monkeypatch):
    x = np.arange(17).astype(dtype)
    y = np.arange(1, 18).astype(dtype)
    expression = "x ^ y" if dtype == "bool" else "x + (y - 1)"
    expected = x ^ y if dtype == "bool" else x + (y - 1)

    def forbidden(*args, **kwargs):
        pytest.fail("Native preparation evaluated dummy operands or repeated syntax admission")

    with monkeypatch.context() as patch:
        patch.setattr(lazy, "_numpy_eval_expr", forbidden)
        patch.setattr(native, "_native_first_syntax", forbidden)
        expr = blosc2.lazyexpr(expression, {"x": x, "y": y})
        assert expr.dtype == expected.dtype
        assert hasattr(expr, "_resident_native")
        actual = blosc2.evaluate(expression, local_dict={"x": x, "y": y})
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(expr[:], expected)
    assert expr._native_execution_report["route"] == "native-resident"


@pytest.mark.parametrize("dtype", ["int64", "uint64", "float32", "float64"])
def test_compressed_metadata_plan_keeps_execution_admission(dtype, monkeypatch):
    x = np.arange(257).astype(dtype)
    y = x + 1
    inputs = {
        "x": blosc2.asarray(x, chunks=(128,), blocks=(32,)),
        "y": blosc2.asarray(y, chunks=(128,), blocks=(32,)),
    }

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata preparation read or evaluated operands")

    with monkeypatch.context() as patch:
        patch.setattr(lazy, "_numpy_eval_expr", forbidden)
        patch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    assert hasattr(expr, "_metadata_native")
    assert not hasattr(expr, "_prepared_native")
    assert expr.dtype == x.dtype
    expected = x + (y - 1)
    np.testing.assert_array_equal(expr[:], expected)
    np.testing.assert_array_equal(expr.compute()[:], expected)


@pytest.mark.parametrize("dtype", NUMERIC_TYPES)
def test_division_infers_output_separately_from_inputs(dtype):
    x = np.array([9, 12], dtype=dtype)
    y = np.array([2, 5], dtype=dtype)
    expected = x / y
    expr = blosc2.lazyexpr("x / y", {"x": x, "y": y})
    assert expr.dtype == expected.dtype
    assert hasattr(expr, "_resident_native")
    actual = blosc2.evaluate("x / y", local_dict={"x": x, "y": y})
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    out = np.empty_like(expected)
    assert blosc2.evaluate("x / y", local_dict={"x": x, "y": y}, out=out) is out
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize(("left", "right"), [("int8", "int64"), ("int64", "uint64"), ("int32", "float32")])
def test_mixed_input_types_keep_numpy_promotion(left, right):
    x, y = np.arange(17).astype(left), np.arange(17).astype(right)
    expected = x + y
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
    assert hasattr(expr, "_resident_native")
    assert expr.dtype == expected.dtype
    np.testing.assert_array_equal(expr[:], expected)
    actual = blosc2.evaluate("x + y", local_dict={"x": x, "y": y})
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["int64", "uint64"])
def test_integer_extrema_preserve_modular_arithmetic(dtype):
    limits = np.iinfo(dtype)
    x = np.array([limits.max, limits.min, limits.max - 1], dtype=dtype)
    y = np.array([1, 1, 2], dtype=dtype)
    actual = blosc2.evaluate("x + y", local_dict={"x": x, "y": y})
    assert actual.dtype == x.dtype
    np.testing.assert_array_equal(actual, x + y)


def test_same_width_dtype_change_invalidates_resident_schedule():
    x = np.arange(17, dtype="int64")
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    signed = expr[:]
    expr.operands["o0"] = np.array([2**63 + 7] * 17, dtype="uint64")
    unsigned = expr[:]
    assert signed.dtype == np.dtype("int64")
    assert unsigned.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(unsigned, expr.operands["o0"] + 1)


def test_integer_division_rejects_integer_destination_without_writing():
    x = np.arange(1, 18, dtype="int64")
    out = np.full_like(x, -1)
    with pytest.raises(TypeError, match="Cannot cast result"):
        blosc2.evaluate("x / 2", local_dict={"x": x}, out=out)
    np.testing.assert_array_equal(out, -1)


@pytest.mark.parametrize("expression", ["x @ y", "not x", "x[:2]", "x and y"])
def test_non_elementwise_syntax_keeps_legacy_preparation(expression):
    graph = importlib.import_module("blosc2.expression_graph").parse_expression(expression)
    assert not graph.arithmetic


@pytest.mark.parametrize(("dtype", "constant"), [("int8", 128), ("uint8", -1), ("uint64", -1)])
def test_out_of_range_weak_scalar_keeps_overflow_error(dtype, constant):
    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        pytest.skip("Menudet follows NumPy 2 weak-scalar overflow diagnostics, not NumPy 1 wrapping")
    x = np.arange(3, dtype=dtype)
    with pytest.raises(OverflowError):
        blosc2.evaluate(f"x + ({constant})", local_dict={"x": x})
