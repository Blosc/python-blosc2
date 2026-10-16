"""Public evaluate dispatch/output contracts, not native numerical cross-products."""

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("inputs", ["numpy", "blosc2", "mixed"])
@pytest.mark.parametrize("output", ["new", "numpy", "blosc2"])
@pytest.mark.parametrize("recipe", ["integer", "math"])
def test_evaluate_native_array_routes(inputs, output, recipe, native_graph_runtime, monkeypatch):
    import blosc2.native_graph as native

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_MAP_ITEMS", 1)
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 64)
    dtype = "int64" if recipe == "integer" else "float64"
    x = np.arange(1027, dtype=dtype) % 3 + 1
    y = np.ones_like(x)
    expression = "((x + y) * 2 - y) * 3 + x * 2 - y" if recipe == "integer" else "sin(x) + cos(y) + 1"
    expected = ((x + y) * 2 - y) * 3 + x * 2 - y if recipe == "integer" else np.sin(x) + np.cos(y) + 1
    operands = {"x": x, "y": y}
    for key in operands:
        if inputs == "blosc2" or (inputs == "mixed" and key == "x"):
            operands[key] = blosc2.asarray(operands[key], chunks=(512,), blocks=(128,))
    destination = None if output == "new" else np.empty_like(x)
    if output == "blosc2":
        destination = blosc2.empty(x.shape, dtype=x.dtype, chunks=(512,), blocks=(128,))

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible evaluate expression fell back to legacy numerical execution")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate(expression, local_dict=operands, out=destination)
    if destination is not None:
        assert result is destination
    np.testing.assert_allclose(
        result[:] if isinstance(result, blosc2.NDArray) else result, expected, rtol=2e-12, atol=2e-12
    )
    # Current operands must be rebound, not retained as cached values.
    operands["x"][:] = 2
    result = blosc2.evaluate(expression, local_dict=operands, out=destination)
    expected = ((2 + y) * 2 - y) * 3 + 4 - y if recipe == "integer" else np.sin(2) + np.cos(y) + 1
    np.testing.assert_allclose(
        result[:] if isinstance(result, blosc2.NDArray) else result, expected, rtol=2e-12, atol=2e-12
    )


def test_evaluate_stored_multidimensional_broadcast(native_graph_runtime, monkeypatch):
    x = np.arange(51, dtype="int64").reshape(3, 17)
    y = np.arange(17, dtype="int64").reshape(1, 17)
    stored = blosc2.asarray(x, chunks=(2, 8), blocks=(1, 4))
    output = blosc2.empty(x.shape, dtype="int64", chunks=(2, 8), blocks=(1, 4))

    def forbidden(*args, **kwargs):
        pytest.fail("Stored native map fell back to legacy")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate("x + y * 2", local_dict={"x": stored, "y": y}, out=output)
    assert result is output
    np.testing.assert_array_equal(output[:], x + y * 2)


@pytest.mark.parametrize("expression", ["x / y", "x // y", "x % y", "x ** y", "x << y", "x >> y"])
def test_evaluate_native_integer_operators(expression, native_graph_runtime, monkeypatch):
    x = np.array([-9, -1, 0, 1, 9], dtype="int64")
    y = np.full_like(x, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("Numeric operator did not reach native lowering")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate(expression, local_dict={"x": x, "y": y})
    np.testing.assert_array_equal(result, eval(expression, {"x": x, "y": y}))
