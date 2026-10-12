import gc

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def runtime():
    if not hasattr(blosc2.blosc2_ext.NativeGraphHandle, "resident_artifact"):
        pytest.skip("Requires borrowed graph artifact binding")
    previous = blosc2.nthreads
    blosc2.set_nthreads(4)
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("expression", ["x + (y - 1)", "x > y", "x * 0.5 + y"])
def test_native_graph_parallel(dtype, expression):
    x = np.linspace(0, 1, 600_003, dtype=dtype)
    y = np.linspace(1, 2, 600_003, dtype=dtype)
    expr = blosc2.lazyexpr(expression, {"x": x, "y": y})
    expected = expr.compute(_require_native=True, jit=True, _native_tile_items=1023, _getitem=True)
    np.testing.assert_array_equal(expr[:], expected)
    report = expr._native_execution_report
    assert report["route"] == "parallel-resident"
    assert report["backend"] == "portable-jit"
    assert report["threads"] == 4
    assert report["input_materialization_bytes"] == 0
    assert report["normalization_bytes"] == 0


@pytest.mark.parametrize(
    "case", ["strided", "broadcast", "partial", "serial", "tile-override", "typed-capture"]
)
def test_original_graph_execution_retained(case):
    x = np.linspace(0, 1, 600_003)
    y = x[::-1] if case == "strided" else x[:1] if case == "broadcast" else x
    expr = blosc2.lazyexpr(
        "x + (y - c)", {"x": x, "y": y, "c": np.float64(1) if case == "typed-capture" else 1}
    )
    kwargs = {"_require_native": True, "jit": True, "_getitem": True}
    item = (slice(0, 400_000),) if case == "partial" else ()
    if case == "serial":
        blosc2.set_nthreads(1)
    if case == "tile-override":
        kwargs["_native_tile_items"] = 64
    result = expr.compute(item=item, **kwargs)
    np.testing.assert_array_equal(result, (x + (y - 1))[item])
    assert expr._native_execution_report.get("route") != "parallel-resident"


def test_reduction_and_staged_graph_retained():
    x = np.ones(600_003)
    for expression, expected in (("sum(x)", float(x.size)), ("x + sum(x)", x + x.size)):
        expr = blosc2.lazyexpr(expression, {"x": x})
        np.testing.assert_array_equal(expr.compute(_require_native=True, jit=True, _getitem=True), expected)
        assert expr._native_execution_report.get("route") != "parallel-resident"


def test_graph_fp_status_matches_original():
    x = np.full(600_003, np.finfo(np.float64).max)
    expr = blosc2.lazyexpr("x * x", {"x": x})
    expected = expr.compute(_require_native=True, jit=True, _native_tile_items=1023, _getitem=True)
    status = expr._native_execution_report["fp_flags"]
    np.testing.assert_array_equal(expr[:], expected)
    assert expr._native_execution_report["fp_flags"] == status


def test_lazy_map_participation_mask_preserved():
    x = np.linspace(-1, 1, 600_003)
    y = np.ones_like(x)
    # Avoid the existing where-construction path, which requires a Blosc2 condition.
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
    names = tuple(expr.operands)
    expr.expression = f"where({names[0]} > 0, sqrt({names[0]}), {names[1]})"
    expected = expr.compute(_require_native=True, jit=True, _native_tile_items=1023, _getitem=True)
    result = expr.compute(_require_native=True, jit=True, _getitem=True)
    np.testing.assert_array_equal(result, expected)
    assert expr._native_execution_report["route"] == "parallel-resident"
    assert expr._native_execution_report["fp_flags"] == 0


def test_graph_execution_failure_never_replays(monkeypatch):
    def fail(*args, **kwargs):
        raise ArithmeticError("Parallel graph failed")

    monkeypatch.setattr("blosc2.resident_execution.evaluate_resident", fail)
    x = np.ones(600_003)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    with pytest.raises(ArithmeticError, match="Parallel graph failed"):
        expr[:]


def test_borrowed_artifact_retains_native_owner():
    from blosc2.graph_plan import NativeGraph

    plan = NativeGraph.from_expression("x + 1", {"x": np.dtype("float64")}, jit=True)
    borrowed = plan._handle.resident_artifact()
    assert borrowed is not None
    del plan
    gc.collect()
    values = np.arange(100.0)
    result = borrowed.evaluate_block({"x": values}, values.shape)
    np.testing.assert_array_equal(result, values + 1)
    del borrowed
    gc.collect()
