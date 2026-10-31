"""Eligible LazyExpr execution follows the requested output storage."""

import importlib

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph


@pytest.fixture(autouse=True)
def prefilter_runtime(native_graph_runtime):
    if blosc2.IS_WASM or not blosc2.blosc2_ext.menudet_prefilter_available():
        pytest.skip("Native compressed prefilters require a host runtime")
    if not NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True).has_jit:
        pytest.skip("Automatic output dispatch requires a native JIT")


@pytest.fixture
def parallel_threads():
    # Other modules can leave the process-wide codec thread count at one.
    # A test requiring the parallel map must establish and restore its policy.
    previous = blosc2.set_nthreads(2)
    try:
        yield
    finally:
        blosc2.set_nthreads(previous)


def operands(size=257, dtype="float64"):
    x = np.linspace(0, 1, size, dtype=dtype)
    y = np.linspace(1, 2, size, dtype=dtype)
    return {
        "x": blosc2.asarray(x, chunks=(128,), blocks=(32,)),
        "y": blosc2.asarray(y, chunks=(128,), blocks=(32,)),
    }, x + (y - 1)


def forbid_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Native output dispatch used a numerical fallback")

    monkeypatch.setattr(importlib.import_module("blosc2.lazyexpr"), "evaluate_chunk", forbidden)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    return forbidden


@pytest.mark.parametrize("mode", ["full", "safe"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("explicit_grid", [False, True])
def test_blosc2_output_native_prefilter(monkeypatch, mode, dtype, explicit_grid):
    inputs, expected = operands(dtype=dtype)
    with blosc2.expression_evaluation(mode):
        expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    forbidden = forbid_fallback(monkeypatch)
    options = {"cparams": {"clevel": 0}}
    if explicit_grid:
        options.update(chunks=(128,), blocks=(32,))
    with monkeypatch.context() as patch:
        patch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        first = expr.compute(**options)
        patch.setattr(NativeGraph, "from_expression", forbidden)
        patch.setattr(NativeGraph, "__init__", forbidden)
        second = expr.compute(**options)
    assert first is not second
    assert first.chunks == (128,)
    assert first.blocks == (32,)
    assert first.cparams.clevel == 0
    np.testing.assert_array_equal(first[:], expected)
    report = expr._native_execution_report
    assert report["route"] == "menudet-prefilter"
    assert report["input_materialization_bytes"] == 0
    assert report["output_copy_bytes"] == 0


@pytest.mark.parametrize("bound", [False, True])
def test_matching_blosc2_out_and_evaluate(monkeypatch, bound):
    inputs, expected = operands()
    output = blosc2.uninit(expected.shape, chunks=(128,), blocks=(32,), cparams={"clevel": 0})
    expr = blosc2.lazyexpr("x + (y - 1)", inputs, **({"out": output} if bound else {}))
    forbid_fallback(monkeypatch)
    result = expr.compute(**({} if bound else {"out": output}))
    assert result is output
    assert expr._native_execution_report["route"] == "menudet-prefilter"
    np.testing.assert_array_equal(result[:], expected)
    assert blosc2.evaluate("x + (y - 1)", local_dict=inputs, out=output) is output


@pytest.mark.parametrize("explicit_out", [False, True])
@pytest.mark.parametrize("bound", [False, True])
def test_numpy_output_native_parallel_map(monkeypatch, parallel_threads, explicit_out, bound):
    from blosc2 import native_graph, resident_execution

    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_MAP_ITEMS", 1)
    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    monkeypatch.setattr(resident_execution, "MINIMUM_WORKER_ITEMS", 16384)
    inputs, expected = operands(size=200003)
    output = np.empty_like(expected)
    expr = blosc2.lazyexpr("x + (y - 1)", inputs, **({"out": output} if explicit_out and bound else {}))
    forbid_fallback(monkeypatch)
    if explicit_out:
        result = expr.compute(**({} if bound else {"out": output}))
        assert result is output
    else:
        result = expr[:]
        assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)
    assert expr._native_execution_report["route"] == "menudet-prefilter"


@pytest.mark.parametrize("clevel", [0, 1])
@pytest.mark.parametrize("explicit_out", [False, True])
def test_resident_inputs_use_stored_prefilter(monkeypatch, parallel_threads, clevel, explicit_out):
    bounds = np.iinfo("int64")
    values = np.resize(np.array([bounds.min, bounds.max, 0, 1], dtype="int64"), 1027)
    other = values[::-1].copy()
    expected = values + (other - 1)
    inputs = {"x": values, "y": other}
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    options = {"chunks": (256,), "blocks": (64,), "cparams": {"clevel": clevel, "nthreads": 2}}
    output = blosc2.uninit(values.shape, dtype=values.dtype, **options) if explicit_out else None
    forbid_fallback(monkeypatch)
    result = expr.compute(out=output) if explicit_out else expr.compute(**options)
    assert isinstance(result, blosc2.NDArray)
    if explicit_out:
        assert result is output
    assert result.chunks == (256,)
    assert result.blocks == (64,)
    assert result.cparams.clevel == clevel
    np.testing.assert_array_equal(result[:], expected)
    report = expr._native_execution_report
    assert report["route"] == "menudet-prefilter"
    assert report["input_buffer_allocations"] == report["native_decoder_contexts"] == 0
    assert report["input_materialization_bytes"] == report["output_copy_bytes"] == 0
    tail = np.frombuffer(result.schunk.decompress_chunk(result.schunk.nchunks - 1), dtype=values.dtype)
    np.testing.assert_array_equal(tail[3:], 0)


def test_resident_stored_prefilter_failure_is_not_replayed(monkeypatch):
    values = np.arange(257.0)
    expr = blosc2.lazyexpr("x + 1", {"x": values})
    forbid_fallback(monkeypatch)

    def failure(*args, **kwargs):
        raise OSError("Stored prefilter failed")

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", failure)
    with pytest.raises(OSError, match="Stored prefilter failed"):
        expr.compute()


def test_resident_stored_prefilter_options_and_rebinding(monkeypatch, tmp_path):
    values = np.arange(1027.0)
    expr = blosc2.lazyexpr("x + 1", {"x": values})
    params = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1, nthreads=2)
    path = tmp_path / "resident-prefilter.b2nd"
    forbid_fallback(monkeypatch)
    first = expr.compute(
        chunks=(256,), blocks=(64,), cparams=params, urlpath=path, meta={"sink": "resident"}
    )
    assert params.clevel == 1
    assert params.nthreads == 2
    assert first.schunk.meta["sink"] == "resident"
    np.testing.assert_array_equal(blosc2.open(path)[:], values + 1)
    values += 10
    second = expr.compute(chunks=(256,), blocks=(64,), cparams=params)
    np.testing.assert_array_equal(second[:], values + 1)
    np.testing.assert_array_equal(first[:], values - 9)


def test_resident_numpy_out_avoids_full_result_allocation(monkeypatch, parallel_threads):
    values = np.arange(300003, dtype="int64")
    inputs = {"x": values, "y": values[::-1].copy()}
    expected = values + (inputs["y"] - 1)
    output = np.empty_like(values)
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    expr.compute(out=output)  # Prepare artifacts and workers before instrumentation.
    original_empty = np.empty

    def bounded_empty(shape, *args, **kwargs):
        dtype = np.dtype(kwargs.get("dtype", args[0] if args else "float64"))
        if shape == values.shape and dtype.itemsize:
            pytest.fail("Explicit resident out allocated a full-size temporary")
        return original_empty(shape, *args, **kwargs)

    def forbidden_copy(*args, **kwargs):
        pytest.fail("Explicit resident out copied a full-size result")

    monkeypatch.setattr(np, "empty", bounded_empty)
    monkeypatch.setattr(np, "copyto", forbidden_copy)
    assert expr.compute(out=output) is output
    np.testing.assert_array_equal(output, expected)
    assert expr._native_execution_report["output_copy_bytes"] == 0
    assert blosc2.evaluate("x + (y - 1)", local_dict=inputs, out=output) is output
    np.testing.assert_array_equal(output, expected)


def test_native_prefilter_storage_options_and_rebinding(tmp_path):
    inputs, expected = operands()
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    path = tmp_path / "expr.b2nd"
    first = expr.compute(cparams={"clevel": 5}, urlpath=path, meta={"test": "native"})
    assert first.cparams.clevel == 5
    assert first.schunk.meta["test"] == "native"
    np.testing.assert_array_equal(blosc2.open(path)[:], expected)
    inputs["x"][:] = inputs["x"][:] + 10
    second = expr.compute(cparams={"clevel": 0})
    np.testing.assert_allclose(second[:], expected + 10, rtol=1e-15)
    np.testing.assert_array_equal(first[:], expected)


@pytest.mark.parametrize("partial", [False, True])
def test_other_grids_and_partial_selection_keep_fallback(partial):
    inputs, expected = operands()
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    item = slice(3, 250, 2) if partial else ()
    result = expr.compute(item, chunks=(64,), blocks=(16,), cparams={"clevel": 0})
    np.testing.assert_array_equal(result[:], expected[item] if partial else expected)
    assert result.chunks == (64,)
    assert not hasattr(expr, "_native_execution_report")


def test_native_prefilter_error_is_not_replayed(monkeypatch):
    inputs, _ = operands()
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    forbid_fallback(monkeypatch)
    calls = []

    def failing(*args, **kwargs):
        calls.append(True)
        raise RuntimeError("Injected native prefilter failure")

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", failing)
    with pytest.raises(RuntimeError, match="Injected native"):
        expr.compute(cparams={"clevel": 0})
    assert len(calls) == 1


def test_matching_grid_with_out_and_comparison_output(monkeypatch):
    inputs, _ = operands()
    expected = inputs["x"][:] > 0.5
    expr = blosc2.lazyexpr("x > 0.5", inputs)
    output = blosc2.uninit(expected.shape, dtype="bool", chunks=(128,), blocks=(32,), cparams={"clevel": 0})
    forbid_fallback(monkeypatch)
    assert expr.compute(out=output, chunks=(128,), blocks=(32,)) is output
    np.testing.assert_array_equal(output[:], expected)
    assert expr._native_execution_report["route"] == "menudet-prefilter"


def test_input_output_alias_retains_fallback():
    inputs, expected = operands()
    expr = blosc2.lazyexpr("x + (y - 1)", inputs)
    result = expr.compute(out=inputs["x"])
    assert result is inputs["x"]
    np.testing.assert_array_equal(result[:], expected)
    assert not hasattr(expr, "_native_execution_report")


def test_nonfinite_named_literals_retain_fallback():
    inputs, _ = operands()
    expr = (inputs["x"] + 1) == np.nan
    result = expr.compute(cparams={"clevel": 0})
    np.testing.assert_array_equal(result[:], False)
    assert not hasattr(expr, "_native_execution_report")
