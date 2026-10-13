"""Parallel any/all evaluate all regions and preserve native status and failures."""

import numpy as np
import pytest

import blosc2
from blosc2.native_graph import prepare_native_first


@pytest.fixture(autouse=True)
def graph_truth_runtime(native_graph_runtime):
    expr = blosc2.lazyexpr("x > 0", {"x": np.ones(4)})
    if prepare_native_first(expr, ()) is None:
        pytest.skip("Parallel graph truth reductions require an available graph JIT")


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", ["bool", "int8", "int64", "uint64", "float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_parallel_resident_truth(op, dtype, backend, keepdims, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    values = (np.arange(4 * 128 * 1024) % 3).astype(dtype).reshape(4, -1)
    expr = blosc2.lazyexpr("x", {"x": values})
    previous = blosc2.set_nthreads(2)
    try:
        result = getattr(expr, op)(axis=(0, 1), keepdims=keepdims)
        expected = getattr(np, op)(values, axis=(0, 1), keepdims=keepdims)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.dtype("bool")
        assert result.shape == expected.shape
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
        assert expr._native_execution_report["threads"] == 2
        serial = getattr(expr, op)(axis=(0, 1), keepdims=keepdims, _require_native=True, jit=True)
        assert result.tobytes() == serial.tobytes()
        assert expr._native_execution_report.get("route") != "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
def test_parallel_truth_late_fp_status_is_not_short_circuited(op, dtype, backend, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    x = np.full(4 * 128 * 1024, 1 if op == "any" else 0, dtype=dtype)
    y = np.ones_like(x)
    # The first region decides the Boolean result, but the second still performs
    # inf*0. Its invalid-operation flag must be recorded, not skipped.
    x[x.size // 2] = np.inf
    y[x.size // 2] = 0
    expr = blosc2.lazyexpr("x * y", {"x": x, "y": y})
    previous = blosc2.set_nthreads(2)
    try:
        serial = getattr(expr, op)(_require_native=True, jit=True)
        flags = expr._native_execution_report["fp_flags"]
        result = getattr(expr, op)()
        assert bool(result) == (op == "any")
        assert result.tobytes() == serial.tobytes()
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
        assert expr._native_execution_report["fp_flags"] == flags
        assert flags & 1
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", ["bool", "int32", "uint64", "float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("root_expression", [False, True])
def test_parallel_compressed_truth(op, dtype, backend, root_expression, monkeypatch):
    import blosc2.native_graph as native
    from blosc2.storage_readers import StorageReader

    if not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        pytest.skip("Installed extension does not implement graph storage readers")
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 4 * 128 * 1024)
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    values = (np.arange(4 * 128 * 1024 + 7) % 3).astype(dtype)
    x = blosc2.asarray(values, chunks=(128 * 1024,), blocks=(8192,))
    expression = f"{op}(x)" if root_expression else "x"
    expr = blosc2.lazyexpr(expression, {"x": x})
    previous = blosc2.set_nthreads(2)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("Compressed truth reduction used Python/materialization/legacy execution")

        monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        monkeypatch.setattr(StorageReader, "read", forbidden)
        monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
        result = expr.compute() if root_expression else getattr(expr, op)()
        assert result.dtype == np.dtype("bool")
        assert bool(result) == bool(getattr(np, op)(values))
        report = expr._native_execution_report
        assert report["route"] == "parallel-graph-bounded-readers"
        assert report["threads"] == 2
        assert report["reader_bytes"] == values.nbytes
        assert report["maximum_reader_items"] <= report["tile_items"]
        assert report["input_materialization_bytes"] == 0
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
def test_parallel_truth_late_worker_failure_is_not_short_circuited(op, monkeypatch):
    import blosc2.resident_execution as resident

    submissions = []

    class FailedLaterPool:
        def submit(self, operation, inputs):
            from concurrent.futures import Future

            result = Future()
            submissions.append(True)
            if len(submissions) == 1:
                result.set_result(operation(inputs))
            else:
                result.set_exception(OSError("Later truth worker failed"))
            return result

    monkeypatch.setattr(resident, "resident_pool", lambda: FailedLaterPool())
    previous = blosc2.set_nthreads(2)
    try:
        values = np.full(4 * 128 * 1024, op == "any", dtype=bool)
        expr = blosc2.lazyexpr("x", {"x": values})
        with pytest.raises(OSError, match="Later truth worker failed"):
            getattr(expr, op)()
        assert len(submissions) == 2
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
def test_parallel_truth_direct_snan_status_matches_serial(op, dtype, backend, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    values = np.full(4 * 128 * 1024, 1 if op == "any" else 0, dtype=dtype)
    values.view("uint32" if dtype == "float32" else "uint64")[values.size // 2] = (
        0x7F800011 if dtype == "float32" else 0x7FF0000000000011
    )
    expr = blosc2.lazyexpr("x", {"x": values})
    previous = blosc2.set_nthreads(2)
    try:
        serial = getattr(expr, op)(_require_native=True, jit=True)
        flags = expr._native_execution_report["fp_flags"]
        result = getattr(expr, op)()
        assert result.tobytes() == serial.tobytes()
        assert expr._native_execution_report["fp_flags"] == flags
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
@pytest.mark.parametrize("dtype", ["int32", "float32", "float64"])
def test_parallel_truth_comparison_maps(op, dtype):
    values = np.arange(4 * 128 * 1024).astype(dtype)
    expr = blosc2.lazyexpr("x > y", {"x": values, "y": np.ones_like(values)})
    previous = blosc2.set_nthreads(2)
    try:
        result = getattr(expr, op)()
        assert bool(result) == bool(getattr(np, op)(values > 1))
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["any", "all"])
def test_parallel_compressed_truth_preserves_late_status(op, monkeypatch):
    import blosc2.native_graph as native

    if not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        pytest.skip("Installed extension does not implement graph storage readers")
    size = 4 * 128 * 1024
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", size)
    values = np.full(size, 1 if op == "any" else 0, dtype="float64")
    factor = np.ones_like(values)
    values[size // 2] = np.inf
    factor[size // 2] = 0
    inputs = {
        name: blosc2.asarray(value, chunks=(128 * 1024,), blocks=(8192,))
        for name, value in {"x": values, "y": factor}.items()
    }
    expr = blosc2.lazyexpr("x * y", inputs)
    previous = blosc2.set_nthreads(2)
    try:
        serial = getattr(expr, op)(_require_native=True, jit=True)
        flags = expr._native_execution_report["fp_flags"]
        result = getattr(expr, op)()
        assert result.tobytes() == serial.tobytes()
        report = expr._native_execution_report
        assert report["fp_flags"] == flags
        assert flags & 1
        assert report["reader_bytes"] == 2 * values.nbytes
        assert report["input_materialization_bytes"] == 0
        assert report["route"] == "parallel-graph-bounded-readers"
    finally:
        blosc2.set_nthreads(previous)
