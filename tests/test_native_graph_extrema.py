"""Parallel graph extrema retain native NaN, signed-zero and status semantics."""

import numpy as np
import pytest

import blosc2
from blosc2.native_graph import prepare_native_first


@pytest.fixture(autouse=True)
def graph_extrema_runtime(native_graph_runtime, request, monkeypatch):
    if blosc2.IS_WASM:
        pytest.skip("Parallel graph reductions require threads")
    backend = getattr(request.node, "callspec", None)
    backend = backend.params.get("backend") if backend is not None else None
    if backend is not None:
        monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    expr = blosc2.lazyexpr("x + 1", {"x": np.ones(4)})
    if prepare_native_first(expr, ()) is None:
        pytest.skip("Parallel graph extrema require an available native graph JIT")


def _extrema_values(dtype, case, size, first, second):
    x = np.linspace(-3, 4, size, dtype=dtype)
    if case in ("nan", "snan"):
        bits = x.view("uint32" if dtype == "float32" else "uint64")
        base = (
            (0x7F800000 if case == "snan" else 0x7FC00000)
            if dtype == "float32"
            else (0x7FF0000000000000 if case == "snan" else 0x7FF8000000000000)
        )
        sign = 0x80000000 if dtype == "float32" else 0x8000000000000000
        bits[first] = base | sign | 0x11
        bits[second] = base | 0x22
    elif case == "zeros":
        x.fill(-0.0)
        x[second] = 0.0
    elif case == "infinities":
        x[first] = -np.inf
        x[second] = np.inf
    return x


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("keepdims", [False, True])
def test_parallel_resident_extrema(op, dtype, backend, keepdims, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    x = np.linspace(-3, 4, 4 * 128 * 1024, dtype=dtype).reshape(4, -1)
    y = np.ones_like(x)
    expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
    previous = blosc2.set_nthreads(2)
    try:
        result = getattr(expr, op)(axis=(0, 1), keepdims=keepdims)
        expected = getattr(np, op)(x, axis=(0, 1), keepdims=keepdims)
        np.testing.assert_array_equal(result, expected)
        assert result.shape == expected.shape
        assert result.dtype == expected.dtype
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
        assert expr._native_execution_report["threads"] == 2
        serial = getattr(expr, op)(axis=(0, 1), keepdims=keepdims, _require_native=True, jit=True)
        assert result.tobytes() == serial.tobytes()
        assert expr._native_execution_report.get("route") != "parallel-resident-reduction"
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("case", ["nan", "snan", "zeros", "infinities"])
def test_parallel_extrema_special_values_match_serial(op, dtype, backend, case, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    size = 4 * 128 * 1024
    x = _extrema_values(dtype, case, size, size // 2 - 1, size // 2)
    expr = blosc2.lazyexpr("x * 1", {"x": x})
    previous = blosc2.set_nthreads(2)
    try:
        serial = getattr(expr, op)(_require_native=True, jit=True)
        serial_flags = expr._native_execution_report["fp_flags"]
        parallel = getattr(expr, op)()
        assert expr._native_execution_report["route"] == "parallel-resident-reduction"
        assert parallel.tobytes() == serial.tobytes()
        assert expr._native_execution_report["fp_flags"] == serial_flags
        if case == "zeros":
            assert bool(np.signbit(parallel)) == (op == "min")
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("kind", ["axis", "initial", "strided", "broadcast"])
def test_parallel_extrema_unsupported_layout_stays_serial(op, kind):
    x = np.ones((4, 128 * 1024))
    if kind == "strided":
        x = x[:, ::-1]
    y = np.ones((1, x.shape[1])) if kind == "broadcast" else np.ones_like(x)
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
    options = {"axis": 0} if kind == "axis" else {"initial": -3} if kind == "initial" else {}
    result = getattr(expr, op)(**options)
    expected = getattr(np, op)(x + y, **options)
    np.testing.assert_array_equal(result, expected)
    assert expr._native_execution_report.get("route") != "parallel-resident-reduction"


@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("root_expression", [False, True])
@pytest.mark.parametrize("case", ["finite", "nan", "snan", "zeros", "infinities"])
def test_parallel_compressed_extrema(op, dtype, backend, root_expression, case, monkeypatch):
    import blosc2.native_graph as native
    from blosc2.storage_readers import StorageReader

    if not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        pytest.skip("Installed extension does not implement graph storage readers")
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 4 * 128 * 1024)
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    # Special values in separate chunk regions exercise ordered partial
    # combination through native readers, including a ragged final chunk.
    values = _extrema_values(dtype, case, 4 * 128 * 1024 + 7, 200_000, 400_000)
    x = blosc2.asarray(values, chunks=(128 * 1024,), blocks=(8192,))
    expression = f"{op}(x * 1)" if root_expression else "x * 1"
    expr = blosc2.lazyexpr(expression, {"x": x})
    previous = blosc2.set_nthreads(2)
    try:
        serial = (
            expr.compute(_require_native=True, jit=True)
            if root_expression
            else getattr(expr, op)(_require_native=True, jit=True)
        )
        serial_flags = expr._native_execution_report["fp_flags"]

        def forbidden(*args, **kwargs):
            pytest.fail("Parallel compressed extrema used Python/materialization/legacy execution")

        monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        monkeypatch.setattr(StorageReader, "read", forbidden)
        monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
        result = expr.compute() if root_expression else getattr(expr, op)()
        assert result.tobytes() == serial.tobytes()
        report = expr._native_execution_report
        assert report["route"] == "parallel-graph-bounded-readers"
        assert report["fp_flags"] == serial_flags
        assert report["threads"] == 2
        assert report["input_materialization_bytes"] == 0
        assert report["maximum_reader_items"] <= report["tile_items"]
        assert report["reader_bytes"] == values.nbytes
    finally:
        blosc2.set_nthreads(previous)
