"""Graph reductions stream compressed inputs without full operand materialization."""

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def graph_sources_runtime(native_graph_runtime):
    if not getattr(blosc2.blosc2_ext, "native_graph_sources_available", lambda: False)():
        pytest.skip("Installed extension does not implement graph storage readers")


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("axis", [None, 0, -1, ()])
@pytest.mark.parametrize("keepdims", [False, True])
def test_graph_sources_axes_broadcast_and_mixed_storage(op, axis, keepdims, monkeypatch):
    values = np.arange(1, 25.0).reshape(4, 6)
    x = blosc2.asarray(values, chunks=(2, 3), blocks=(1, 3))
    y = np.ones((1, 6))
    expression = f"{op}(x + y, axis={axis!r}, keepdims={keepdims!r})"
    plan = blosc2.NativeGraph.from_expression(expression, {"x": "float64", "y": "float64"}, jit=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Graph source materialized a compressed operand")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    actual, report = plan.evaluate_sources({"x": x, "y": y}, tile_items=5, return_report=True)
    expected = getattr(np, op)(values + y, axis=axis, keepdims=keepdims)
    np.testing.assert_allclose(actual, expected)
    assert report["input_materialization_bytes"] == 0
    assert report["maximum_reader_items"] <= 5
    assert report["reader_bytes"] == values.nbytes


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "uint64", "bool"])
def test_graph_sources_direct_reduction_dtype(dtype):
    values = np.arange(20).astype(dtype)
    x = blosc2.asarray(values, chunks=(7,), blocks=(3,))
    plan = blosc2.NativeGraph.from_expression("sum(x)", {"x": dtype})
    result, report = plan.evaluate_sources({"x": x}, tile_items=4, return_report=True)
    expected = np.sum(values)
    assert result.dtype == expected.dtype
    assert result == expected
    assert report["evaluated_tiles"] == 0
    assert report["reader_bytes"] == values.nbytes


def test_required_compressed_graph_is_serial_and_bounded(monkeypatch):
    values = np.linspace(0, 1, 500_003)
    x = blosc2.asarray(values, chunks=(100_000,), blocks=(6250,))
    expr = blosc2.lazyexpr("x + 1", {"x": x})

    def forbidden(*args, **kwargs):
        pytest.fail("Strict graph reduction materialized input")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    result = expr.sum(_require_native=True, jit=True)
    np.testing.assert_allclose(result, np.sum(values + 1), rtol=1e-12)
    report = expr._native_execution_report
    assert report["route"] == "graph-bounded-readers"
    assert report["input_materialization_bytes"] == 0
    assert report["maximum_reader_items"] <= report["tile_items"]


@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("keepdims", [False, True])
def test_automatic_compressed_graph_parallel_sum(backend, mixed, keepdims, monkeypatch):
    import blosc2.native_graph as native
    from blosc2.storage_readers import StorageReader

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 4_000_000)
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend)
    values = np.linspace(0, 1, 4_200_003)
    x = blosc2.asarray(values, chunks=(1_000_000,), blocks=(6250,))
    y = (
        np.ones_like(values)
        if mixed
        else blosc2.asarray(np.ones_like(values), chunks=x.chunks, blocks=x.blocks)
    )
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})
    previous = blosc2.set_nthreads(2)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("Automatic compressed graph used materialization/legacy execution")

        monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        monkeypatch.setattr(StorageReader, "read", forbidden)
        monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
        result = expr.sum(keepdims=keepdims)
        np.testing.assert_allclose(result, np.sum(values + 1, keepdims=keepdims), rtol=1e-12)
        report = expr._native_execution_report
        assert report["route"] == "parallel-graph-bounded-readers"
        assert report["threads"] == 2
        assert report["input_materialization_bytes"] == 0
        assert report["maximum_reader_items"] <= report["tile_items"]
        assert report["reader_bytes"] == values.nbytes * (1 if mixed else 2)
    finally:
        blosc2.set_nthreads(previous)


def test_graph_sources_validation_precedes_reads(monkeypatch):
    from blosc2.storage_readers import StorageReader

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid graph bindings read storage")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    plan = blosc2.NativeGraph.from_expression("sum(x + y)", {"x": "float64", "y": "float64"})
    x = blosc2.asarray(np.ones(8))
    y = blosc2.asarray(np.ones(7))
    with pytest.raises(blosc2.PortableArtifactError):
        plan.evaluate_sources({"x": x, "y": y})


def test_graph_sources_reader_failure_is_not_replayed(monkeypatch):
    from blosc2.storage_readers import StorageReader

    calls = []

    def failure(*args, **kwargs):
        calls.append(True)
        raise OSError("Storage read failed")

    monkeypatch.setattr(StorageReader, "read", failure)
    expr = blosc2.lazyexpr("x + 1", {"x": blosc2.asarray(np.ones(20))})
    with pytest.raises(OSError, match="Storage read failed"):
        expr.sum(_require_native=True)
    assert calls == [True]


def test_graph_sources_staged_rejects_before_reads(monkeypatch):
    from blosc2.storage_readers import StorageReader

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported staged graph read storage")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    plan = blosc2.NativeGraph.from_expression("x - sum(x)", {"x": "float64"})
    with pytest.raises(ValueError, match="single-region"):
        plan.evaluate_sources({"x": blosc2.asarray(np.ones(8))})


def test_graph_sources_mismatched_partitions_and_initial(monkeypatch):
    values = np.linspace(0, 1, 101)
    x = blosc2.asarray(values, chunks=(17,), blocks=(5,))
    y = blosc2.asarray(values + 1, chunks=(23,), blocks=(7,))
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Streamed reduction materialized an operand")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    actual = expr.sum(initial=3, _require_native=True, jit=True)
    np.testing.assert_allclose(actual, np.sum(values + values + 1, initial=3))
    assert expr._native_execution_report["input_materialization_bytes"] == 0


def test_automatic_compressed_graph_persisted_native_readers(tmp_path, monkeypatch):
    import blosc2.native_graph as native
    from blosc2.storage_readers import StorageReader

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 4_000_000)
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc")
    values = np.linspace(0, 1, 4_200_003)
    x = blosc2.asarray(values, chunks=(1_000_000,), blocks=(6250,), urlpath=tmp_path / "x.b2nd")
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    previous = blosc2.set_nthreads(2)
    try:

        def forbidden(*args, **kwargs):
            pytest.fail("Native compressed graph called Python/materialization readers")

        monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        monkeypatch.setattr(StorageReader, "read", forbidden)
        for _ in range(2):
            np.testing.assert_allclose(expr.sum(), np.sum(values + 1), rtol=1e-12)
        assert expr._native_execution_report["route"] == "parallel-graph-bounded-readers"
        assert expr._native_execution_report["input_materialization_bytes"] == 0
    finally:
        blosc2.set_nthreads(previous)


def test_graph_sources_resident_mask_participation(monkeypatch):
    values = np.arange(20.0)
    x = blosc2.asarray(values, chunks=(7,), blocks=(3,))
    mask = np.arange(values.size) % 2 == 0
    plan = blosc2.NativeGraph.from_expression(
        "sum(x + 1, where=mask)", {"x": "float64", "mask": "bool"}, jit=True
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Masked graph materialized input")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    result, report = plan.evaluate_sources({"x": x, "mask": mask}, tile_items=5, return_report=True)
    assert result == np.sum(values[mask] + 1)
    assert report["reader_bytes"] == values[mask].nbytes
    assert report["maximum_reader_items"] <= 5


def test_graph_sources_compressed_mask_rejected_before_reads(monkeypatch):
    from blosc2.storage_readers import StorageReader

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported compressed mask read storage")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    inputs = {"x": blosc2.asarray(np.ones(8)), "mask": blosc2.asarray(np.ones(8, dtype=bool))}
    plan = blosc2.NativeGraph.from_expression("sum(x, where=mask)", {"x": "float64", "mask": "bool"})
    with pytest.raises(blosc2.PortableArtifactError):
        plan.evaluate_sources(inputs)


def test_graph_sources_empty_domain_does_not_read(monkeypatch):
    from blosc2.storage_readers import StorageReader

    def forbidden(*args, **kwargs):
        pytest.fail("Empty graph read storage")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    plan = blosc2.NativeGraph.from_expression("sum(x + 1)", {"x": "float64"})
    result, report = plan.evaluate_sources({"x": blosc2.asarray(np.empty(0))}, return_report=True)
    assert result == 0
    assert report["reader_bytes"] == report["maximum_reader_items"] == 0


def test_automatic_compressed_worker_failure_is_not_replayed(monkeypatch):
    import blosc2.native_graph as native
    import blosc2.resident_execution as resident

    class FailedPool:
        def submit(self, *args, **kwargs):
            from concurrent.futures import Future

            result = Future()
            result.set_exception(OSError("Compressed worker failed"))
            return result

    def forbidden(*args, **kwargs):
        pytest.fail("Failed compressed graph replayed legacy execution")

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 4_000_000)
    monkeypatch.setattr(resident, "resident_pool", lambda: FailedPool())
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    x = blosc2.asarray(np.ones(4_200_003), chunks=(1_000_000,), blocks=(6250,))
    previous = blosc2.set_nthreads(2)
    try:
        with pytest.raises(OSError, match="Compressed worker failed"):
            blosc2.lazyexpr("x + 1", {"x": x}).sum()
    finally:
        blosc2.set_nthreads(previous)
