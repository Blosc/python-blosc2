"""Compressed maps share native prefilters, with bounded readers for other layouts."""

from concurrent.futures import Future
from itertools import pairwise

import numpy as np
import pytest

import blosc2
from conftest import orthogonal_cases


@pytest.fixture(autouse=True)
def map_runtime(native_graph_runtime, monkeypatch, request):
    import blosc2.native_graph as native

    if blosc2.IS_WASM:
        pytest.skip("Parallel maps require threads")
    backend = getattr(request.node, "callspec", None)
    if backend is not None and "backend" in backend.params:
        monkeypatch.setenv("ME_DSL_JIT_COMPILER", backend.params["backend"])
    probe = blosc2.NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True)
    if not probe.has_jit:
        pytest.skip("Graph JIT unavailable")
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 128 * 1024)
    previous = blosc2.set_nthreads(2)
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize(
    ("backend", "dtype", "mixed", "comparison"),
    orthogonal_cases(["tcc", "cc"], ["float32", "float64", "int32", "uint32"], [False, True], [False, True]),
)
def test_parallel_compressed_map(backend, dtype, mixed, comparison, monkeypatch):
    size = 4 * 65536 + 17
    values = (np.arange(size) % 251).astype(dtype)
    x = blosc2.asarray(values, chunks=(65536,), blocks=(4096,))
    y = values if mixed else blosc2.asarray(values, chunks=x.chunks, blocks=x.blocks)
    source = "x + (y - 1)" if not comparison else "x > y - 1"
    expected = values + (values - 1) if not comparison else values > values - 1
    expression = blosc2.lazyexpr(source, {"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Graph map must not materialize inputs or invoke Python expression prefilters")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)
    result = expression[:]
    np.testing.assert_array_equal(result, expected)
    report = expression._native_execution_report
    assert report["route"] == "portable-prefilter"
    assert report["threads"] == 2
    assert report["native_decoder_contexts"] == (2 if mixed else 4)
    assert report["reader_bytes"] == values.nbytes * (1 if mixed else 2)
    assert report["maximum_reader_items"] <= report["tile_items"]
    # Prefilter scratch contains only compressed input values, even when the
    # final output is Boolean; resident operands are borrowed without copying.
    assert report["input_buffer_bytes"] == (
        report["tile_items"] * report["threads"] * (1 if mixed else 2) * values.itemsize
    )
    assert report["temporary_bytes"] <= 32 * 1024**2
    assert report["output_copy_bytes"] == report["output_zero_fill_bytes"] == 0
    again = blosc2.evaluate(source, local_dict={"x": x, "y": y})
    assert again is not result
    np.testing.assert_array_equal(again, expected)


def test_parallel_map_persisted_sources(tmp_path):
    values = np.arange(4 * 65536 + 1, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(65536,), blocks=(8192,), urlpath=str(tmp_path / "x.b2nd"))
    expression = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expression[:], values + 1)
    assert expression._native_execution_report["route"] == "parallel-graph-bounded-map"


@pytest.mark.parametrize("mixed", [False, True])
def test_integer_prefilter_avoids_gather_indices(monkeypatch, mixed):
    monkeypatch.setattr("blosc2.storage_readers.DENSE_MATERIALIZATION_BUDGET", 2048)
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 64)
    monkeypatch.setattr("blosc2.native_graph.MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    values = np.arange(1027, dtype="int16") + np.int16(32000)
    x = blosc2.asarray(values, chunks=(256,), blocks=(64,))
    y = values if mixed else blosc2.asarray(values, chunks=x.chunks, blocks=x.blocks)
    expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Chunk map used a gather schedule or materialized compressed inputs")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr("blosc2.native_graph._compressed_map_schedules", forbidden)
    expected = values + (values - 1)
    np.testing.assert_array_equal(expr[:], expected)
    output = np.empty_like(values)
    assert blosc2.evaluate("x + (y - 1)", local_dict={"x": x, "y": y}, out=output) is output
    np.testing.assert_array_equal(output, expected)
    report = expr._native_execution_report
    assert report["route"] == "portable-prefilter"
    assert report["gathered_bytes"] == report["output_copy_bytes"] == 0
    assert report["temporary_bytes"] <= 2048
    assert report["evaluated_tiles"] == 4 * (256 // report["tile_items"]) + 1
    assert report["reader_bytes"] == values.nbytes * (1 if mixed else 2)


def test_prefilter_workers_join_failures(monkeypatch):
    from types import SimpleNamespace

    from blosc2.storage_readers import evaluate_prefilter_numpy

    values = np.arange(4099, dtype="int16")
    x = blosc2.asarray(values, chunks=(256,), blocks=(64,))
    author = blosc2.DSLKernel.from_source("def k(x):\n    return x + 1\n")
    native = blosc2.PortableKernel.from_json(author.export({"x": "int16"}, "int16"), jit=True)
    output = np.empty_like(values)
    completed = []

    def execute(inputs, *, numpy_out, _chunk_range, _driver_input, _block_items):
        if _chunk_range[0] == 0:
            raise OSError("decoder failure")
        result = native._handle.evaluate_prefilter(
            inputs,
            numpy_out=numpy_out,
            _chunk_range=_chunk_range,
            _driver_input=_driver_input,
            _block_items=_block_items,
        )
        completed.append(_chunk_range)
        return result

    fake = SimpleNamespace(
        output_dtype=native.output_dtype, _handle=SimpleNamespace(evaluate_prefilter=execute)
    )
    with pytest.raises(OSError, match="decoder failure"):
        evaluate_prefilter_numpy(fake, {"x": x}, out=output, workers=2)
    # The second worker finishes before the first worker's error escapes.
    assert completed == [(8, 17)]
    np.testing.assert_array_equal(output[8 * 256 :], (values + 1)[8 * 256 :])
    result, _ = evaluate_prefilter_numpy(native, {"x": x}, workers=2)
    np.testing.assert_array_equal(result, values + 1)


def test_integer_prefilter_keeps_wide_inputs_parallel(monkeypatch):
    monkeypatch.setattr("blosc2.storage_readers.DENSE_MATERIALIZATION_BUDGET", 4608)
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 64)
    monkeypatch.setattr("blosc2.native_graph.MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    bounds = np.iinfo("int64")
    values = np.resize(np.array([bounds.min, bounds.max, 0, 1], dtype="int64"), 1027)
    x = blosc2.asarray(values, chunks=(256,), blocks=(64,))
    y = blosc2.asarray(values, chunks=x.chunks, blocks=x.blocks)
    expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Wide-input map fell back instead of reducing its scratch batch size")

    monkeypatch.setattr("blosc2.native_graph._compressed_map_schedules", forbidden)
    np.testing.assert_array_equal(expr[:], values + (values - 1))
    report = expr._native_execution_report
    assert report["route"] == "portable-prefilter"
    assert report["threads"] == 2
    assert report["tile_items"] == 64
    assert report["temporary_bytes"] <= 4608
    assert report["evaluated_tiles"] == 17


def test_parallel_map_padded_chunk_layout():
    values = np.arange(4 * 65536 + 19, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(60000,), blocks=(4096,))
    expression = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expression[:], values + 1)
    assert expression._native_execution_report["route"] == "parallel-graph-bounded-map"


def test_parallel_map_keeps_compressed_compute_and_serial_fallback():
    values = np.arange(4 * 65536, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(65536,), blocks=(4096,))
    expression = blosc2.lazyexpr("x + 1", {"x": x})
    result = expression.compute()
    assert isinstance(result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], values + 1)
    blosc2.set_nthreads(1)
    np.testing.assert_array_equal(expression[:], values + 1)
    assert not hasattr(expression, "_native_execution_report")


def test_parallel_map_aggregates_late_floating_status():
    values = np.ones(4 * 65536, dtype=np.float64)
    values[-1] = 1e308
    x = blosc2.asarray(values, chunks=(65536,), blocks=(4096,))
    expression = blosc2.lazyexpr("x * x", {"x": x})
    result = expression[:]
    assert np.isinf(result[-1])
    np.testing.assert_array_equal(result[:-1], 1)
    assert expression._native_execution_report["fp_flags"] != 0


def test_parallel_map_python_postfilter():
    values = np.arange(4 * 65536, dtype=np.int32)
    x = blosc2.asarray(values, chunks=(65536,), blocks=(4096,))
    x.schunk.dparams = blosc2.DParams(nthreads=1)

    @x.schunk.postfilter(np.int32, np.int32)
    def add_offset(input, output, offset):
        output[:] = input + 10

    expression = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expression[:], values + 11)
    assert expression._native_execution_report["route"] == "parallel-graph-bounded-map"


def test_parallel_map_worker_failure_is_not_replayed(monkeypatch):
    import blosc2.resident_execution as resident

    submissions = []

    class FailedPool:
        def submit(self, *args, **kwargs):
            submissions.append(kwargs["_chunk_range"])
            result = Future()
            result.set_exception(OSError("map worker failed"))
            return result

    def forbidden(*args, **kwargs):
        pytest.fail("Failed graph execution was replayed")

    x = blosc2.asarray(np.ones(4 * 65536), chunks=(65536,), blocks=(4096,))
    monkeypatch.setattr(resident, "resident_pool", lambda: FailedPool())
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    with pytest.raises(OSError, match="map worker failed"):
        blosc2.lazyexpr("x + 1", {"x": x})[:]
    assert len(submissions) == 2
    assert submissions == [(0, 2), (2, 4)]


def test_graph_sources_explicit_output_validation():
    values = np.arange(16, dtype=np.float64)
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True)
    schedule = plan.specialize({"x": (values.dtype, values.shape)})
    out = np.empty_like(values)
    result, _ = schedule.execute_sources({"x": values}, {}, out=out)
    assert result is out
    np.testing.assert_array_equal(out, values + 1)
    for invalid in (np.empty(15), np.empty(16, dtype=np.float32), np.empty(32)[::2]):
        with pytest.raises(ValueError):
            schedule.execute_sources({"x": values}, {}, out=invalid)
    out.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        schedule.execute_sources({"x": values}, {}, out=out)
    with pytest.raises(ValueError, match="overlap"):
        schedule.execute_sources({"x": values}, {}, out=values)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32"])
@pytest.mark.parametrize("threads", [4, 8, 12])
@pytest.mark.parametrize("size", [1_000_000, 2_000_000])
def test_map_crossover_policy(dtype, threads, size, monkeypatch):
    import blosc2.native_graph as native

    # Restore production limits rather than the fixture's small test threshold.
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 16_000_000)
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_MAP_ITEMS", 2_000_000)
    blosc2.set_nthreads(threads)
    values = np.arange(size, dtype=dtype)
    x = blosc2.asarray(values, chunks=(size,), blocks=(15625,))
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    prepared = native.prepare_native_first(expr, (), options={"_getitem": True})
    eligible = dtype in ("float64", "int32") and threads >= 8 and size >= 2_000_000
    assert (prepared is not None) == eligible
    if eligible:
        result = expr[:]
        np.testing.assert_array_equal(result, values + 1)
        assert expr._native_execution_report["threads"] > x.schunk.nchunks
        assert expr._native_execution_report["tile_items"] == x.blocks[0]
    # Reductions keep their old floor even for the new map crossover.
    assert native.prepare_native_first(blosc2.lazyexpr("sum(x + 1)", {"x": x}), ()) is None


def test_map_schedules_are_immutable_and_operand_free(tmp_path):
    import gc
    import weakref

    import blosc2.native_graph as native

    native._compressed_map_schedules.cache_clear()
    values = np.arange(4 * 65536, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(4 * 65536,), blocks=(4096,), urlpath=str(tmp_path / "source.b2nd"))
    reference = weakref.ref(x)
    # Immutable gather schedules remain the fallback for persisted operands.
    expr = blosc2.lazyexpr("x + y", {"x": x, "y": np.ones_like(values)})
    np.testing.assert_array_equal(expr[:], values + 1)
    hits = native._compressed_map_schedules.cache_info().hits
    np.testing.assert_array_equal(expr[:], values + 1)
    assert native._compressed_map_schedules.cache_info().hits > hits
    del x, expr
    gc.collect()
    assert reference() is None


def test_balanced_regions_split_physical_blocks_with_tail_padding():
    import blosc2.native_graph as native

    values = np.arange(4 * 65536 + 1, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(60000,), blocks=(4096,))
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True)
    units = native._compressed_map_blocks(x, values.size)
    regions = native._compressed_map_schedules(plan, values.shape, 8, 60000, 4096, units, 4096)
    assert regions[0][0] == 0
    assert regions[-1][1] == values.size
    for start, stop, _ in regions:
        assert start < stop
        assert start % 60000 % 4096 == 0
    assert all(left[1] == right[0] for left, right in pairwise(regions))


@pytest.mark.parametrize("padded", [False, True])
def test_map_groups_tiny_blocks_to_amortize_dispatch(padded):
    values = np.arange(4 * 65536, dtype=np.float64)
    # Padded chunk grids retain bounded readers rather than the prefilter.
    x = blosc2.asarray(values, chunks=(60000 if padded else 65536,), blocks=(128,))
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expr[:], values + 1)
    assert expr._native_execution_report["tile_items"] == 4096
    assert expr._native_execution_report["route"] == (
        "parallel-graph-bounded-map" if padded else "portable-prefilter"
    )
