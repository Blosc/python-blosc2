"""Whole-domain compressed maps reuse readers and write directly into NumPy output."""

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
        pytest.fail("Graph map must not materialize inputs or invoke compression prefilters")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)
    result = expression[:]
    np.testing.assert_array_equal(result, expected)
    report = expression._native_execution_report
    assert report["route"] == "parallel-graph-bounded-map"
    assert report["threads"] == 2
    assert report["native_decoder_contexts"] == (2 if mixed else 4)
    assert report["reader_bytes"] == values.nbytes * (1 if mixed else 2)
    assert report["maximum_reader_items"] <= report["tile_items"]
    # Reader scratch includes input values and 64-bit gather indices, even
    # when the output is only Boolean. Bound it by tiles/workers, not output size.
    assert report["temporary_bytes"] <= report["tile_items"] * report["threads"] * (2 * values.itemsize + 16)
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
            submissions.append(kwargs["out"])
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
    assert not np.shares_memory(*submissions)
    assert submissions[0].base is submissions[1].base


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
    eligible = dtype == "float64" and threads >= 8 and size >= 2_000_000
    assert (prepared is not None) == eligible
    if eligible:
        result = expr[:]
        np.testing.assert_array_equal(result, values + 1)
        assert expr._native_execution_report["threads"] > x.schunk.nchunks
        assert expr._native_execution_report["tile_items"] == x.blocks[0]
    # Reductions keep their old floor even for the new map crossover.
    assert native.prepare_native_first(blosc2.lazyexpr("sum(x + 1)", {"x": x}), ()) is None


def test_map_schedules_are_immutable_and_operand_free():
    import gc
    import weakref

    import blosc2.native_graph as native

    native._compressed_map_schedules.cache_clear()
    values = np.arange(4 * 65536, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(4 * 65536,), blocks=(4096,))
    reference = weakref.ref(x)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
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


def test_map_groups_tiny_blocks_to_amortize_dispatch():
    values = np.arange(4 * 65536, dtype=np.float64)
    x = blosc2.asarray(values, chunks=(65536,), blocks=(128,))
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    np.testing.assert_array_equal(expr[:], values + 1)
    assert expr._native_execution_report["tile_items"] == 4096
