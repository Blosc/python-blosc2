import numpy as np
import pytest

import blosc2
from blosc2.storage_readers import StorageReader, storage_reader_eligible


@pytest.fixture
def storage_runtime():
    if not getattr(blosc2.blosc2_ext, "native_array_sources_available", lambda: False)():
        pytest.skip("Native dependency does not provide bounded storage readers")


def make_kernel(source="def k(x, y):\n    return x + (y - 1)\n", inputs=None, output="float64", jit=True):
    author = blosc2.DSLKernel.from_source(source)
    return blosc2.PortableKernel.from_json(
        author.export(inputs or {"x": "float64", "y": "float64"}, output), jit=jit
    )


@pytest.mark.parametrize("jit", [False, True])
def test_bounded_full_reads(storage_runtime, jit, monkeypatch):
    size, tile = 200_000, 4096
    x_np, y_np = np.arange(size, dtype="float64"), np.ones(size)
    x, y = blosc2.asarray(x_np), blosc2.asarray(y_np)
    original = blosc2.NDArray.get_1d_span_numpy
    requested = []

    def bounded(self, array, chunk, start, count, **kwargs):
        requested.append(count)
        assert requested[-1] <= tile
        return original(self, array, chunk, start, count, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "get_1d_span_numpy", bounded)
    kernel = make_kernel(jit=jit)
    values, report = kernel.evaluate_sources({"x": x, "y": y}, tile_items=tile, return_report=True)
    np.testing.assert_array_equal(values, x_np + (y_np - 1))
    assert max(requested) == report["maximum_reader_items"] == tile
    assert report["reader_bytes"] == x_np.nbytes + y_np.nbytes
    assert report["temporary_bytes"] <= tile * 24  # two values plus logical indices


def test_lazy_and_compute_use_readers(storage_runtime, monkeypatch):
    x_np = np.arange(200_000, dtype="float64")
    x, y = blosc2.asarray(x_np, chunks=(8192,), blocks=(1024,)), np.ones(x_np.size)
    kernel = make_kernel()
    lazy = kernel.lazy({"x": x, "y": y})

    def forbidden(*args, **kwargs):
        pytest.fail("Original whole-partition block path used")

    monkeypatch.setattr(blosc2.PortableKernel, "evaluate_block", forbidden)
    np.testing.assert_array_equal(lazy[:], x_np)
    assert lazy._source_execution_report["maximum_reader_items"] <= 65536
    np.testing.assert_array_equal(lazy.compute(chunks=(8192,), blocks=(1024,))[:], x_np)


@pytest.mark.parametrize(
    "item",
    [
        (slice(1, 10, 2), slice(None, None, -2)),
        (3, slice(2, 11)),
        (slice(None), 1),
        (slice(0, 0), slice(None)),
    ],
)
def test_partial_broadcasting(storage_runtime, item):
    x_np = np.arange(256.0).reshape(16, 16)
    y_np = np.arange(16.0).reshape(1, 16)
    kernel = make_kernel()
    lazy = kernel.lazy({"x": blosc2.asarray(x_np), "y": blosc2.asarray(y_np)})
    np.testing.assert_array_equal(lazy[item], (x_np + (y_np - 1))[item])


def test_strided_numpy_and_scalar_broadcast(storage_runtime):
    x_np = np.arange(32.0).reshape(4, 8)
    y_np = x_np[:, ::-1]
    kernel = make_kernel()
    np.testing.assert_array_equal(
        kernel.lazy({"x": blosc2.asarray(x_np), "y": y_np})[::-1, ::2],
        (x_np + (y_np - 1))[::-1, ::2],
    )
    scalar = blosc2.asarray(np.array(2.0))
    np.testing.assert_array_equal(kernel.lazy({"x": scalar, "y": blosc2.asarray(y_np)})[:], 2 + y_np - 1)


def test_reader_error_not_replayed(storage_runtime, monkeypatch):
    kernel = make_kernel()
    x = blosc2.asarray(np.arange(200_000.0))
    lazy = kernel.lazy({"x": x, "y": x})
    calls = []
    original = StorageReader.read

    def failure(self, indices):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("Storage failed")
        return original(self, indices)

    def forbidden(*args, **kwargs):
        pytest.fail("Storage failure replayed through whole-partition execution")

    monkeypatch.setattr(StorageReader, "read", failure)
    monkeypatch.setattr(blosc2.PortableKernel, "evaluate_block", forbidden)
    with pytest.raises(OSError, match="Storage failed"):
        lazy[::2]
    assert len(calls) == 2


def test_preflight_without_reads(storage_runtime, monkeypatch):
    kernel = make_kernel()
    x = blosc2.asarray(np.arange(4.0))

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid request read storage")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    with pytest.raises(ValueError, match="broadcast domain"):
        kernel.evaluate_sources({"x": x, "y": x}, shape=(8,))
    with pytest.raises(ValueError, match="tile_items"):
        kernel.evaluate_sources({"x": x, "y": x}, tile_items=0)


def test_block_dependency_preserves_original_groups(storage_runtime):
    kernel = make_kernel("def k(x):\n    return x - block_sum(x)\n", {"x": "float64"})
    assert not storage_reader_eligible(kernel)
    x = blosc2.asarray(np.arange(4.0))
    np.testing.assert_array_equal(kernel.lazy({"x": x})[:1], [-6])


def test_context_and_final_cast_retain_original_path(storage_runtime):
    cast = make_kernel("def k(x):\n    return x\n", {"x": "float64"}, "int32")
    assert not storage_reader_eligible(cast)
    author = blosc2.DSLKernel.from_source("def k(x):\n    return x + _i0\n")
    context = blosc2.PortableKernel.from_json(author.export({"x": "float64"}, "float64", ndim=1))
    assert not storage_reader_eligible(context)
    np.testing.assert_array_equal(context.lazy({"x": blosc2.asarray(np.ones(4))})[2:], [3, 4])


def test_old_runtime_retains_existing_path(monkeypatch):
    kernel = make_kernel(jit=False)
    monkeypatch.setattr(blosc2.blosc2_ext, "native_array_sources_available", lambda: False, raising=False)
    x = blosc2.asarray(np.arange(4.0))
    assert not storage_reader_eligible(kernel)
    np.testing.assert_array_equal(kernel.lazy({"x": x, "y": x})[:], np.arange(4.0) * 2 - 1)


def test_weak_intermediates_keep_original_error_domain(storage_runtime):
    kernel = make_kernel("def k(x):\n    return x + (2 * 3)\n", {"x": "float64"})
    assert not storage_reader_eligible(kernel)
    kernel = make_kernel("def k(x):\n    return x + 1\n", {"x": "int64"}, "int64")
    assert not storage_reader_eligible(kernel)


def test_chunk_crossing_span(storage_runtime):
    values = np.arange(10003.0)
    x = blosc2.asarray(values, chunks=(3000,), blocks=(300,))
    kernel = make_kernel()
    actual, report = kernel.evaluate_sources({"x": x, "y": x}, tile_items=4096, return_report=True)
    np.testing.assert_array_equal(actual, values * 2 - 1)
    assert report["maximum_reader_items"] == 4096


@pytest.mark.parametrize("jit", [False, True])
def test_numerical_status_matches_resident_tiles(storage_runtime, jit):
    kernel = make_kernel("def k(x):\n    return sqrt(x)\n", {"x": "float64"}, jit=jit)
    values = np.linspace(-1, 1, 1001)
    expected, resident = kernel.evaluate_array({"x": values}, tile_items=128, return_report=True)
    actual, streamed = kernel.evaluate_sources(
        {"x": blosc2.asarray(values)}, tile_items=128, return_report=True
    )
    np.testing.assert_array_equal(actual, expected)
    assert streamed["fp_flags"] == resident["fp_flags"]
    assert streamed["fp_supported"] == resident["fp_supported"]


@pytest.mark.parametrize("size", [17, 257])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("threads", [1, 2, 12])
def test_compute_returns_prefilter_output_without_numpy_staging(
    storage_runtime, monkeypatch, size, dtype, jit, threads
):
    values = np.arange(size, dtype=dtype)
    x = blosc2.asarray(values, chunks=(128,), blocks=(32,))
    kernel = make_kernel(inputs={"x": dtype, "y": dtype}, output=dtype, jit=jit)
    lazy = kernel.lazy({"y": x, "x": x})  # Binding order need not match signature order.
    original = blosc2.NDArray.__getitem__

    def forbidden(*args, **kwargs):
        pytest.fail("Compute materialized a NumPy intermediate")

    with monkeypatch.context() as patch:
        patch.setattr(blosc2.NDArray, "__getitem__", forbidden)
        patch.setattr(lazy, "_values", forbidden)
        patch.setattr(blosc2.PortableKernel, "evaluate_sources", forbidden)
        result = lazy.compute(cparams={"clevel": 0, "nthreads": threads})
        second = lazy.compute(cparams={"clevel": 0, "nthreads": threads})
    assert isinstance(result, blosc2.NDArray)
    assert result is not second
    assert result.chunks == x.chunks
    assert result.blocks == x.blocks
    assert result.cparams.clevel == 0
    assert lazy._source_execution_report["route"] == "portable-prefilter"
    assert lazy._source_execution_report["evaluated_blocks"] == (size + 31) // 32
    report = lazy._source_execution_report
    assert report["threads"] == threads
    assert 2 <= report["native_decoder_contexts"] <= threads * 2
    assert report["input_buffer_allocations"] == report["native_decoder_contexts"]
    assert report["input_chunk_fetches"] == x.schunk.nchunks * 2
    padded_items = x.schunk.nchunks * x.chunks[0] - size
    assert report["output_zero_fill_bytes"] == padded_items * result.dtype.itemsize
    np.testing.assert_array_equal(original(result, slice(None)), values * 2 - 1)
    tail = np.frombuffer(result.schunk.decompress_chunk(result.schunk.nchunks - 1), dtype=result.dtype)
    live_items = size - (result.schunk.nchunks - 1) * result.chunks[0]
    np.testing.assert_array_equal(tail[live_items:], 0)


def test_compute_prefilter_preserves_storage_options(storage_runtime, tmp_path):
    values = np.arange(257.0)
    x = blosc2.asarray(values, chunks=(128,), blocks=(32,))
    lazy = make_kernel().lazy({"x": x, "y": x})
    cparams = blosc2.CParams(clevel=5, nthreads=2)
    path = tmp_path / "portable-compute.b2nd"
    result = lazy.compute(cparams=cparams, urlpath=path, meta={"test": "direct"})
    assert cparams.nthreads == 2  # The caller's options are not mutated.
    assert result.cparams.clevel == 5
    assert result.schunk.meta["test"] == "direct"
    assert lazy._source_execution_report["output_zero_fill_bytes"] == x.schunk.nchunks * x.schunk.chunksize
    np.testing.assert_array_equal(result[:], values * 2 - 1)
    np.testing.assert_array_equal(blosc2.open(path)[:], values * 2 - 1)


@pytest.mark.parametrize("partial", [False, True])
def test_compute_other_layouts_and_selections_keep_fallback(storage_runtime, partial):
    values = np.arange(257.0)
    x = blosc2.asarray(values, chunks=(128,), blocks=(32,))
    lazy = make_kernel().lazy({"x": x, "y": x})
    item = slice(3, 250, 2) if partial else None
    result = lazy.compute(item=item, chunks=(64,), blocks=(16,), cparams={"clevel": 0})
    expected = values * 2 - 1
    np.testing.assert_array_equal(result[:], expected[item] if partial else expected)
    assert result.chunks == (64,)
    assert result.blocks == (16,)
    assert lazy._source_execution_report["route"] == "bounded-readers"


def test_compute_prefilter_fp_status(storage_runtime):
    values = np.linspace(-1, 1, 257)
    x = blosc2.asarray(values, chunks=(128,), blocks=(32,))
    kernel = make_kernel("def k(x):\n    return sqrt(x)\n", {"x": "float64"})
    expected, status = kernel.evaluate_array({"x": values}, tile_items=32, return_report=True)
    lazy = kernel.lazy({"x": x})
    result = lazy.compute(cparams={"clevel": 0})
    np.testing.assert_array_equal(result[:], expected)
    assert lazy._source_execution_report["fp_flags"] == status["fp_flags"]
    assert lazy._source_execution_report["fp_supported"] == status["fp_supported"]


def test_prefilter_reuses_scratch_across_chunks_and_rebinding(storage_runtime):
    values = np.arange(10003.0)
    x = blosc2.asarray(values, chunks=(3000,), blocks=(300,))
    lazy = make_kernel().lazy({"x": x, "y": x})
    options = {"cparams": {"clevel": 0, "nthreads": 3}}
    first = lazy.compute(**options)
    np.testing.assert_array_equal(first[:], values * 2 - 1)
    report = lazy._source_execution_report
    assert report["evaluated_blocks"] == 34
    assert 2 <= report["native_decoder_contexts"] <= 6
    assert report["input_buffer_allocations"] == report["native_decoder_contexts"]
    assert report["input_chunk_fetches"] == 8
    x[:] = values + 10
    second = lazy.compute(**options)
    np.testing.assert_array_equal(second[:], (values + 10) * 2 - 1)
    np.testing.assert_array_equal(first[:], values * 2 - 1)


@pytest.mark.parametrize("clevel", [0, 5])
def test_prefilter_failure_is_raised_and_output_can_be_reused(storage_runtime, clevel):
    # Call the native adapter directly to exercise worker failures. The Python
    # lane-local gate intentionally excludes this checked weak-scalar operation.
    failing = make_kernel("def k(x):\n    return x + (1 // 0)\n", {"x": "float64"})
    x = blosc2.asarray(np.arange(257.0), chunks=(128,), blocks=(32,))
    output = blosc2.uninit(
        x.shape, dtype=x.dtype, chunks=x.chunks, blocks=x.blocks, cparams={"clevel": clevel, "nthreads": 3}
    )
    for _ in range(3):
        with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
            failing._handle.evaluate_prefilter({"x": x}, out=output)
    healthy = make_kernel()
    result, report = healthy._handle.evaluate_prefilter({"x": x, "y": x}, out=output)
    assert result is output
    assert report["evaluated_blocks"] == 9
    np.testing.assert_array_equal(result[:], np.arange(257.0) * 2 - 1)


@pytest.mark.parametrize("clevel", [0, 1, 5])
def test_prefilter_incompressible_output_and_concurrent_calls(storage_runtime, clevel):
    from concurrent.futures import ThreadPoolExecutor

    values = np.random.default_rng(42).normal(size=10003)
    x = blosc2.asarray(values, chunks=(3000,), blocks=(300,))
    kernel = make_kernel()

    def compute():
        return kernel.lazy({"x": x, "y": x}).compute(cparams={"clevel": clevel, "nthreads": 3})

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: compute(), range(4)))
    for result in results:
        np.testing.assert_array_equal(result[:], values + (values - 1))
        assert result.cparams.clevel == clevel
