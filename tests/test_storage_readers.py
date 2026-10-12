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
