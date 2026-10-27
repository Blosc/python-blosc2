import numpy as np
import pytest

import blosc2
from blosc2.storage_readers import StorageReader, storage_reader_eligible


@pytest.fixture
def dispatch_runtime():
    if not getattr(blosc2.blosc2_ext, "menudet_prefilter_available", lambda: False)():
        pytest.skip("Runtime does not provide Menudet prefilter dispatch")


@pytest.fixture
def dense_dispatch_runtime(dispatch_runtime):
    if blosc2.IS_WASM:
        pytest.skip("Dense chunk/prefilter dispatch is host-only; WASM uses bounded readers")


def kernel(source="def k(x, y):\n    return x + (y - 1)\n", dtype="float64", output=None, jit=True):
    author = blosc2.DSLKernel.from_source(source)
    return blosc2.MenudetKernel.from_json(author.export({"x": dtype, "y": dtype}, output or dtype), jit=jit)


def forbidden(*args, **kwargs):
    pytest.fail("Unexpected evaluation route or execution replay")


def test_single_chunk_materializes_once(dense_dispatch_runtime, monkeypatch):
    values = np.arange(100_000.0)
    x = blosc2.asarray(values)
    lazy = kernel().lazy({"x": x, "y": x})
    monkeypatch.setattr(StorageReader, "read", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    np.testing.assert_array_equal(lazy[:], values * 2 - 1)
    assert lazy._source_execution_report["route"] == "single-chunk"
    assert lazy._source_execution_report["input_buffer_bytes"] == 1_600_000
    assert lazy._source_execution_report["fp_flags"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("jit", [False, True])
def test_multichunk_uses_menudet_prefilter(dense_dispatch_runtime, monkeypatch, dtype, jit):
    values = np.arange(2003, dtype=dtype)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    lazy = kernel(dtype=dtype, jit=jit).lazy({"y": x, "x": x})
    monkeypatch.setattr(StorageReader, "read", forbidden)
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", forbidden)
    np.testing.assert_array_equal(lazy[:], values + (values - 1))
    report = lazy._source_execution_report
    assert report["route"] == "menudet-prefilter"
    assert report["evaluated_blocks"] == 3  # Tiny source blocks share one batch per chunk.
    assert report["input_buffer_bytes"] <= 32 * 1024**2


def test_prefilter_output_dtype_and_padding(dense_dispatch_runtime):
    values = np.linspace(-1, 1, 2003)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    native = kernel("def k(x, y):\n    return x > y\n", output="bool")
    y = blosc2.asarray(np.zeros_like(values), chunks=(1000,), blocks=(100,))
    lazy = native.lazy({"x": x, "y": y})
    np.testing.assert_array_equal(lazy[:], values > 0)
    assert lazy._source_execution_report["evaluated_blocks"] == 3


def test_partial_read_stays_bounded(dispatch_runtime, monkeypatch):
    values = np.arange(100_000.0)
    x = blosc2.asarray(values)
    lazy = kernel().lazy({"x": x, "y": x})
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    np.testing.assert_array_equal(lazy[100:200], (values * 2 - 1)[100:200])
    assert lazy._source_execution_report["route"] == "bounded-readers"


def test_mismatched_layouts_stay_bounded(dispatch_runtime, monkeypatch):
    values = np.arange(5003.0)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    y = blosc2.asarray(values, chunks=(700,), blocks=(70,))
    lazy = kernel().lazy({"x": x, "y": y})
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    np.testing.assert_array_equal(lazy[:], values * 2 - 1)
    assert lazy._source_execution_report["route"] == "bounded-readers"


def test_single_chunk_failure_never_replays(dense_dispatch_runtime, monkeypatch):
    x = blosc2.asarray(np.arange(100.0))
    lazy = kernel().lazy({"x": x, "y": x})
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise ArithmeticError("Numerical execution failed")

    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", fail)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    with pytest.raises(ArithmeticError, match="Numerical execution failed"):
        lazy[:]
    assert len(calls) == 1


def test_single_chunk_decoder_failure_never_replays(dense_dispatch_runtime, monkeypatch):
    x = blosc2.asarray(np.arange(100.0))
    lazy = kernel().lazy({"x": x, "y": x})

    def fail(*args, **kwargs):
        raise OSError("Single-chunk decoding failed")

    monkeypatch.setattr(blosc2.SChunk, "decompress_chunk", fail)
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    with pytest.raises(OSError, match="Single-chunk decoding failed"):
        lazy[:]


def test_prefilter_failure_never_replays(dense_dispatch_runtime, monkeypatch):
    x = blosc2.asarray(np.arange(2000.0), chunks=(1000,), blocks=(100,))
    lazy = kernel().lazy({"x": x, "y": x})

    def fail(*args, **kwargs):
        raise OSError("Prefilter storage execution failed")

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", fail)
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    with pytest.raises(OSError, match="Prefilter storage execution failed"):
        lazy[:]


def test_native_prefilter_corrupt_layout_failure(dense_dispatch_runtime, monkeypatch):
    import struct

    x = blosc2.asarray(np.arange(2000.0), chunks=(1000,), blocks=(100,))
    compressed = bytearray(x.schunk.get_chunk(0))
    struct.pack_into("<i", compressed, 8, 400)  # Wrong physical block byte count.
    x.schunk.update_chunk(0, bytes(compressed))
    lazy = kernel().lazy({"x": x, "y": x})
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    with pytest.raises(blosc2.MenudetError, match="layout mismatch"):
        lazy[:]


def test_materialization_budget(dense_dispatch_runtime, monkeypatch):
    values = np.arange(2_125_000.0)
    x = blosc2.asarray(values, chunks=(values.size,), blocks=(15625,))
    lazy = kernel().lazy({"x": x, "y": x})
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    np.testing.assert_array_equal(lazy[:], values * 2 - 1)
    assert lazy._source_execution_report["route"] == "menudet-prefilter"


def test_materialization_budget_uses_physical_chunk_capacity(dense_dispatch_runtime, monkeypatch):
    values = np.arange(3.0)
    # Logical input bytes are tiny, but physical chunks exceed the materialization budget.
    x = blosc2.asarray(values, chunks=(2_125_000,), blocks=(15625,))
    lazy = kernel().lazy({"x": x, "y": x})
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    np.testing.assert_array_equal(lazy[:], values * 2 - 1)
    assert lazy._source_execution_report["route"] == "menudet-prefilter"
    assert lazy._source_execution_report["evaluated_blocks"] == 1


def test_dtype_preflight_before_chunk_reads(dispatch_runtime, monkeypatch):
    x = blosc2.asarray(np.arange(100.0, dtype="float32"))
    monkeypatch.setattr(blosc2.SChunk, "decompress_chunk", forbidden)
    with pytest.raises(ValueError, match="match native dtype"):
        kernel().lazy({"x": x, "y": x})


def test_persisted_inputs_use_readers(dispatch_runtime, tmp_path):
    values = np.arange(2003.0)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,), urlpath=str(tmp_path / "x.b2nd"))
    lazy = kernel().lazy({"x": x, "y": x})
    np.testing.assert_array_equal(lazy[:], values * 2 - 1)
    assert lazy._source_execution_report["route"] == "bounded-readers"


def test_prefilter_fp_status_excludes_padding(dispatch_runtime):
    values = np.ones(2003)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    native = kernel("def k(x, y):\n    return log(x) + y\n")
    lazy = native.lazy({"x": x, "y": x})
    np.testing.assert_array_equal(lazy[:], values)
    assert lazy._source_execution_report["fp_flags"] == 0  # No log(0) in padded lanes.


def test_prefilter_fp_status_matches_resident(dispatch_runtime):
    values = np.linspace(-1, 1, 2003)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    native = kernel("def k(x, y):\n    return sqrt(x) + y\n")
    lazy = native.lazy({"x": x, "y": x})
    expected, status = native.evaluate_block({"x": values, "y": values}, return_status=True)
    np.testing.assert_array_equal(lazy[:], expected)
    report = lazy._source_execution_report
    assert report["fp_flags"] == status["flags"]
    assert report["fp_supported"] == status["supported"]


@pytest.mark.skipif(blosc2.IS_WASM, reason="Concurrent prefilters require host threads")
def test_shared_artifact_concurrent_prefilters(dispatch_runtime):
    from concurrent.futures import ThreadPoolExecutor

    values = np.arange(2003.0)
    x = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    native = kernel()

    def evaluate(_):
        return native.lazy({"x": x, "y": x})[:]

    with ThreadPoolExecutor(max_workers=2) as workers:
        for result in workers.map(evaluate, range(4)):
            np.testing.assert_array_equal(result, values * 2 - 1)


@pytest.mark.parametrize("expression", ["x + captured", "where(x > 0, x + 1e100, y)"])
def test_checked_weak_conversion_metadata_rejects_retiling(dispatch_runtime, expression):
    from types import SimpleNamespace

    dtype = np.dtype("float32")
    metadata = SimpleNamespace(
        source=f"def k(x, y):\n    return {expression}\n",
        input_dtypes={"x": dtype, "y": dtype},
        output_dtype=dtype,
        inferred_dtype=dtype,
        result_cardinality="elementwise",
        context_ndim=0,
    )
    assert not storage_reader_eligible(metadata)


@pytest.mark.parametrize("chunks", [1000, 3000])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("output", ["float", "bool"])
def test_threadless_compressed_reads_use_bounded_readers(
    dispatch_runtime, monkeypatch, chunks, dtype, output
):
    values = np.linspace(-1, 1, 2003, dtype=dtype)
    x = blosc2.asarray(values, chunks=(chunks,), blocks=(100,))
    boolean = output == "bool"
    native = kernel(
        "def k(x, y):\n    return x > y\n" if boolean else "def k(x, y):\n    return x + (y - 1)\n",
        dtype=dtype,
        output="bool" if boolean else dtype,
    )
    lazy = native.lazy({"x": x, "y": x})
    # Exercise the real threadless dispatch gate even when this test runs on a host.
    monkeypatch.setattr(blosc2, "IS_WASM", True)
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    np.testing.assert_array_equal(lazy[:], values > values if boolean else values + (values - 1))
    assert lazy._source_execution_report["route"] == "bounded-readers"


def test_threadless_reader_failure_never_replays(dispatch_runtime, monkeypatch):
    x = blosc2.asarray(np.arange(2003.0), chunks=(1000,), blocks=(100,))
    lazy = kernel().lazy({"x": x, "y": x})
    calls = []

    def fail(*args, **kwargs):
        calls.append(True)
        raise OSError("Threadless storage read failed")

    monkeypatch.setattr(blosc2, "IS_WASM", True)
    monkeypatch.setattr(StorageReader, "read", fail)
    monkeypatch.setattr(blosc2.MenudetKernel, "evaluate_block", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", forbidden)
    with pytest.raises(OSError, match="Threadless storage read failed"):
        lazy[:]
    assert len(calls) == 1
