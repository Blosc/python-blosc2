"""NumPy output uses bounded prefilter scratch rather than a full stored result."""

import sys

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "complex128"])
@pytest.mark.parametrize("mode", ["evaluate", "getitem", "out", "strided-out"])
def test_numpy_prefilter_output(dtype, mode, monkeypatch):
    if dtype == "complex128" and (blosc2.IS_WASM or sys.platform == "win32"):
        pytest.skip("Complex prefilter execution is unavailable on this platform")
    values = np.arange(257, dtype=np.float64).astype(dtype)
    inputs = {
        "x": blosc2.asarray(values, chunks=(128,), blocks=(32,)),
        "y": blosc2.asarray(values + 1, chunks=(128,), blocks=(32,)),
    }
    expected = values + values
    expression = "x + (y - 1)"
    lazy = blosc2.lazyexpr(expression, inputs)
    calls = []
    buffers = []
    original = blosc2.SChunk._prefilter_data

    def prefilter(self, nchunk, data, chunk_data):
        calls.append((nchunk, id(chunk_data), chunk_data.size))
        buffers.append(chunk_data)
        return original(self, nchunk, data, chunk_data)

    def no_stored_chunks(*args, **kwargs):
        pytest.fail("NumPy results must not store intermediate result chunks")

    monkeypatch.setattr(blosc2.SChunk, "_prefilter_data", prefilter)
    monkeypatch.setattr(blosc2.SChunk, "update_data", no_stored_chunks)
    if mode == "getitem":
        result = lazy[:]
    elif mode == "evaluate":
        result = blosc2.evaluate(expression, local_dict=inputs)
        again = blosc2.evaluate(expression, local_dict=inputs)
        assert result is not again
        np.testing.assert_array_equal(again, expected)
    else:
        out = np.empty(257 if mode == "out" else 514, dtype=dtype)
        if mode == "strided-out":
            out[:] = -99
            storage = out
            out = out[::2]
        result = blosc2.evaluate(expression, local_dict=inputs, out=out)
        assert result is out
        if mode == "strided-out":
            np.testing.assert_array_equal(storage[1::2], -99)
    np.testing.assert_array_equal(result, expected)
    assert [call[0] for call in calls] == [0, 1, 2] * (2 if mode == "evaluate" else 1)
    if mode in ("out", "strided-out"):
        assert len({call[1] for call in calls[:3]}) == 1
        assert not np.shares_memory(result, buffers[0])
    else:
        assert np.shares_memory(result, buffers[0])
    assert calls[0][2] == 128 * np.dtype(dtype).itemsize + blosc2.MAX_OVERHEAD


def test_numpy_prefilter_non_aligned_blocks_fall_back():
    values = np.arange(257, dtype=np.float64)
    inputs = {"x": blosc2.asarray(values, chunks=(100,), blocks=(32,))}
    out = np.empty_like(values)
    result = blosc2.evaluate("x + 1", local_dict=inputs, out=out)
    assert result is out
    np.testing.assert_array_equal(result, values + 1)


def test_numpy_prefilter_multidimensional_output_falls_back():
    values = np.arange(99, dtype=np.float64).reshape(9, 11)
    inputs = {"x": blosc2.asarray(values, chunks=(4, 6), blocks=(2, 3))}
    out = np.empty_like(values)
    result = blosc2.evaluate("x + 1", local_dict=inputs, out=out)
    assert result is out
    np.testing.assert_array_equal(result, values + 1)


def test_evaluate_without_options_preserves_native_eligibility(monkeypatch):
    import importlib

    module = importlib.import_module("blosc2.lazyexpr")
    original = module.lazyexpr
    observed = []

    def lazyexpr(*args, **kwargs):
        observed.append(kwargs["ne_args"])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "lazyexpr", lazyexpr)
    values = np.arange(257, dtype=np.float64)
    np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": values}), values + 1)
    assert observed == [None]
    np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": values}, order="C"), values + 1)
    assert observed[-1] == {"order": "C"}


@pytest.mark.parametrize(("size", "chunks", "blocks"), [(100, 9, 1), (257, 128, 32)])
def test_numpy_prefilter_existing_expression_and_zero_result(size, chunks, blocks):
    values = np.arange(size, dtype=np.float32)
    inputs = {name: blosc2.asarray(values, chunks=(chunks,), blocks=(blocks,)) for name in ("x", "y")}
    out = np.empty_like(values)
    blosc2.evaluate("x + y", local_dict=inputs, out=out)
    expression = blosc2.lazyexpr("x - y", inputs)
    result = blosc2.lazyexpr(expression, operands=inputs, out=out).compute()
    assert result is out
    np.testing.assert_array_equal(result, 0)
    # Fresh calls must also handle all-zero results on the optimized route.
    np.testing.assert_array_equal(blosc2.evaluate("x - y", local_dict=inputs), 0)
