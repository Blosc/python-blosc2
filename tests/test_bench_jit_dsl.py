"""Binding checks for the JIT benchmark; no timing thresholds."""

import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/ndarray/jit-dsl.py"))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mandelbrot_captures_iteration_bound(benchmark, dtype):
    cr = np.linspace(-0.3, 0.1, 64, dtype=dtype)
    ci = np.linspace(-0.1, 0.1, 64, dtype=dtype)
    inputs = {
        "cr": blosc2.asarray(cr, chunks=(64,), blocks=(16,)),
        "ci": blosc2.asarray(ci, chunks=(64,), blocks=(16,)),
    }
    # Distinct factories must retain their own bounds, not a mutable global.
    authors = [(count, benchmark["make_mandelbrot"](count)) for count in (0, 2, 5)]
    with blosc2.jit_options(jit=False):
        for count, author in reversed(authors):
            assert isinstance(author, blosc2.MenudetFunction)
            lazy = author(**inputs)
            assert set(lazy.inputs) == {"cr", "ci"}
            assert lazy.kernel.native_partitionable
            result = lazy.compute()
            assert result.dtype == np.dtype(dtype)
            assert result.shape == cr.shape
            expected_route = "bounded-readers" if blosc2.IS_WASM else "menudet-prefilter"
            assert lazy._source_execution_report["route"] == expected_route
            zr, zi = cr * 0.0, ci * 0.0
            for _ in range(count):
                zr, zi = zr * zr - zi * zi + cr, zr * zi * 2.0 + ci
            np.testing.assert_allclose(result[:], zr * zr + zi * zi, rtol=1e-5, atol=1e-6)


def test_fresh_cache_clears_factory_specialization(benchmark):
    author = benchmark["make_mandelbrot"](2)
    x = np.zeros(16, dtype=np.float64)
    with blosc2.jit_options(jit=False):
        author.specialize(x, x)
        assert author._kernels
        with benchmark["_fresh_tmpdir"](True, author):
            assert not author._kernels
            lazy = author(x, x)
            np.testing.assert_array_equal(lazy[:], x)
