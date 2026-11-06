"""Representative coordinate/reduction binding checks for the Menudet benchmark."""

import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/ndarray/sum-linear-idx.py"))


def test_menudet_ramp_coordinates_and_reductions(benchmark):
    shape = (7, 9)
    expected = np.arange(63, dtype=np.int64).reshape(shape)
    author = benchmark["kernel_ramp"]
    assert isinstance(author, blosc2.MenudetFunction)
    with blosc2.jit_options(jit=False):
        lazy = author.specialize().lazy({}, shape=shape, partitions=(3, 4))
        storage = {"chunks": (3, 4), "blocks": (1, 2), "cparams": blosc2.CParams(clevel=1)}
        result = lazy.compute(**storage)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result[:], expected)
        assert result.sum() == 63 * 62 // 2
        assert lazy.sum(**storage) == 63 * 62 // 2


def test_benchmark_storage_matches_logical_partitions(benchmark):
    lazy, storage = benchmark["make_lazy"]((31, 37), blosc2.CParams(clevel=1))
    assert lazy.partitions == storage["chunks"]
    assert lazy.shape == (31, 37)
    assert lazy.dtype == np.dtype("int64")
