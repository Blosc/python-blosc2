"""Check the public Menudet constructor benchmark's captures and chunk coordinates."""

import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/ndarray/linear-constructor.py"))


@pytest.mark.parametrize("shape", [(1, 1), (7, 9)])
def test_ramp_matches_linspace_across_chunks(benchmark, shape):
    start, stop = -0.3, 1.7
    count = int(np.prod(shape))
    author = benchmark["make_ramp"](start, stop, count)
    assert isinstance(author, blosc2.MenudetFunction)
    with blosc2.jit_options(jit=False):
        lazy = author.specialize().lazy({}, shape=shape, partitions=(3, 4))
        result = lazy.compute(chunks=(3, 4), blocks=(1, 2))
        assert result.shape == shape
        assert result.dtype == np.dtype("float64")
        np.testing.assert_array_equal(result[:], np.linspace(start, stop, count).reshape(shape))


def test_native_materializer_preserves_coordinates_and_storage(benchmark):
    shape = (7, 9)
    storage = {"chunks": (3, 4), "blocks": (1, 2), "cparams": blosc2.CParams(clevel=1)}
    expected = np.linspace(-0.3, 1.7, 63).reshape(shape)
    with blosc2.jit_options(jit=False):
        kernel = benchmark["make_ramp"](-0.3, 1.7, 63).specialize()
        result, report = benchmark["materialize_ramp"](kernel, shape, storage)
    assert result.chunks == storage["chunks"]
    assert result.blocks == storage["blocks"]
    assert report["route"] == "menudet-prefilter"
    assert report["evaluated_blocks"] > 1
    benchmark["check_result"](result, expected)


def test_measure_validates_before_warm_measurements(benchmark):
    expected = np.zeros((2, 3), dtype=np.float64)
    calls = []

    def compute():
        calls.append(True)
        return expected.copy()

    first, warm = benchmark["measure"](compute, expected, warmups=1, samples=2)
    assert first >= 0
    assert warm >= 0
    assert len(calls) == 5
    calls.clear()
    with pytest.raises(AssertionError):
        benchmark["measure"](compute, np.ones_like(expected), warmups=1, samples=2)
    assert len(calls) == 1
