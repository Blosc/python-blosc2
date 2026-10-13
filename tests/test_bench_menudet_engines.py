"""Scalar and elementwise correctness of the engine comparison benchmark."""

import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def engine_benchmark():
    pytest.importorskip("numexpr")
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench" / "menudet-engines.py"))


@pytest.mark.parametrize("size", [1, 257])
@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
@pytest.mark.parametrize("reduc", [None, "sum", "mean", "min", "max", "any", "all"])
def test_engine_results(engine_benchmark, size, kind, reduc):
    inputs, values = engine_benchmark["build_operands"](size, "numpy")
    expected = getattr(np, reduc)(values) if reduc else values
    if kind == "blosc2":
        # Multiple chunks plus a partial tail for the nontrivial size.
        inputs = {name: blosc2.asarray(value, chunks=(128,), blocks=(32,)) for name, value in inputs.items()}
    engines = engine_benchmark["make_engines"](inputs, kind, reduc)
    for name, operation in engines.items():
        result = operation()
        if reduc:
            assert np.ndim(result) == 0, name
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, err_msg=name)
        else:
            np.testing.assert_array_equal(result, expected, err_msg=name)
