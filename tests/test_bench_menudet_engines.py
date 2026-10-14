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
    assert tuple(engines) == engine_benchmark["ENGINES"]
    for name, operation in engines.items():
        result = operation()
        if reduc:
            assert np.ndim(result) == 0, name
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12, err_msg=name)
        else:
            if name == "LazyExpr" or (kind == "blosc2" and name == "Menudet"):
                assert isinstance(result, blosc2.NDArray), name
                assert result.cparams.clevel == 0, name
                if kind == "blosc2":
                    assert result.chunks == inputs["x"].chunks, name
                    assert result.blocks == inputs["x"].blocks, name
                result = result[:]
            else:
                assert isinstance(result, np.ndarray), name
            np.testing.assert_array_equal(result, expected, err_msg=name)


@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
def test_evaluate_column_calls_public_api_every_time(engine_benchmark, monkeypatch, kind):
    inputs, expected = engine_benchmark["build_operands"](257, kind)
    calls = []

    def evaluate(expression, *, local_dict):
        calls.append((expression, local_dict))
        return expected.copy()

    monkeypatch.setattr(blosc2, "evaluate", evaluate)
    engines = engine_benchmark["make_engines"](inputs, kind)
    first = engines["evaluate"]()
    second = engines["evaluate"]()
    assert first is not second
    assert calls == [(engine_benchmark["EXPRESSION"], inputs)] * 2
