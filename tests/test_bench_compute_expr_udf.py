"""Representative predicate authoring/binding checks, not a native dtype matrix."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/ndarray/compute_expr_udf.py"))


def test_predicates_boolean_lazy_binding(benchmark):
    x = np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9)
    values = (x, 2 * x - 1, 10 * x)
    arrays = tuple(blosc2.asarray(v, chunks=(4, 6), blocks=(2, 3)) for v in values)
    with blosc2.jit_options(jit=False):
        for case, author in enumerate(benchmark["KERNELS"]):
            assert isinstance(author, blosc2.MenudetFunction)
            lazy = author(*arrays[: case + 1])
            assert isinstance(lazy, blosc2.LazyArray)
            expected = benchmark["reference"](case, values[: case + 1])
            benchmark["check_result"](lazy[:], expected, False)
            benchmark["check_result"](lazy.compute(chunks=(4, 6), blocks=(2, 3)), expected, True)


def test_measure_checks_results_before_warm_runs(benchmark):
    calls = []

    def wrong():
        calls.append(True)
        return np.ones((2, 3), dtype=bool)

    with pytest.raises(AssertionError):
        benchmark["measure"](wrong, np.zeros((2, 3), dtype=bool))
    assert len(calls) == 1
    timing = benchmark["qualified_measure"](wrong, np.zeros((2, 3), dtype=bool))
    assert timing["qualified"] is False
    assert timing["best_s"] is None
    assert timing["samples_s"] == []
    assert len(calls) == 2


def test_measure_samples_and_menudet_report(benchmark):
    expected = np.array([[False, True]])
    timing = benchmark["measure"](lambda: expected.copy(), expected, repeat=2)
    assert len(timing["samples_s"]) == 2
    assert timing["best_s"] == min(timing["samples_s"])
    report = {"route": "bounded-readers"}
    assert benchmark["execution_report"](SimpleNamespace(_source_execution_report=report)) is report
