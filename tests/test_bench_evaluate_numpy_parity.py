"""Qualification and output contracts of the resident-output comparison script."""

import json
import os
import random
import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/evaluate-numpy-parity.py"))


def test_benchmark_rejects_wrong_integer_values_before_timing(benchmark, monkeypatch):
    inputs = {"x": np.array([2**60], dtype="int64")}

    def wrong_result(expression, local_dict, out=None):
        if out is None:
            out = np.empty_like(local_dict["x"])
        out[...] = local_dict["x"] + 1
        return out

    monkeypatch.setattr(blosc2, "evaluate", wrong_result)
    with pytest.raises(AssertionError):
        benchmark["compare_case"]("x", inputs, 1, 0, random.Random(42))


def test_benchmark_rejects_engine_promotion_differences(benchmark, monkeypatch):
    from types import SimpleNamespace

    external = SimpleNamespace(evaluate=lambda *args, **kwargs: np.arange(17, dtype="float64"))
    monkeypatch.setitem(benchmark["compare_case"].__globals__, "ne", external)
    x = np.arange(17, dtype="int64")
    result = benchmark["compare_case"]("minimum(x,y)", {"x": x, "y": x + 1}, 1, 0, random.Random(42))
    assert result["numexpr"] is None
    assert "result dtype differs" in result["numexpr_error"]
    assert result["numpy"] > 0
    assert result["blosc2"] > 0


@pytest.mark.parametrize("absent", [False, True])
def test_benchmark_cli_qualifies_explicit_outputs(benchmark, monkeypatch, tmp_path, capsys, absent):
    if absent:
        monkeypatch.setitem(benchmark["main"].__globals__, "ne", None)
    numexpr = benchmark["main"].__globals__["ne"]

    report = tmp_path / "parity.json"
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", os.environ.get("ME_DSL_JIT_COMPILER", "tcc"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate-numpy-parity.py",
            "--items",
            "257",
            "--cases",
            "linear",
            "complex",
            "--samples",
            "1",
            "--warmups",
            "0",
            "--threads",
            "2",
            "--json",
            str(report),
        ],
    )
    previous = blosc2.nthreads
    previous_ne = numexpr.get_num_threads() if numexpr is not None else None
    try:
        benchmark["main"]()
    finally:
        blosc2.set_nthreads(previous)
        if numexpr is not None:
            numexpr.set_num_threads(previous_ne)
    results = json.loads(report.read_text())["results"]
    assert len(results) == 2
    assert all(result["status"] == "qualified" for result in results)
    assert all(result["numpy"] > 0 and result["blosc2"] > 0 for result in results)
    assert all(result["numexpr"] is None or result["numexpr"] > 0 for result in results)
    if absent:
        assert all(result["numexpr"] is None for result in results)
    assert results[-1]["route"] == "numpy/eager"
    assert "shared warmed NumPy out" in capsys.readouterr().out
