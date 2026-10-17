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
    pytest.importorskip("numexpr")
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/evaluate-numpy-parity.py"))


def test_benchmark_rejects_wrong_integer_values_before_timing(benchmark, monkeypatch):
    inputs = {"x": np.array([2**60], dtype="int64")}

    def wrong_result(expression, local_dict, out):
        out[...] = local_dict["x"] + 1
        return out

    monkeypatch.setattr(blosc2, "evaluate", wrong_result)
    with pytest.raises(AssertionError):
        benchmark["compare_case"]("x", inputs, 1, 0, random.Random(42))


def test_benchmark_cli_qualifies_explicit_outputs(benchmark, monkeypatch, tmp_path, capsys):
    import numexpr

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
    previous_ne = numexpr.get_num_threads()
    try:
        benchmark["main"]()
    finally:
        blosc2.set_nthreads(previous)
        numexpr.set_num_threads(previous_ne)
    results = json.loads(report.read_text())["results"]
    assert len(results) == 2
    assert all(result["status"] == "qualified" for result in results)
    assert all(result["numexpr"] > 0 and result["blosc2"] > 0 for result in results)
    assert results[-1]["route"] == "numexpr-fallback"
    assert "shared warmed NumPy out" in capsys.readouterr().out
