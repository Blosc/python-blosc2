"""Qualification probes validate results and expose dispatch, not just timings."""

import json
import runpy
from pathlib import Path

import pytest

import blosc2

pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="Host allocation/timing diagnostic")


def test_fallback_probe_reports_correctness_routes_and_memory(monkeypatch, capsys):
    benchmark = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "bench/ndarray/evaluator_fallbacks.py")
    )
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", "typed")
    monkeypatch.setattr("sys.argv", ["evaluator_fallbacks.py", "--side", "16", "--rounds", "1"])
    previous_threads = blosc2.nthreads
    try:
        benchmark["main"]()
    finally:
        blosc2.set_nthreads(previous_threads)
    report = json.loads(capsys.readouterr().out)
    assert len(report["rows"]) == 11
    for row in report["rows"]:
        assert row["correct"]
        for mode, data in row["modes"].items():
            assert data["median_ms"] >= 0
            assert data["tracked_peak_bytes"] > 0
            assert "numpy_evaluations" in data["dispatch_calls"]
            if mode == "typed":
                assert data["dispatch_calls"]["uniform_prefilters"] == 0
    import os

    assert os.environ["BLOSC2_EXPRESSION_EVALUATOR"] == "typed"
