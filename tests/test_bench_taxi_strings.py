"""Public authoring and storage contracts for the Chicago Taxi string benchmark."""

import runpy
from pathlib import Path

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def benchmark():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "bench/chicago-taxi/string-ops.py"))


@pytest.mark.parametrize("compression", ["DEFAULT", "RAW"])
def test_public_string_kernel_and_expression_routes(benchmark, monkeypatch, compression):
    # Cross chunks and physical blocks, including a short final chunk. Include
    # suffix/non-suffix cab matches, empty inputs and a full-width output.
    companies = ["Yellow CAB", "Cab Company", "Taxi Cab Service", " cab", "TAXI", "", "A" * 44, "ÄBC CAB"]
    co = np.resize(np.array(companies, dtype="U44"), 4103)
    pt = np.resize(np.array(["Cash", "Credit Card", "", "UNKNOWN"], dtype="U11"), co.size)
    monkeypatch.setitem(benchmark["blosc2_setup"].__globals__, "CHUNKS", (1024,))
    monkeypatch.setitem(benchmark["blosc2_setup"].__globals__, "BLOCKS", (128,))
    a, b, cp = benchmark["blosc2_setup"](co, pt, benchmark[compression])
    author = benchmark["make_taxi_label"](co.dtype, pt.dtype)
    assert isinstance(author, blosc2.MenudetFunction)
    kernel = author.specialize(a, b)
    assert kernel.output_dtype == np.dtype("U62")
    assert not kernel.has_jit
    result = benchmark["blosc2_kernel"](a, b, cp)
    expected = []
    for company, payment in zip(co.tolist(), pt.tolist(), strict=True):
        company, payment = company.lower(), payment.lower()
        expected.append(
            "cab|" + company.removesuffix(" cab") + "|" + payment
            if " cab" in company
            else "other|" + company + "|" + payment
        )
    np.testing.assert_array_equal(result[:], np.array(expected, dtype="U62"))
    np.testing.assert_array_equal(result[:], benchmark["numpy_kernel"](co, pt))
    assert result.dtype == kernel.output_dtype
    assert result.chunks == a.chunks
    assert result.blocks == a.blocks
    assert result.cparams.clevel == benchmark[compression].clevel
    if blosc2.blosc2_ext.menudet_prefilter_inputs_eligible(
        {"company": a, "ptype": b}, output_dtype=kernel.output_dtype
    ):
        report = benchmark["blosc2_kernel"].execution_report
        assert report["route"] == "menudet-prefilter"
        assert report["evaluated_blocks"] == 33
    for task in ("filter", "transform"):
        result = benchmark[f"blosc2_{task}"](a, b, cp)
        np.testing.assert_array_equal(result[:], benchmark[f"numpy_{task}"](co, pt))


def test_string_author_width_specializations_and_input_contract(benchmark):
    author = benchmark["make_taxi_label"](np.dtype("U36"), np.dtype("U11"))
    assert author is benchmark["make_taxi_label"](np.dtype("U36"), np.dtype("U11"))
    assert author.specialize(
        np.array(["Taxi"], dtype="U36"), np.array(["Cash"], dtype="U11")
    ).output_dtype == np.dtype("U54")
    with pytest.raises(TypeError, match="fixed-width Unicode"):
        benchmark["make_taxi_label"](np.dtype("S36"), np.dtype("U11"))
    with pytest.raises(TypeError, match="expected dtype"):
        author(np.array(["Taxi"], dtype="U4"), np.array(["Cash"], dtype="U11"))


def test_dependency_versions_are_recorded_for_selected_engines(benchmark):
    versions = benchmark["dependency_versions"](["blosc2", "pandas", "polars", "duckdb"])
    assert versions["numpy"] == np.__version__
    assert all(versions[name] for name in ("pyarrow", "blosc2", "pandas", "polars", "duckdb"))


def test_route_control_script_is_loadable():
    route_control = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "bench/chicago-taxi/compare-string-kernel-routes.py")
    )
    assert callable(route_control["main"])
    assert route_control["legacy_label"] is not None


def test_bench_retains_raw_samples(benchmark):
    calls = []

    def compute():
        calls.append(True)
        return np.array(["other|taxi|cash"])

    best, result, samples = benchmark["bench"](compute, (), 2)
    assert len(calls) == 3
    assert len(samples) == 2
    assert best == min(samples)
    np.testing.assert_array_equal(result, ["other|taxi|cash"])
