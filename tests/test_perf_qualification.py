"""Deterministic qualification tests: no wall-clock thresholds or timing gates."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM, reason="Qualification workers require host subprocess execution"
)
HOST_WORKER_TESTS = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Qualification profiles target Linux/macOS runners with native JIT support",
)

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench" / "qualification"


@pytest.fixture(scope="module")
def qualification():
    # The script bootstraps only bench/, just as isolated subprocess workers do.
    spec = importlib.util.spec_from_file_location("qualification_cli", BENCH / "menudet.py")
    module = importlib.util.module_from_spec(spec)
    old_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = old_path


def test_profiles(qualification):
    for name in ("smoke", "observation", "macos-arm"):
        profile = qualification.read_profile(BENCH / "profiles" / f"{name}.json")
        assert not profile["calibrated"]
        assert profile["threads"] in (1, 2)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": 2},
        {"cases": []},
        {"cases": ["unknown"]},
        {"size": 0},
        {"threads": True},
        {"max_memory_bytes": 1},
        {"cases": ["evaluate-comparison-cc-new"] * 2},
    ],
)
def test_bad_profiles(qualification, tmp_path, change):
    profile = json.loads((BENCH / "profiles" / "smoke.json").read_text())
    profile.update(change)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        qualification.read_profile(path)


def test_bindings_and_contracts(qualification):
    from qualification.workloads import CASES, validate_result

    assert len(CASES) >= 12
    for case in CASES.values():
        if case.storage not in ("numpy", "strided"):
            continue
        inputs, expected, destination = case.bind(31, 42)
        again, reference, _ = case.bind(31, 42)
        np.testing.assert_array_equal(inputs["x"], again["x"])
        np.testing.assert_array_equal(expected, reference)
        result = expected.copy() if destination is None else destination
        result[:] = expected
        validate_result(result, expected, destination)
        with pytest.raises(AssertionError):
            validate_result(result.astype("int64"), expected, destination)
        if destination is not None:
            with pytest.raises(AssertionError, match="identity"):
                validate_result(result.copy(), expected, destination)


def test_timer_result_lifetime_and_batch_normalization(qualification):
    from qualification.measurement import Settings, warm_samples

    events = []

    class Result:
        def __del__(self):
            events.append("released")

    ticks = iter(range(0, 10000, 10))

    def clock():
        events.append("clock")
        return next(ticks)

    def call():
        events.append("call")
        return Result()

    samples = warm_samples(call, 10, Settings(1, 2, 0.00002, 10), float("inf"), clock)
    assert samples == [{"elapsed_ns": 20, "iterations": 2, "per_call_ns": 10}] * 2
    assert events == ["clock", "call", "clock", "released"] * 6


def test_budget_exhaustion(qualification):
    from qualification.measurement import BudgetExhausted, Settings, warm_samples

    with pytest.raises(BudgetExhausted):
        warm_samples(lambda: pytest.fail("Executed after budget"), 1, Settings(), 0)


def test_cache_isolation_does_not_configure_auto_route(qualification, monkeypatch, tmp_path):
    import os

    from qualification.measurement import compiler_cache

    monkeypatch.setenv("ME_DSL_JIT_CACHE_DIR", "previous")
    with compiler_cache(tmp_path):
        assert os.environ["ME_DSL_JIT_CACHE_DIR"] == str(tmp_path)
        import blosc2

        assert blosc2.get_jit_options()["cache_dir"] is None
    assert os.environ["ME_DSL_JIT_CACHE_DIR"] == "previous"


@HOST_WORKER_TESTS
def test_auto_and_configured_routes_are_distinct(qualification, tmp_path):
    request = {
        "case_id": "evaluate-polynomial-auto-new",
        "arm": "candidate",
        "block_id": 0,
        "order_index": 0,
        "local_control": True,
        "size": 31,
        "seed": 42,
        "threads": 1,
        "settings": {"warmups": 0, "samples": 1, "target_ms": 0.001},
        "cache_dir": str(tmp_path),
        "budget_seconds": 60,
    }
    row = qualification.worker(request)
    assert row["status"] == "ok", row.get("error")
    assert row["execution"]["route"] == "resident.native_graph"


@pytest.mark.parametrize("kwargs", [{"samples": 0}, {"warmups": -1}, {"target_ms": float("nan")}])
def test_settings(qualification, kwargs):
    with pytest.raises(ValueError):
        qualification.Settings(**kwargs)


def test_import_isolation(qualification, tmp_path):
    qualification.verify_imports([str(tmp_path / "package.py")], tmp_path)
    with pytest.raises(ValueError, match="outside"):
        qualification.verify_imports([str(tmp_path.parent / "foreign.py")], tmp_path)
    qualification.verify_imports([str(tmp_path.parent / "foreign.py")], tmp_path, True)


def test_runtime_stack_drift(qualification):
    previous = [
        {
            "status": "ok",
            "arm": "baseline",
            "fingerprint": {"dependencies": ["blosc2==1", "numpy==2"], "extension_sha256": "a"},
        }
    ]
    row = {
        "status": "ok",
        "arm": "candidate",
        "fingerprint": {"dependencies": ["blosc2==2", "numpy==2"], "extension_sha256": "b"},
    }
    qualification.verify_stack(row, previous)
    row["arm"] = "baseline"
    with pytest.raises(ValueError, match="extension changed"):
        qualification.verify_stack(row, previous)
    row["fingerprint"]["dependencies"] = ["numpy==3"]
    with pytest.raises(ValueError, match="dependency versions"):
        qualification.verify_stack(row, previous)


def test_order_and_pairing(qualification):
    assert list(qualification.execution_order(2, 42)) == [
        (0, ["baseline", "candidate"]),
        (1, ["candidate", "baseline"]),
    ]
    assert list(qualification.execution_order(2, 43))[0][1] == ["candidate", "baseline"]


def timing_blocks(candidate=(100, 100), baseline=(100, 100)):
    return [
        {
            "case_id": "test",
            "arm": arm,
            "block_id": index,
            "status": "ok",
            "warm_samples": [{"per_call_ns": value}] * 5,
        }
        for arm, values in (("baseline", baseline), ("candidate", candidate))
        for index, value in enumerate(values)
    ]


@pytest.mark.parametrize(
    ("candidate", "verdict"),
    [((100, 100), "observed"), ((150, 150), "warning"), ((100, 200), "inconclusive")],
)
def test_comparisons(qualification, candidate, verdict):
    rows = qualification.compare(timing_blocks(candidate), ["test"], 2)
    assert rows[0]["verdict"] == verdict
    assert rows[0]["ratio"] == np.median(candidate) / 100


def test_incomplete_and_failed_not_ranked(qualification):
    blocks = timing_blocks()
    assert qualification.compare(blocks[:-1], ["test"], 2)[0]["ratio"] is None
    blocks[0]["status"] = "failed"
    assert qualification.compare(blocks, ["test"], 2)[0] == {
        "case_id": "test",
        "shape": None,
        "ratio": None,
        "verdict": "failed",
    }


def test_duplicate_process_blocks(qualification):
    blocks = timing_blocks()
    blocks[1]["block_id"] = 0
    with pytest.raises(ValueError, match="identities"):
        qualification.compare(blocks, ["test"], 2)


def test_missing_report_metrics_and_fallback(qualification):
    from qualification.diagnostics import validate_contract
    from qualification.workloads import CASES

    case = CASES["evaluate-polynomial-cc-new"]
    with pytest.raises(AssertionError, match="fallback"):
        validate_contract(case, [], 4096, 1)
    with pytest.raises(AssertionError, match="capability"):
        validate_contract(
            case, [{"route": "resident.configured_map", "engine": "menudet-interpreter"}], 4096, 1
        )


def test_manifest_revisions(qualification, tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"python_revision": "main"}))
    with pytest.raises(ValueError, match="exact"):
        qualification.load_manifest(path)


@HOST_WORKER_TESTS
def test_real_worker_smoke(qualification, tmp_path):
    # Interpreter-independent numerical semantics remain native tests' concern.
    # Here a missing required compiler is a recorded failure, never a skip/pass.
    request = {
        "case_id": "evaluate-polynomial-cc-out",
        "arm": "candidate",
        "block_id": 0,
        "order_index": 0,
        "local_control": True,
        "size": 31,
        "seed": 42,
        "threads": 1,
        "settings": {"warmups": 0, "samples": 1, "target_ms": 0.001},
        "cache_dir": str(tmp_path),
        "budget_seconds": 60,
    }
    row = qualification.worker(request)
    assert row["status"] == "ok", row.get("error")
    assert row["execution"]["route"] == "resident.configured_map"
    assert row["execution"]["effective_compiler"] is None
    assert row["preparation_ns"] is None
    assert len(row["warm_samples"]) == 1


def test_worker_execution_failure_never_replays(qualification, monkeypatch, tmp_path):
    import blosc2

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise ArithmeticError("probe execution failure")

    monkeypatch.setattr(blosc2, "evaluate", fail)
    row = qualification.worker(
        {
            "case_id": "evaluate-polynomial-cc-new",
            "arm": "candidate",
            "block_id": 0,
            "order_index": 0,
            "local_control": True,
            "size": 31,
            "seed": 42,
            "threads": 1,
            "settings": {},
            "cache_dir": str(tmp_path),
            "budget_seconds": 60,
        }
    )
    assert row["status"] == "failed"
    assert "probe execution failure" in row["error"]
    assert calls == [1]
    assert row["warm_samples"] == []


@pytest.mark.parametrize("failure", ["process", "timeout", "missing-interpreter"])
def test_pair_records_failed_process_and_artifacts(qualification, monkeypatch, tmp_path, failure):
    import subprocess

    caches = []

    def run(*args, **kwargs):
        cache = kwargs["env"]["ME_DSL_JIT_CACHE_DIR"]
        assert Path(cache).is_dir()
        assert not list(Path(cache).iterdir())
        caches.append(cache)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 1)
        if failure == "missing-interpreter":
            raise FileNotFoundError("No interpreter")
        return subprocess.CompletedProcess(args, 1, "", "missing API")

    monkeypatch.setattr(qualification.subprocess, "run", run)
    args = argparse.Namespace(
        profile="smoke",
        cases=["evaluate-polynomial-cc-out"],
        threads=1,
        budget_seconds=60,
        blocks=2,
        warmups=0,
        samples=1,
        target_ms=1,
        baseline_python=sys.executable,
        candidate_python=sys.executable,
        baseline_manifest=None,
        candidate_manifest=None,
        local_control=True,
        seed=42,
        output=tmp_path / "result.json",
    )
    result = qualification.run_pair(args)
    expected = "incomplete" if failure == "timeout" else "failed"
    assert result["coverage"] == expected
    assert len(result["blocks"]) == 4
    assert len(set(caches)) == 4
    assert all(row["error"] for row in result["blocks"])
    assert json.loads(args.output.read_text())["schema_version"] == 1
    assert expected in args.output.with_suffix(".md").read_text()


@pytest.mark.parametrize(
    "case_id",
    [
        "authored-slabs-compressed",
        "authored-slabs-numpy",
        "lazyexpr-slabs-compressed",
        "authored-loop-interpreter",
        "authored-loop-cc",
        "evaluate-transcendental-system",
        "lazyexpr-tiny-compressed",
        "lazyexpr-above-compressed",
        "evaluate-strided-fallback",
    ],
)
@HOST_WORKER_TESTS
def test_curated_route_smoke(qualification, tmp_path, case_id):
    row = qualification.worker(
        {
            "case_id": case_id,
            "arm": "candidate",
            "block_id": 0,
            "order_index": 0,
            "local_control": True,
            "size": 65536,
            "seed": 42,
            "threads": 2,
            "settings": {"warmups": 0, "samples": 1, "target_ms": 0.001},
            "cache_dir": str(tmp_path),
            "budget_seconds": 60,
        }
    )
    assert row["status"] == "ok", row.get("error")


def test_installed_wheel_hash_validation(qualification, tmp_path):
    package = tmp_path / "blosc2"
    package.mkdir()
    file = package / "__init__.py"
    file.write_text("test")
    manifest = {"installed_files": {"blosc2/__init__.py": qualification.digest(file)}}
    qualification.verify_installed_wheel(manifest, file)
    file.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        qualification.verify_installed_wheel(manifest, file)
    with pytest.raises(ValueError, match="lacks"):
        qualification.verify_installed_wheel({}, file)


def test_build_native_pins_and_archive_wheel_hashes(qualification, tmp_path):
    import zipfile

    from qualification.build import exact_sha, native_pins, wheel_files

    pins = native_pins((ROOT / "CMakeLists.txt").read_text())
    assert all(len(value) == 40 for value in pins.values())
    with pytest.raises(ValueError, match="exact"):
        exact_sha("main; echo unsafe")
    with pytest.raises(ValueError):
        native_pins(
            "FetchContent_Declare(miniexpr GIT_REPOSITORY url GIT_TAG main)\nset(BLOSC2_BUNDLED_VERSION main)"
        )
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("blosc2/__init__.py", "test")
        archive.writestr("test.dist-info/METADATA", "metadata")
    assert list(wheel_files(wheel)) == ["blosc2/__init__.py"]


def test_ci_resolves_only_exact_safe_inputs(qualification, monkeypatch, tmp_path):
    import platform
    import subprocess

    from qualification.ci import resolve

    monkeypatch.chdir(tmp_path)
    anchor = tmp_path / ".github" / "menudet-performance-baseline.json"
    anchor.parent.mkdir()
    anchor.write_text(json.dumps({"approved_baseline": None, "timing_gates_enabled": False}))
    monkeypatch.setenv("EXPECTED_ARCH", platform.machine())
    monkeypatch.setenv("PROFILE", "observation")
    monkeypatch.setenv("REQUESTED_MODE", "observation")
    monkeypatch.setenv("REQUESTED_BASELINE", "main; echo unsafe")
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "a" * 40)
    with pytest.raises(ValueError, match="exact"):
        resolve(tmp_path / "identity.json")
    monkeypatch.setenv("REQUESTED_BASELINE", "b" * 40)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0))
    resolve(tmp_path / "identity.json")
    result = json.loads((tmp_path / "identity.json").read_text())
    assert result["baseline"] == "b" * 40
    assert result["candidate"] == "a" * 40
    monkeypatch.setenv("REQUESTED_MODE", "calibration")
    resolve(tmp_path / "aa.json")
    result = json.loads((tmp_path / "aa.json").read_text())
    assert result["baseline"] == result["candidate"]
    assert result["baseline_source"] == "independent-build-aa"


def test_schema_and_descriptive_interval(qualification):
    from qualification.report import ratio_interval, validate_document

    with pytest.raises(ValueError, match="schema"):
        validate_document({"schema_version": 2})
    with pytest.raises(ValueError, match="evidence"):
        validate_document({"schema_version": 1, "blocks": [{"status": "ok"}]})
    assert ratio_interval([1, 1]) is None
    assert ratio_interval([2, 2, 2, 2]) == [2, 2]


def test_freshness_requires_two_successful_jobs(qualification):
    import datetime as dt

    from qualification.freshness import complete_run, stale

    run = {"conclusion": "success", "display_title": "Menudet perf observation"}
    assert complete_run(run, [{"conclusion": "success"}] * 2)
    assert not complete_run(run, [{"conclusion": "skipped"}] * 2)
    assert stale(None, dt.datetime.now(dt.UTC))
    assert stale("2026-01-01T00:00:00Z", dt.datetime(2026, 1, 4, tzinfo=dt.UTC))
    assert not stale("2026-01-01T00:00:00Z", dt.datetime(2026, 1, 2, tzinfo=dt.UTC))


def test_calibration_evidence_and_holdout_accounting(qualification):
    from qualification.calibrate import evidence_summary, wilson

    def evidence(identity, control):
        return {
            "allocation_id": identity,
            "schema_version": 1,
            "kind": "calibration",
            "independent_builds": True,
            "coverage": "complete",
            "control": control,
            "sizes": [{"eligible": True, "comparisons": [{"verdict": "observed"}]}],
        }

    result = evidence_summary([evidence("one", "aa")], [evidence("two", "aa")])
    assert result["groups"]["held_out"]["aa_allocations"] == 1
    assert not result["profile_approved"]
    assert wilson(0, 0) is None
    assert wilson(0, 1)[1] > 0.5
    with pytest.raises(ValueError, match="Repeated"):
        evidence_summary([evidence("same", "aa")], [evidence("same", "aa")])


@pytest.mark.parametrize("historical", [False, True])
def test_calibration_rejects_all_ineligible_except_reproduced_control(
    qualification, monkeypatch, tmp_path, historical
):
    from qualification import calibrate

    case_id = "evaluate-polynomial-cc-new"
    blocks = [
        {
            "case_id": case_id,
            "arm": "baseline",
            "status": "ok",
            "execution": {"tasks": 1},
            "workload": {"route": "resident", "storage": "numpy"},
        },
        {
            "case_id": case_id,
            "arm": "candidate",
            "status": "failed" if historical else "ok",
            "phase": "execution-contract",
            "execution": {"tasks": 1},
            "workload": {"route": "resident", "storage": "numpy"},
        },
    ]
    monkeypatch.setattr(
        calibrate,
        "run_pair",
        lambda args: {
            "coverage": "failed" if historical else "complete",
            "blocks": blocks,
            "profile": {"threads": 2},
            "comparisons": [],
            "wall_seconds": 0,
        },
    )
    args = argparse.Namespace(
        control="historical" if historical else "aa",
        historical_label="control",
        local_control=True,
        cases=[case_id],
        profile="observation",
        sizes=[4096],
        budget_seconds=10,
        output=tmp_path / "calibration.json",
    )
    result = calibrate.sweep(args)
    assert result["coverage"] == ("complete" if historical else "incomplete")
    assert result["mechanism_only_minimum_sizes"][case_id] is None
    assert not result["sizes"][0]["eligible"]


def test_confirmation_runs_once_and_stops(qualification, monkeypatch, tmp_path):
    import subprocess

    def run(command, **kwargs):
        request_path, response_path = map(Path, command[-2:])
        request = json.loads(request_path.read_text())
        value = 200 if request["arm"] == "candidate" else 100
        row = {key: request[key] for key in ("case_id", "arm", "block_id", "order_index")}
        row.update(
            status="ok",
            warm_samples=[{"per_call_ns": value, "elapsed_ns": value, "iterations": 1}],
            execution={"engine": "menudet-jit", "route": "resident.configured_map"},
            fingerprint={"dependencies": [], "extension_sha256": "a"},
        )
        response_path.write_text(json.dumps(row))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(qualification.subprocess, "run", run)
    args = argparse.Namespace(
        profile="smoke",
        cases=["evaluate-polynomial-cc-out"],
        threads=1,
        budget_seconds=60,
        blocks=2,
        confirmation_blocks=2,
        warmups=0,
        samples=1,
        target_ms=1,
        baseline_python=sys.executable,
        candidate_python=sys.executable,
        baseline_manifest=None,
        candidate_manifest=None,
        local_control=True,
        seed=42,
        output=tmp_path / "result.json",
    )
    result = qualification.run_pair(args)
    assert result["coverage"] == "complete"
    assert len(result["blocks"]) == 8
    assert result["comparisons"][0]["verdict"] == "warning"
    assert result["escalation"][args.cases[0]]["confirmation"]


def test_resource_fingerprint_never_raises():
    from qualification.resources import cpu_count, memory_limit, physical_memory

    assert cpu_count() >= 1
    assert physical_memory() is None or physical_memory() > 0
    assert memory_limit() is None or memory_limit() > 0
