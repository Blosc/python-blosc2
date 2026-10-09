"""M2 comparator, layout, minimizer and shared native corpus regressions."""

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tools import menudet_conformance as compat


@pytest.fixture
def native_inputs():
    corpus = os.environ.get("MENUDET_NUMPY_CORPUS_V2")
    runner = os.environ.get("MENUDET_NUMPY_RUNNER")
    if not corpus or not runner:
        pytest.skip("Select authoritative v2 corpus and native runner explicitly")
    return json.loads(Path(corpus).read_text()), Path(runner)


@pytest.mark.parametrize("kind", ["bitwise", "exact", "ulp", "tolerance"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_comparison_policies(kind, dtype):
    expected = np.array([1], dtype=dtype)
    adjacent = np.nextafter(expected, np.array([2], dtype=dtype))
    policy = {**compat.BITWISE, "kind": kind, "max_ulp": 1}
    errors = compat.compare(compat.describe(adjacent), compat.describe(expected), policy)
    assert bool(errors) == (kind in {"bitwise", "exact"})
    policy = {**compat.BITWISE, "kind": "tolerance", "atol": 1e-3}
    assert not compat.compare(
        compat.describe(expected + np.array([1e-4], dtype=dtype)), compat.describe(expected), policy
    )
    assert compat.compare(compat.describe(expected + 0.1), compat.describe(expected), policy)


def test_nan_payload_and_signed_zero_policies():
    positive = compat.describe(np.array([0.0]))
    negative = compat.describe(np.array([-0.0]))
    assert compat.compare(positive, negative, compat.BITWISE)
    assert not compat.compare(positive, negative, {**compat.BITWISE, "signed_zero": "ignore"})
    first = {"dtype": "float64", "shape": [1], "encoding": "raw-be-hex", "hex": "7ff8000000000001"}
    second = {**first, "hex": "7ff8000000000002"}
    assert compat.compare(first, second, compat.BITWISE)
    assert not compat.compare(first, second, {**compat.BITWISE, "nan": "equal"})
    assert compat.compare(
        compat.describe(np.array([np.inf])), compat.describe(np.array([-np.inf])), compat.BITWISE
    )


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("byteorder", ["native", "little", "big"])
def test_layout_recipes(order, byteorder):
    array = np.arange(12, dtype="int64").reshape(3, 4)
    item = compat.describe(
        array, "x", {"order": order, "byteorder": byteorder, "step": 2, "reverse_axis": 0}
    )
    actual = compat.decode(item)
    np.testing.assert_array_equal(actual, array)
    assert actual.strides[0] < 0
    assert not actual.flags.c_contiguous


@pytest.mark.parametrize(
    ("machine", "key"),
    [("AMD64", "x86_64"), ("x86_64", "x86_64"), ("aarch64", "arm64"), ("arm64", "arm64")],
)
def test_platform_qualified_reference_is_exact(machine, key, monkeypatch):
    arm = compat.describe(np.array([2**63 - 1], dtype="int64"))
    x64 = compat.describe(np.array([-(2**63)], dtype="int64"))
    case = {
        "expected": arm,
        "platform_qualification": "Host-specific nonfinite cast sentinel",
        "reference_expected_by_machine": {"arm64": arm, "x86_64": x64},
        "baseline": {"status": -5, "category": "evaluation_error"},
    }
    monkeypatch.setattr(compat.platform, "machine", lambda: machine)
    expected = compat.reference_expected(case)
    assert expected == case["reference_expected_by_machine"][key]
    assert compat.compare(x64 if key == "arm64" else arm, expected, compat.BITWISE)
    assert case["baseline"] == {"status": -5, "category": "evaluation_error"}
    monkeypatch.setattr(compat.platform, "machine", lambda: "unlisted")
    assert compat.reference_expected(case) == arm
    del case["platform_qualification"]
    with pytest.raises(ValueError, match="platform qualification"):
        compat.reference_expected(case)


def test_seeded_generator_and_minimization():
    assert compat.property_cases(1729) == compat.property_cases(1729)
    failure = compat.property_cases(1729)[-1]

    def fails(case):
        # Reference predicate for testing the generic minimizer, not a second
        # execution backend or an arithmetic contract used by production code.
        arrays = [compat.checkpoint.decode(item).astype("int64") for item in case["inputs"]]
        return bool(np.any(arrays[0] + arrays[1] > 127))

    reduced = compat.minimized_failure(failure, fails)
    assert [item["hex"] for item in reduced["inputs"]] == ["7f", "01"]
    with pytest.raises(ValueError, match="passing case"):
        compat.minimized_failure(compat.property_cases(1729)[0], fails)


def test_paired_corpus(native_inputs, tmp_path, monkeypatch):
    import numexpr

    corpus, runner = native_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("No NumExpr fallback in portable conformance")

    monkeypatch.setattr(numexpr, "evaluate", forbidden)
    report = compat.paired(corpus, runner, tmp_path)
    compat.check(report)
    assert report["eligible_jit_cases"] == 0
    assert report["reference_drift"] == []
    for mode in ("off", "on"):
        rows = report["requests"][mode]["native"]
        assert len(rows) == len(corpus["cases"])
        assert sum(row["outcome"] == "skipped" for row in rows) == 2
        assert all(row.get("environment_restored", True) for row in rows)
        assert all(row.get("recovery_ok", True) for row in rows)


@pytest.mark.parametrize("kind", ["bitwise", "exact", "ulp", "tolerance"])
def test_native_python_comparator_parity(native_inputs, tmp_path, kind):
    corpus, runner = native_inputs
    actual = np.array([1.0])
    expected = np.nextafter(actual, np.array([2.0]))
    case = compat.make_case(
        "comparator-oracle", "identity", "x", [actual], policy={**compat.BITWISE, "kind": kind, "max_ulp": 1}
    )
    case["expected"] = compat.describe(expected)
    test = {**corpus, "cases": [case]}
    row = compat.native(test, runner, tmp_path, observe=True)[0]
    assert row["reference_match"] == (
        not compat.compare(compat.describe(actual), case["expected"], case["comparison"])
    )


@pytest.mark.parametrize(
    "corruption", ["nan-policy", "negative-tolerance", "encoding", "layout", "extent", "baseline"]
)
def test_native_rejects_or_reports_corruption(native_inputs, tmp_path, corruption):
    corpus, runner = native_inputs
    case = copy.deepcopy(next(case for case in corpus["cases"] if case["id"] == "float32-weak-literal"))
    corpus = {**corpus, "cases": [case]}
    if corruption == "nan-policy":
        case["comparison"]["nan"] = "future"
    elif corruption == "negative-tolerance":
        case["comparison"]["atol"] = -1
    elif corruption == "encoding":
        case["inputs"][0]["hex"] = "zz"
    elif corruption == "layout":
        case["inputs"][0]["layout"]["reverse_axis"] = 8
    elif corruption == "extent":
        case["inputs"][0]["shape"] = [2**63]
    else:
        case["baseline"]["output"]["hex"] = "3f8000003f800000"
    path = tmp_path / "corrupt.json"
    path.write_text(json.dumps(corpus))
    proc = subprocess.run([str(runner), str(path), "off"], capture_output=True, text=True, check=False)
    assert proc.returncode == 1
    if corruption == "baseline":
        assert json.loads(proc.stdout)["outcome"] == "regression"


def test_seeded_native_properties(native_inputs, tmp_path):
    corpus, runner = native_inputs
    report = compat.properties(corpus, runner, tmp_path, [1729])
    assert report["results"][0]["matching"] == 48
    assert report["results"][0]["promoted_case_id"] == "seed-1729-minimized-int8-overflow"
