"""Function contracts and scoped native floating-status integration."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

import blosc2
from tools import menudet_conformance as compat


def kernel(expression, dtype="float64", output=None, version="1.1"):
    author = blosc2.DSLKernel.from_source(f"def k(x):\n    return {expression}\n")
    return blosc2.PortableKernel.from_json(author.export({"x": dtype}, output or dtype, version=version))


@pytest.fixture(autouse=True)
def status_runtime():
    try:
        kernel("isfinite(x)", output="bool")
    except blosc2.PortableArtifactError as error:
        if os.environ.get("MENUDET_NUMPY_FUNCTION_CORPUS"):
            pytest.fail(f"Explicit M4 testing requires updated native function runtime: {error}")
        pytest.skip("Installed native dependency does not implement M4 functions/status")


@pytest.mark.parametrize(
    ("expression", "values", "flags"),
    [
        ("sqrt(x)", [-1.0], 1),
        ("log(x)", [0.0], 2),
        ("exp(x)", [1000.0], 4),
        ("exp(x)", [-1000.0], 8),
        ("sqrt(x)", [4.0], 0),
        ("log(x) + sqrt(x)", [-1.0, 0.0], 3),
    ],
)
def test_status_and_raise_recovery(expression, values, flags):
    k = kernel(expression)
    _values, status = k.evaluate({"x": np.array(values)}, return_status=True)
    assert status == {"supported": True, "flags": flags}
    if flags:
        with pytest.raises(blosc2.PortableArtifactError, match="floating exception") as error:
            k.evaluate({"x": np.array(values)}, fp_errors="raise")
        assert error.value.fp_status == status
    _result, clean = k.evaluate({"x": np.array([1.0])}, return_status=True)
    assert clean["flags"] == 0


def test_mask_and_unselected_where_do_not_report():
    k = kernel("sqrt(x)")
    result, status = k.evaluate_block(
        {"x": np.array([-1.0, 4.0])}, valid_mask=np.array([False, True]), return_status=True
    )
    assert result[1] == 2
    assert status["flags"] == 0
    lazy = kernel("where(x > 0, sqrt(x), 0)")
    result, status = lazy.evaluate({"x": np.array([-1.0, 4.0])}, return_status=True, fp_errors="raise")
    np.testing.assert_array_equal(result, [0, 2])
    assert status["flags"] == 0


def test_clear_and_explicit_block_aggregation():
    k = kernel("log(x)")
    flags = 0
    for values, expected in [([-1.0], 1), ([0.0], 2), ([1.0], 0), ([], 0)]:
        _result, status = k.evaluate({"x": np.array(values)}, return_status=True)
        assert status["flags"] == expected
        flags |= status["flags"]
    assert flags == 3


def test_same_handle_thread_status_isolation():
    k = kernel("log(x)")

    def run(value):
        return k.evaluate({"x": np.array([value])}, return_status=True)[1]["flags"]

    values = [-1.0, 0.0, 1.0] * 64
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(run, values)) == [1, 2, 0] * 64


def test_policy_validation_and_legacy_profile():
    k = kernel("sqrt(x)")
    with pytest.raises(ValueError, match="fp_errors"):
        k.evaluate({"x": np.array([1.0])}, fp_errors="warn")
    old = kernel("round(x)", version="1.0")
    assert old.evaluate({"x": np.array([0.5])})[0] == 1
    with pytest.raises(blosc2.PortableArtifactError, match="unsupported"):
        old.evaluate({"x": np.array([0.5])}, return_status=True)
    new = kernel("round(x)")
    assert new.evaluate({"x": np.array([0.5])})[0] == 0


@pytest.mark.parametrize(
    "name", ["minimum", "maximum", "isfinite", "isinf", "isnan", "signbit", "fabs", "absolute"]
)
def test_new_spellings_do_not_change_legacy_routing(name):
    expression = f"{name}(x, x)" if name in {"minimum", "maximum"} else f"{name}(x)"
    assert blosc2.blosc2_ext.me_output_dtype(expression, {"x": np.dtype("float64")}) is None
    with pytest.raises(blosc2.PortableArtifactError, match="invalid_source"):
        kernel(expression, version="1.0")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_extrema_zero_contract_and_nan_policy(dtype):
    author = blosc2.DSLKernel.from_source("def k(x, y):\n    return minimum(x, y)\n")
    k = blosc2.PortableKernel.from_json(author.export({"x": dtype, "y": dtype}, dtype, version="1.1"))
    result = k.evaluate(
        {
            "x": np.array([0.0, -0.0, np.nan, 1], dtype=dtype),
            "y": np.array([-0.0, 0.0, 1, np.nan], dtype=dtype),
        }
    )
    assert np.signbit(result[:2]).all()
    assert np.isnan(result[2:]).all()
    artifact = json.loads(k.to_json())
    artifact["source"] = artifact["source"].replace("minimum", "fmin")
    fmin = blosc2.PortableKernel.from_json(json.dumps(artifact))
    result = fmin.evaluate(
        {"x": np.array([np.nan, 1], dtype=dtype), "y": np.array([1, np.nan], dtype=dtype)}
    )
    np.testing.assert_array_equal(result, [1, 1])


def test_shared_function_corpus(tmp_path):
    corpus = os.environ.get("MENUDET_NUMPY_FUNCTION_CORPUS")
    runner = os.environ.get("MENUDET_NUMPY_RUNNER")
    if not corpus or not runner:
        pytest.skip("Select authoritative function corpus and native runner explicitly")
    report = compat.paired(json.loads(Path(corpus).read_text()), Path(runner), tmp_path)
    compat.check(report)
    if os.environ.get("MENUDET_REQUIRE_JIT"):
        assert report["eligible_jit_cases"] > 0
    assert not any(row["jit_eligible"] for row in report["requests"]["off"]["native"])
