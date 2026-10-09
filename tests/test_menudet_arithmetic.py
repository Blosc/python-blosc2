"""NumPy-profile arithmetic, scalar authoring and persistence regressions."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import blosc2
from tools import menudet_conformance as compat


def export(source, inputs, output, **kwargs):
    return blosc2.DSLKernel.from_source(source).export(inputs, output, version="1.1", **kwargs)


@pytest.fixture(autouse=True)
def numpy_runtime():
    try:
        export("def k(x):\n    return x\n", {"x": "int8"}, "int8")
    except blosc2.PortableArtifactError as error:
        if error.status != "unsupported_requirement":
            raise
        if os.environ.get("MENUDET_NUMPY_ARITHMETIC_CORPUS"):
            pytest.fail("Explicit M3 testing requires a rebuilt 1.1 native dependency")
        pytest.skip("Installed native dependency does not implement profile 1.1")


def test_version_transition_and_metadata_only_inference():
    source = "def k(x):\n    return (x + 1) * 2\n"
    author = blosc2.DSLKernel.from_source(source)
    old = blosc2.PortableKernel.from_json(author.export({"x": "int8"}, "int64"))
    new = blosc2.PortableKernel.from_json(export(source, {"x": "int8"}, "int64"))
    assert old.schema_version == "1.0"
    assert old.inferred_dtype is None
    assert new.schema_version == "1.1"
    assert new.inferred_dtype == np.dtype("int8")
    x = np.array([127, -128], dtype="int8")
    with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
        old.evaluate({"x": x})
    with np.errstate(over="ignore"):
        np.testing.assert_array_equal(new.evaluate({"x": x}), ((x + 1) * 2).astype("int64"))
    assert new.to_json() == export(source, {"x": "int8"}, "int64")
    # Semantic identity is explicit: partial upgrades and future profiles reject.
    for schema, language in [("1.0", "1.1"), ("1.1", "1.0"), ("9.0", "9.0")]:
        artifact = json.loads(new.to_json())
        artifact["schema_version"], artifact["language"]["version"] = schema, language
        with pytest.raises(blosc2.PortableArtifactError):
            blosc2.PortableKernel.from_json(json.dumps(artifact))


@pytest.mark.parametrize(
    ("value", "explicit", "expected_dtype"),
    [
        (1, None, "float32"),
        (np.int64(1), None, "float64"),
        (1, "int64", "float64"),
        (np.float64(1), None, "float64"),
    ],
)
def test_weak_and_typed_constants(value, explicit, expected_dtype):
    artifact = export(
        "def k(x, c):\n    return x + c\n",
        {"x": "float32"},
        expected_dtype,
        constants={"c": value},
        capture_dtypes={} if explicit is None else {"c": explicit},
    )
    encoded = json.loads(artifact)["constants"][0]
    assert encoded["category"] == ("weak" if type(value) is int and explicit is None else "typed_scalar")
    kernel = blosc2.PortableKernel.from_json(artifact)
    assert kernel.inferred_dtype == np.dtype(expected_dtype)
    x = np.array([1], dtype="float32")
    np.testing.assert_array_equal(kernel.evaluate({"x": x}), x + (np.int64(value) if explicit else value))


def test_weak_constant_range_and_same_handle_recovery():
    artifact = export("def k(x, c):\n    return x + c\n", {"x": "int8"}, "int8", constants={"c": 128})
    kernel = blosc2.PortableKernel.from_json(artifact)
    for _ in range(3):
        with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
            kernel.evaluate({"x": np.array([1], dtype="int8")})
    # A typed scalar participates in strong promotion before explicit narrowing.
    typed = blosc2.PortableKernel.from_json(
        export("def k(x, c):\n    return x + c\n", {"x": "int8"}, "int8", constants={"c": np.int64(128)})
    )
    assert typed.inferred_dtype == np.dtype("int64")
    assert typed.evaluate({"x": np.array([1], dtype="int8")})[0] == -127


def test_float_cast_failure_then_success():
    kernel = blosc2.PortableKernel.from_json(
        export("def k(x):\n    return int8(x)\n", {"x": "float64"}, "int8")
    )
    for values in [[np.nan], [np.inf], [128.0], [-129.0]]:
        with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
            kernel.evaluate({"x": np.array(values)})
        assert kernel.evaluate({"x": np.array([-1.9])})[0] == -1


def test_construction_is_not_array_narrowing():
    # A weak scalar constructor is checked; conversion of array elements wraps.
    literal = blosc2.PortableKernel.from_json(
        export("def k(x):\n    return x + int8(128)\n", {"x": "int8"}, "int8")
    )
    with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
        literal.evaluate({"x": np.array([1], dtype="int8")})
    array = blosc2.PortableKernel.from_json(
        export("def k(x):\n    return int8(x)\n", {"x": "int64"}, "int8")
    )
    assert array.evaluate({"x": np.array([128], dtype="int64")})[0] == -128


@pytest.mark.parametrize("shape", [(), (1,)])
def test_typed_zero_dimensional_array(shape):
    source = "def k(x, y):\n    return x + y\n"
    kernel = blosc2.PortableKernel.from_json(export(source, {"x": "float32", "y": "float64"}, "float64"))
    assert kernel.inferred_dtype == np.dtype("float64")
    actual = kernel.evaluate({"x": np.ones(shape, dtype="float32"), "y": np.ones(shape, dtype="float64")})
    assert actual.shape == shape
    assert actual.dtype == np.dtype("float64")


def test_cast_policy_rejection_is_metadata_only():
    source = "def k(x):\n    return x\n"
    for policy in ["safe", "same_kind"]:
        with pytest.raises(blosc2.PortableArtifactError, match="cast policy"):
            export(source, {"x": "float64"}, "int8", casting=policy)
    assert blosc2.PortableKernel.from_json(
        export(source, {"x": "int8"}, "int64", casting="safe")
    ).inferred_dtype == np.dtype("int8")


def test_authoring_captured_scalar():
    c = 1

    @blosc2.dsl_kernel
    def k(x):
        return x + c

    artifact = k.export({"x": "float32"}, "float32", version="1.1")
    assert json.loads(artifact)["constants"][0]["category"] == "weak"
    kernel = blosc2.PortableKernel.from_json(artifact)
    assert kernel.inferred_dtype == np.dtype("float32")
    assert kernel.evaluate({"x": np.array([1], dtype="float32")})[0] == 2


def test_local_weak_scalar_and_float_literal_overflow():
    source = "def k(x):\n    c = 1 + 2\n    return x + c\n"
    kernel = blosc2.PortableKernel.from_json(export(source, {"x": "float32"}, "float32"))
    assert kernel.inferred_dtype == np.dtype("float32")
    assert kernel.evaluate({"x": np.array([1], dtype="float32")})[0] == 4
    kernel = blosc2.PortableKernel.from_json(
        export("def k(x):\n    return x + 1e100\n", {"x": "float32"}, "float32")
    )
    assert np.isposinf(kernel.evaluate({"x": np.array([1], dtype="float32")})[0])


def test_persistence_preserves_profile(tmp_path):
    x = blosc2.asarray(np.array([127, -128, 0, 1], dtype="int8"), urlpath=str(tmp_path / "x.b2nd"))
    kernel = blosc2.PortableKernel.from_json(export("def k(x):\n    return x + 1\n", {"x": "int8"}, "int8"))
    lazy = kernel.lazy({"x": x}, partitions=(2,))
    path = str(tmp_path / "recipe.b2nd")
    lazy.save(path)
    restored = blosc2.open(path)
    assert restored.kernel.schema_version == "1.1"
    assert restored.kernel.inferred_dtype == np.dtype("int8")
    assert restored.kernel.to_json() == kernel.to_json()
    np.testing.assert_array_equal(restored[:], np.array([-128, -127, 1, 2], dtype="int8"))


def test_shared_arithmetic_matrix(tmp_path):
    corpus = os.environ.get("MENUDET_NUMPY_ARITHMETIC_CORPUS")
    runner = os.environ.get("MENUDET_NUMPY_RUNNER")
    if not corpus or not runner:
        pytest.skip("Select authoritative arithmetic corpus and native runner explicitly")
    report = compat.paired(json.loads(Path(corpus).read_text()), Path(runner), tmp_path)
    compat.check(report)
    assert all(row["outcome"] == "matching" for row in report["requests"]["off"]["native"])
    if os.environ.get("MENUDET_REQUIRE_JIT"):
        assert report["eligible_jit_cases"] > 0
    assert not any(row["jit_eligible"] for row in report["requests"]["off"]["native"])


@pytest.mark.parametrize("seed", [1729, 20261009])
def test_full_width_seeded_arithmetic(seed, tmp_path):
    from tools.menudet_arithmetic import property_cases

    corpus = os.environ.get("MENUDET_NUMPY_ARITHMETIC_CORPUS")
    runner = os.environ.get("MENUDET_NUMPY_RUNNER")
    if not corpus or not runner:
        pytest.skip("Select authoritative arithmetic corpus and native runner explicitly")
    vectors = json.loads(Path(corpus).read_text())
    vectors["cases"] = property_cases(seed)
    compat.check(compat.paired(vectors, Path(runner), tmp_path))
