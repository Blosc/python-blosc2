#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import blosc2

CORPUS = Path(
    os.environ.get(
        "MINIEXPR_PORTABLE_CORPUS", Path(__file__).resolve().parents[2] / "miniexpr/tests/portable-dsl"
    )
)
ARTIFACTS = CORPUS.parent / "portable-artifacts"
SCALE = 2.0
BIAS = -1.0


@blosc2.dsl_kernel
def affine(x):
    return x * SCALE + BIAS


@pytest.fixture
def native_artifacts():
    from blosc2 import blosc2_ext

    available = getattr(blosc2_ext, "portable_artifact_available", lambda: False)
    if not available():
        pytest.skip("Rebuild with miniexpr artifact support enabled")


def test_export_capture_snapshot(native_artifacts, monkeypatch):
    artifact = affine.export({"x": "float64"}, "float64", metadata={"producer": "test"})
    assert artifact == affine.export({"x": "float64"}, "float64", metadata={"producer": "test"})
    decoded = json.loads(artifact)
    assert "SCALE" not in decoded["source"]
    assert "BIAS" not in decoded["source"]
    assert len(decoded["constants"]) == 2
    monkeypatch.setitem(affine.func.__globals__, "SCALE", 99.0)
    monkeypatch.setitem(affine.func.__globals__, "BIAS", 10.0)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    x = np.arange(600, dtype=np.float64).reshape(20, 30)
    np.testing.assert_array_equal(kernel.evaluate({"x": x}), 2 * x - 1)
    assert not kernel.has_jit
    assert kernel.entry_point == "affine"
    assert kernel.output_dtype == np.dtype("float64")
    assert kernel.input_dtypes == {"x": np.dtype("float64")}
    assert kernel.to_json() == artifact
    with pytest.raises(TypeError):
        kernel.input_dtypes["x"] = np.dtype("int64")


def test_capture_closure_and_frontend_sugar(native_artifacts):
    scale = np.float32(2.0)

    @blosc2.dsl_kernel
    def trig(x):
        return np.sin(x) * scale

    artifact = trig.export({"x": "float32"}, "float32")
    scale = np.float32(99.0)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    x = np.array([0.0, 0.25, 1.0], dtype=np.float32)
    np.testing.assert_allclose(kernel.evaluate({"x": x}), np.sin(x) * 2, rtol=2e-6)
    assert "np.sin" not in kernel.source


def test_capture_collision_and_comments(native_artifacts):
    scale = 2.0

    @blosc2.dsl_kernel
    def collision(x):
        """scale must not be rewritten in strings."""
        _capture_0 = x  # scale stays in this comment
        return _capture_0 * scale

    artifact = collision.export({"x": "float64"}, "float64")
    decoded = json.loads(artifact)
    assert decoded["constants"][0]["name"] == "_capture_1"
    assert "# scale stays in this comment" in decoded["source"]
    assert '"""scale must not be rewritten in strings."""' in decoded["source"]
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    np.testing.assert_array_equal(kernel.evaluate({"x": np.arange(3, dtype=np.float64)}), [0, 2, 4])


@pytest.mark.parametrize(
    ("dtype", "value", "encoding", "encoded"),
    [
        ("bool", True, "boolean", True),
        ("int32", np.int32(-(2**31)), "decimal", "-2147483648"),
        ("int64", 2**63 - 1, "decimal", "9223372036854775807"),
        ("float32", np.float32(-0.0), "ieee754-hex", "80000000"),
        ("float64", -0.0, "ieee754-hex", "8000000000000000"),
        ("float64", np.inf, "ieee754-hex", "7ff0000000000000"),
        ("float64", np.nan, "ieee754-hex", "7ff8000000000000"),
    ],
)
def test_constant_encodings(native_artifacts, dtype, value, encoding, encoded):
    source = "# me:compiler=cc\ndef constant(value):\n    return value\n"
    author = blosc2.DSLKernel.from_source(source)
    artifact = author.export({}, dtype, constants={"value": value})
    scalar = json.loads(artifact)["constants"][0]
    assert scalar == {"name": "value", "dtype": dtype, "encoding": encoding, "value": encoded}
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    assert kernel.source == source
    result = kernel.evaluate({}, shape=(600,))
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(result, np.full(600, value, dtype=dtype))
    if encoded in ("80000000", "8000000000000000"):
        assert np.signbit(result).all()
    assert kernel.evaluate({}, shape=(0,)).shape == (0,)
    with pytest.raises(ValueError, match="explicit shape"):
        kernel.evaluate({})


def test_explicit_capture_dtype(native_artifacts):
    integer = 2

    @blosc2.dsl_kernel
    def k(x):
        return x * integer

    with pytest.raises(blosc2.PortableArtifactError, match="mixed input dtypes"):
        k.export({"x": "float64"}, "float64")
    artifact = k.export({"x": "float64"}, "float64", capture_dtypes={"integer": "float64"})
    np.testing.assert_array_equal(
        blosc2.PortableKernel.from_json(artifact, jit=False).evaluate({"x": np.array([3.0])}), [6.0]
    )
    with pytest.raises(blosc2.PortableArtifactError, match="Unused capture dtype"):
        k.export({"x": "float64"}, "float64", capture_dtypes={"unused": "float64"})


@pytest.mark.parametrize("value", [np.array(2.0), np.arange(2), object(), "hello"])
def test_unsupported_captures(native_artifacts, value):
    capture = value

    @blosc2.dsl_kernel
    def k(x):
        return x + capture

    with pytest.raises(blosc2.PortableArtifactError, match="Capture 'capture'"):
        k.export({"x": "float64"}, "float64")


def test_never_calls_capture_hooks_or_functions(native_artifacts):
    class Trap:
        def __float__(self):
            pytest.fail("capture __float__ executed")

        def item(self):
            pytest.fail("capture item executed")

    capture = Trap()

    @blosc2.dsl_kernel
    def k(x):
        return x + capture

    with pytest.raises(blosc2.PortableArtifactError, match="Capture 'capture'"):
        k.export({"x": "float64"}, "float64", capture_dtypes={"capture": "float64"})

    def callback(x):
        pytest.fail("callback executed")

    @blosc2.dsl_kernel
    def external(x):
        return callback(x)

    with pytest.raises(blosc2.PortableArtifactError, match="callback"):
        external.export({"x": "float64"}, "float64")


def test_shadowed_numpy_alias_is_not_rewritten(native_artifacts):
    @blosc2.dsl_kernel
    def local_alias(x):
        np = x
        return np.sin(x)

    assert "np.sin" in local_alias.dsl_source
    with pytest.raises(blosc2.PortableArtifactError):
        local_alias.export({"x": "float64"}, "float64")


@pytest.mark.parametrize("jit", [False, None, True])
def test_artifact_backend_and_concurrent_calls(native_artifacts, jit):
    from concurrent.futures import ThreadPoolExecutor

    artifact = affine.export({"x": "float64"}, "float64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit is True and not kernel.has_jit:
        pytest.skip("TCC backend not prepared on this host")
    if jit is False:
        assert not kernel.has_jit
    arrays = [np.arange(600, dtype=np.float64) + i for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda array: kernel.evaluate({"x": array}), arrays))
    for array, result in zip(arrays, results, strict=True):
        np.testing.assert_array_equal(result, 2 * array - 1)


def test_capture_range_checks(native_artifacts):
    kernel = blosc2.DSLKernel.from_source("def k(c):\n    return c\n")
    for value, dtype in [(2**63, "int64"), (2**31, "int32"), (2**53 + 1, "float64"), (1e100, "float32")]:
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.export({}, dtype, constants={"c": value}, capture_dtypes={"c": dtype})


def test_nan_payload_and_explicit_float_narrowing(native_artifacts):
    author = blosc2.DSLKernel.from_source("def k(c):\n    return c\n")
    value = np.array([0x7FF8000000000042], dtype=np.uint64).view(np.float64)[0]
    artifact = author.export({}, "float64", constants={"c": value})
    assert json.loads(artifact)["constants"][0]["value"] == "7ff8000000000042"
    value = 1.1
    artifact = author.export({}, "float32", constants={"c": value}, capture_dtypes={"c": "float32"})
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    assert kernel.evaluate({}, shape=())[()] == np.float32(value)


def test_export_bindings_and_metadata_rejections(native_artifacts):
    author = blosc2.DSLKernel.from_source("def k(x):\n    return x\n")
    with pytest.raises(blosc2.PortableArtifactError, match="exactly one"):
        author.export({}, "float64")
    with pytest.raises(blosc2.PortableArtifactError, match="exactly one"):
        author.export({"x": "float64"}, "float64", constants={"x": 1.0})
    with pytest.raises(blosc2.PortableArtifactError) as error:
        author.export({"x": "float64"}, "float64", metadata={"bad": "\x00"})
    assert error.value.status == "invalid_artifact"
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.PortableKernel.from_json(b"{\xff}")
    assert error.value.status == "invalid_artifact"
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.PortableKernel.from_json("{\ud800}")
    assert error.value.status == "invalid_artifact"


def test_unresolved_capture_has_location(native_artifacts):
    @blosc2.dsl_kernel
    def k(x):
        return x + MISSING_PORTABLE_CAPTURE  # noqa: F821

    with pytest.raises(blosc2.PortableArtifactError) as error:
        k.export({"x": "float64"}, "float64")
    assert "MISSING_PORTABLE_CAPTURE" in str(error.value)
    assert error.value.status == "invalid_source"
    assert error.value.line > 0
    assert error.value.column > 0


def test_binding_shapes_and_storage_adaptation(native_artifacts):
    source = "def subtract(x, y):\n    return x - y\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"y": "float64", "x": "float64"}, "float64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    x = np.arange(12, dtype=np.float64).reshape(3, 4)[:, ::2]
    y = np.zeros((3, 2), dtype=">f8")
    np.testing.assert_array_equal(kernel.evaluate({"y": y, "x": x}), x)
    np.testing.assert_array_equal(kernel.evaluate({"y": y, "x": blosc2.asarray(x)}), x)
    with pytest.raises(blosc2.PortableArtifactError, match="dtype"):
        kernel.evaluate({"x": x.astype(np.float32), "y": y})
    with pytest.raises(blosc2.PortableArtifactError, match="shapes"):
        kernel.evaluate({"x": x, "y": y.ravel()})
    with pytest.raises(blosc2.PortableArtifactError, match="Missing or extra"):
        kernel.evaluate({"x": x})
    empty = np.empty((0, 3), dtype=np.float64)
    assert kernel.evaluate({"y": empty, "x": empty}).shape == (0, 3)


def test_import_is_native_only(native_artifacts, monkeypatch):
    if not ARTIFACTS.is_dir():
        pytest.skip("Native fixture checkout unavailable")
    artifact = (ARTIFACTS / "affine.json").read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Python source/JSON preparation was invoked")

    with monkeypatch.context() as context:
        context.setattr(ast, "parse", forbidden)
        context.setattr(json, "loads", forbidden)
        kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
        result = kernel.evaluate({"x": np.array([0.0, 1.0, 2.0, 3.0])})
    np.testing.assert_array_equal(result, [-1, 1, 3, 5])


def test_export_runs_in_standalone_c(native_artifacts, tmp_path):
    runner = Path(
        os.environ.get(
            "MINIEXPR_ARTIFACT_RUNNER", CORPUS.parents[1] / "build-portable/tests/portable_artifact_runner"
        )
    )
    if not runner.is_file():
        pytest.skip("Set MINIEXPR_ARTIFACT_RUNNER to the standalone native runner")
    artifact = tmp_path / "export.json"
    artifact.write_text(affine.export({"x": "float64"}, "float64"))
    result = subprocess.run([str(runner), str(artifact), "off"], check=True, capture_output=True, text=True)
    assert result.stdout.splitlines() == ["jit=0", "-1", "1", "3", "5"]
    # A fresh interpreter imports only blosc2, never this authoring test module.
    program = "import pathlib, blosc2, numpy as np; "
    program += (
        f"k = blosc2.PortableKernel.from_json(pathlib.Path({str(artifact)!r}).read_bytes(), jit=False); "
    )
    program += "print(k.evaluate({'x': np.array([0., 1., 2., 3.])}).tolist())"
    result = subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "[-1.0, 1.0, 3.0, 5.0]"


def test_native_diagnostics_and_runtime_errors(native_artifacts):
    artifact = affine.export({"x": "float64"}, "float64")
    changed = artifact.replace('"schema_version":"0.1"', '"schema_version":"99"')
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.PortableKernel.from_json(changed, jit=False)
    assert error.value.status == "unsupported_requirement"
    changed = artifact.replace('"schema_version":', '"schema_version":"0.1","schema_version":')
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.PortableKernel.from_json(changed, jit=False)
    assert error.value.status == "invalid_artifact"
    source = "def partial(x):\n    if x > 0:\n        return x\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "float64"}, "float64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
    with pytest.raises(blosc2.PortableArtifactError) as error:
        kernel.evaluate({"x": np.array([-1.0])})
    assert error.value.status == "evaluation_error"
    assert error.value.native_status != 0
    assert kernel.evaluate({"x": np.empty(0, dtype=np.float64)}).size == 0


def test_import_missing_native_support(monkeypatch):
    from blosc2 import blosc2_ext

    monkeypatch.delattr(blosc2_ext, "PortableArtifactHandle", raising=False)
    with pytest.raises(NotImplementedError, match="artifact support"):
        blosc2.PortableKernel.from_json("{}")


def test_optional_adapter_availability():
    from blosc2 import blosc2_ext

    available = getattr(blosc2_ext, "portable_artifact_available", lambda: False)()
    if available:
        with pytest.raises(blosc2.PortableArtifactError, match="invalid_artifact"):
            blosc2.PortableKernel.from_json("{}", jit=False)
    else:
        with pytest.raises(NotImplementedError, match="artifact support"):
            blosc2.PortableKernel.from_json("{}", jit=False)
