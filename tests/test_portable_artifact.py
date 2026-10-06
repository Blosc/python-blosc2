#######################################################################
# Copyright (c) 2026, Blosc Development Team <blosc@blosc.org>
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Portable host integration, with separate semantic and execution matrices.

The exhaustive shared DSL corpus is certified by native miniexpr CTest. Here,
boundary values live in compact dtype/operator scenarios; engine/length checks
are independent. Rejected features do not multiply across unused JIT/size axes.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import blosc2

SCALE = 2.0
BIAS = -1.0


def assert_frozen_rejection(source, inputs, output):
    """Check both authoring and native import, without disguising setup errors."""
    info = blosc2.validate_portable_dsl(source, inputs, output)
    assert not info["valid"], info
    assert info["status"] == "unsupported_feature", info
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.DSLKernel.from_source(source).export(inputs, output)
    assert error.value.status == "unsupported_feature"
    # Use a valid envelope, then replace only its source. Import must independently
    # enforce the native frozen boundary even when authoring is bypassed.
    names = ", ".join(inputs)
    identity = f"def k({names}):\n    return {next(iter(inputs))}\n"
    envelope = json.loads(blosc2.DSLKernel.from_source(identity).export(inputs, output))
    envelope["source"] = source
    envelope["entry_point"] = ast.parse(source).body[0].name
    with pytest.raises(blosc2.PortableArtifactError) as error:
        blosc2.PortableKernel.from_json(json.dumps(envelope), jit=False)
    assert error.value.status == "unsupported_requirement"


@pytest.mark.parametrize(
    ("expression", "input_dtype", "output_dtype"),
    [
        ("sin(x)", "float64", "float64"),
        ("cos(x)", "float32", "float32"),
        ("int(x + 0.25)", "float64", "int64"),
        ("float(float(x))", "float32", "float32"),
        ("bool((x + 1.0) - x)", "float32", "bool"),
        ("(x + 1.0) > x", "float32", "bool"),
        ("(x / 2) > 0", "float64", "bool"),
        ("x + 0.5", "int64", "int64"),
        ("x == 9007199254740993.0", "int64", "bool"),
        ("float(x)", "int64", "float32"),
        ("float(-(x * 0.1))", "float32", "float32"),
        ("bool(int(x + 0.25))", "float64", "bool"),
        ("float(int(x) / 2)", "float32", "float32"),
        ("int(x) / 2", "float64", "int64"),
    ],
)
def test_frozen_boundary_without_corpus(native_artifacts, expression, input_dtype, output_dtype):
    assert_frozen_rejection(f"def k(x):\n    return {expression}\n", {"x": input_dtype}, output_dtype)


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

    with pytest.raises(blosc2.PortableArtifactError) as error:
        trig.export({"x": "float32"}, "float32")
    assert error.value.status == "unsupported_feature"


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
    artifact = affine.export({"x": "float64"}, "float64")

    def forbidden(*args, **kwargs):
        pytest.fail("Python source/JSON preparation was invoked")

    with monkeypatch.context() as context:
        context.setattr(ast, "parse", forbidden)
        context.setattr(json, "loads", forbidden)
        kernel = blosc2.PortableKernel.from_json(artifact, jit=False)
        result = kernel.evaluate({"x": np.array([0.0, 1.0, 2.0, 3.0])})
    np.testing.assert_array_equal(result, [-1, 1, 3, 5])


def test_export_runs_in_standalone_c(native_artifacts, tmp_path):
    runner_path = os.environ.get("MINIEXPR_ARTIFACT_RUNNER")
    if not runner_path:
        pytest.skip("Set MINIEXPR_ARTIFACT_RUNNER to the standalone native runner")
    runner = Path(runner_path)
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


# Execution engines and sizes are independent of the numeric scenario matrix.
# Exercise scalar, short-vector, and tile-tail execution, including repeated
# handles, Boolean bytes, float32 rounding, and local negative zero.
@pytest.mark.parametrize(
    ("compiler", "jit"),
    [
        pytest.param("tcc", False, id="interpreter"),
        pytest.param("tcc", True, id="tcc"),
        pytest.param("cc", True, id="cc"),
    ],
)
@pytest.mark.parametrize("count", [1, 2, 257])
def test_numeric_execution_shapes(native_artifacts, compiler, jit, count):
    cases = [
        ("bool", "bool", "    half = x * 0.5\n    return half\n", [True, False], [True, False]),
        (
            "float32",
            "float32",
            "    rounded = float(x) + 0.1\n    return rounded - float(x)\n",
            [2**24, -(2**24)],
            [0.0, 0.0],
        ),
        ("float64", "float64", "    truth = bool(x)\n    return -truth\n", [0.0, 1.0], [-0.0, -1.0]),
    ]
    for input_dtype, output_dtype, body, samples, expected in cases:
        source = f"# me:compiler={compiler}\ndef k(x):\n{body}"
        artifact = blosc2.DSLKernel.from_source(source).export({"x": input_dtype}, output_dtype)
        kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
        assert kernel.has_jit == jit
        for _ in range(2):
            actual = kernel.evaluate({"x": np.resize(np.array(samples, dtype=input_dtype), count)})
            desired = np.resize(np.array(expected, dtype=output_dtype), count)
            np.testing.assert_array_equal(actual, desired)
            if output_dtype == "bool":
                assert np.all(actual.view("uint8") <= 1)
            else:
                np.testing.assert_array_equal(np.signbit(actual), np.signbit(desired))


@pytest.mark.parametrize(
    ("body", "samples"),
    [
        ("    return x + 0.5\n", [True, True]),
        ("    return x * 0.5\n", [False, True]),
        ("    return bool(x + 0.5)\n", [True, True]),
        ("    return (x + x) == 2\n", [False, True]),
        ("    return (x * 0.5) == 0.5\n", [False, True]),
        ("    return int(x + 0.5) == 0\n", [True, False]),
        ("    half = x * 0.5\n    return half == 0.5\n", [False, True]),
        ("    half = x * 0.5\n    return half * 2 == x\n", [True, True]),
        ("    return x - 1\n", [True, False]),
        ("    return -x\n", [False, True]),
        ("    return (x + 9007199254740992) - 9007199254740992\n", [False, True]),
    ],
)
def test_bool_output_numeric_arithmetic(native_artifacts, body, samples):
    compiler, jit, count = "tcc", False, 257
    if "int(x +" in body:
        assert_frozen_rejection(f"# me:compiler={compiler}\ndef k(x):\n{body}", {"x": "bool"}, "bool")
        return
    artifact = blosc2.DSLKernel.from_source(f"# me:compiler={compiler}\ndef k(x):\n{body}").export(
        {"x": "bool"}, "bool"
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    values = np.resize(np.array([False, True]), count)
    expected = np.resize(np.array(samples), count)
    for _ in range(2):
        actual = kernel.evaluate({"x": values})
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.dtype("bool")
        assert np.all(actual.view("uint8") <= 1)


# Test each dtype/operation once, rotating the independent execution paths and
# output widths. Backend/length coverage is tested separately below, not crossed
# with every numeric scenario. Locals and direct expressions both remain covered.
@pytest.mark.parametrize(
    ("input_dtype", "operation", "compiler", "jit", "output_dtype", "local"),
    [
        pytest.param(
            dtype,
            operation,
            "cc" if (d + o) % 3 == 2 else "tcc",
            (d + o) % 3 != 0,
            "float32" if (d + o) % 2 == 0 else "float64",
            (d // 2 + o) % 2 == 1,
            id=f"{dtype}-{operation}",
        )
        for d, dtype in enumerate(["bool", "int32", "int64", "float32", "float64"])
        for o, operation in enumerate(["sum", "difference", "product", "negation"])
    ],
)
def test_bool_cast_floating_arithmetic(
    native_artifacts, compiler, jit, input_dtype, output_dtype, operation, local
):
    count = 257
    expressions = {
        "sum": "bool(x) + bool(x)",
        "difference": "bool(x) - bool(x)",
        "product": "bool(x) * bool(x)",
        "negation": "-bool(x)",
    }
    expression = expressions[operation]
    body = ""
    if local:
        expression = expression.replace("bool(x)", "truth")
        body = "    truth = bool(x)\n"
    source = f"# me:compiler={compiler}\ndef k(x):\n{body}    return {expression}\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": input_dtype}, output_dtype)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    samples = [0, 1, -1, 2]
    if input_dtype == "int64":
        samples += [2**53 + 1, -(2**53 + 1)]
    elif input_dtype.startswith("float"):
        samples += [-0.0, np.finfo(input_dtype).smallest_subnormal, np.inf, -np.inf, np.nan]
    values = np.resize(np.array(samples, dtype=input_dtype), count)
    truth = (values != 0).astype(output_dtype)
    expected = {
        "sum": truth + truth,
        "difference": truth - truth,
        "product": truth * truth,
        "negation": -truth,
    }[operation]
    for _ in range(2):
        actual = kernel.evaluate({"x": values})
        np.testing.assert_array_equal(actual, expected)
        zeros = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("output_dtype", ["float32", "float64"])
def test_float_cast_arithmetic_rounding(native_artifacts, output_dtype):
    compiler, jit, count = "tcc", False, 257
    source = f"# me:compiler={compiler}\ndef k(x):\n    return (float(x) + 0.1) - float(x)\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "float32"}, output_dtype)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    samples = np.array(
        [
            2**24,
            -(2**24),
            0.0,
            -0.0,
            np.finfo("float32").smallest_subnormal,
            np.finfo("float32").max,
            np.inf,
            -np.inf,
            np.nan,
            0.1,
        ],
        dtype="float32",
    )
    values = np.resize(samples, count)
    computation = values.astype(output_dtype)
    constant = np.array(0.1, dtype=output_dtype)
    with np.errstate(all="ignore"):
        expected = (computation + constant) - computation
    for _ in range(2):
        actual = kernel.evaluate({"x": values})
        np.testing.assert_array_equal(actual, expected)
        zeros = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("output_dtype", ["float32", "float64"])
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply"])
def test_float_arithmetic_constant_rounding(native_artifacts, output_dtype, operation):
    compiler, jit, count = "cc", True, 257
    expressions = {
        "add": "(x + 0.1) - x",
        "subtract": "x - 0.1",
        "multiply": "(x * 0.1) - x",
    }
    source = f"# me:compiler={compiler}\n# me:fp=strict\ndef k(x):\n    return {expressions[operation]}\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "float32"}, output_dtype)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    values = np.resize(np.array([0.1, -1, 0, 1, 16777216], dtype="float32"), count)
    # These expressions contain a floating literal: native compilation types
    # that constant from the requested floating output context. Each operation
    # then rounds its operands/result in that context, not C's double literal type.
    arithmetic_values = values.astype(output_dtype)
    constant = np.array(0.1, dtype=output_dtype)
    if operation == "add":
        expected = (arithmetic_values + constant) - arithmetic_values
    elif operation == "subtract":
        expected = arithmetic_values - constant
    else:
        expected = (arithmetic_values * constant) - arithmetic_values
    for _ in range(2):
        np.testing.assert_array_equal(kernel.evaluate({"x": values}), expected)


@pytest.mark.parametrize(("compiler", "jit"), [("tcc", False), ("tcc", True), ("cc", True)])
@pytest.mark.parametrize("cap", ["3", "0", "-1", "invalid"])
def test_while_cap_policy(native_artifacts, monkeypatch, compiler, jit, cap):
    count = 257
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", cap)
    source = f"# me:compiler={compiler}\ndef k(x):\n    n = 0\n    while n < x:\n        n = n + 1\n    return n\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "int64"}, "int64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    for target in [0, 2, 3, 4]:
        values = np.full(count, target, dtype="int64")
        if cap == "3" and target == 4:
            with pytest.raises(blosc2.PortableArtifactError) as error:
                kernel.evaluate({"x": values})
            assert error.value.status == "evaluation_error"
        else:
            np.testing.assert_array_equal(kernel.evaluate({"x": values}), values)
    # Failed output contents are unspecified; the same prepared handle remains usable.
    values = np.full(count, 2, dtype="int64")
    np.testing.assert_array_equal(kernel.evaluate({"x": values}), values)
    assert kernel.evaluate({"x": np.empty(0, dtype="int64")}).size == 0


@pytest.mark.parametrize(("compiler", "jit"), [("tcc", False), ("tcc", True), ("cc", True)])
@pytest.mark.parametrize("compile_cap", ["0", "3"])
def test_while_cap_change_after_loading(native_artifacts, monkeypatch, compiler, jit, compile_cap):
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", compile_cap)
    source = f"# me:compiler={compiler}\ndef k(x):\n    n = 0\n    while n < x:\n        n = n + 1\n    return n\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "int64"}, "int64")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    values = np.array([4], dtype="int64")
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "2")
    with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
        kernel.evaluate({"x": values})
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "4")
    np.testing.assert_array_equal(kernel.evaluate({"x": values}), values)
    # The identical source loaded under a different cap must not reuse code
    # compiled with the old cap from the shared JIT cache.
    new_kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert new_kernel.has_jit
    np.testing.assert_array_equal(new_kernel.evaluate({"x": values}), values)
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    with pytest.raises(blosc2.PortableArtifactError, match="evaluation_error"):
        kernel.evaluate({"x": values})


@pytest.mark.parametrize(
    ("dtype", "compiler", "jit"),
    [("int32", "tcc", False), ("int64", "tcc", True), ("float32", "cc", True), ("float64", "tcc", True)],
)
def test_while_chain_mixed_lane(native_artifacts, monkeypatch, compiler, jit, dtype):
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    source = (
        f"# me:compiler={compiler}\ndef while_cap_chain(x):\n"
        "    n = 0\n    while 0 <= n < x:\n        n = n + 1\n    return n\n"
    )
    artifact = blosc2.DSLKernel.from_source(source).export({"x": dtype}, dtype)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    # Lane ordering is the actual regression boundary, not every size/backend
    # combination. Keep both first-lane states in each dtype scenario.
    for samples in [[0, 2, 3, -1], [3, 2, 0, -1]]:
        values = np.resize(np.array(samples, dtype=dtype), 257)
        expected = np.maximum(values, np.array(0, dtype=dtype))
        for _ in range(2):
            np.testing.assert_array_equal(kernel.evaluate({"x": values}), expected)


@pytest.mark.parametrize(
    ("dtype", "compiler", "jit"),
    [("int32", "tcc", False), ("int64", "tcc", True), ("float32", "cc", True), ("float64", "tcc", True)],
)
@pytest.mark.parametrize("case", ["masked_local_chain", "masked_bool_chain"])
def test_masked_local_constant_chain(native_artifacts, compiler, jit, dtype, case):
    count = 257
    bodies = {
        "masked_local_chain": "    n = 0\n    if x == 0:\n        n = 1\n    return 0 <= n < x\n",
        "masked_bool_chain": (
            "    flag = bool(0)\n    if x == 0:\n        flag = bool(1)\n"
            "    return bool(0) <= flag < bool(x)\n"
        ),
    }
    source = f"# me:compiler={compiler}\ndef {case}(x):\n{bodies[case]}"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": dtype}, "bool")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    values = np.resize(np.array([0, 1, 2, -1], dtype=dtype), count)
    expected = np.resize(np.array([False, True, True, case == "masked_bool_chain"]), count)
    for _ in range(2):
        np.testing.assert_array_equal(kernel.evaluate({"x": values}), expected)


# Cap policy and post-load changes above cover all engines. These scenarios
# isolate control-flow semantics rather than crossing the same policy again.
@pytest.mark.parametrize(
    ("body", "expected", "compiler", "jit"),
    [
        ("    n = 0\n    while 0 <= n < 3:\n        n = n + 1\n    return n\n", 3, "tcc", False),
        (
            "    n = 0\n    while x:\n        n = n + 1\n        if n == 3:\n            break\n    return n\n",
            3,
            "tcc",
            True,
        ),
        (
            "    n = 0\n    while x:\n        n = n + 1\n        if n == 3:\n            return n\n    return n\n",
            3,
            "cc",
            True,
        ),
        (
            "    total = 0\n    for i in range(2):\n        n = 0\n        while n < 3:\n"
            "            n = n + 1\n            total = total + 1\n    return total\n",
            6,
            "tcc",
            True,
        ),
        (
            "    total = 0\n    n = 0\n    while n < 3:\n        n = n + 1\n        m = 0\n"
            "        while m < 3:\n            m = m + 1\n            total = total + 1\n    return total\n",
            9,
            "cc",
            True,
        ),
        ("    if x > 1:\n        while x:\n            pass\n    return x\n", 1, "tcc", False),
    ],
)
def test_while_cap_control_flow(native_artifacts, monkeypatch, compiler, jit, body, expected):
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    artifact = blosc2.DSLKernel.from_source(f"# me:compiler={compiler}\ndef k(x):\n{body}").export(
        {"x": "int64"}, "int64"
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    if jit:
        assert kernel.has_jit
    np.testing.assert_array_equal(
        kernel.evaluate({"x": np.ones(5, dtype="int64")}), np.full(5, expected, dtype="int64")
    )


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("expression", ["int(x)", "bool(x)"])
def test_value_cast_large_integer(native_artifacts, jit, expression):
    source = f"# me:compiler=tcc\ndef k(x):\n    return {expression}\n"
    output_dtype = "int64" if expression == "int(x)" else "bool"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "int64"}, output_dtype)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    values = np.array([-(2**53 + 1), 0, 2**53 + 1, 2**63 - 1], dtype="int64")
    np.testing.assert_array_equal(kernel.evaluate({"x": values}), values.astype(output_dtype))


@pytest.mark.parametrize("jit", [False, True])
def test_value_cast_nonfinite_truth(native_artifacts, jit):
    source = "# me:compiler=tcc\ndef k(x):\n    return bool(x)\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "float64"}, "bool")
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    values = np.array([-0.0, 0.0, np.nan, np.inf, -np.inf, 5e-324])
    np.testing.assert_array_equal(kernel.evaluate({"x": values}), values.astype("bool"))
