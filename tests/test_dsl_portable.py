"""Initial raw-source portable conformance; the corpus is owned by miniexpr."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.dsl_kernel import DSLKernel

CORPUS = Path(
    os.environ.get(
        "MINIEXPR_PORTABLE_CORPUS",
        Path(__file__).resolve().parents[2] / "miniexpr" / "tests" / "portable-dsl",
    )
)


CASES = [
    "affine",
    "chains",
    "loops",
    "int32",
    "int64",
    "float32",
    "bool",
    "compare",
    "special_floats",
    "flow",
    "boolean_results",
    "cast_int",
    "math",
    "division_int",
    "division_bool",
    "division_literals",
    "division_locals",
    "division_loop",
    "division_float32",
    "division_condition",
    "cast_argument_int",
    "cast_argument_bool",
    "cast_argument_nested",
    "cast_integer_exact",
    "cast_nonfinite_truth",
    "math_widen",
    "math_condition",
    "math_local_widen",
    "arithmetic_round",
    "arithmetic_decimal",
    "arithmetic_scalar_sub",
    "arithmetic_local_round",
    "arithmetic_widen",
    "while_cap_exact",
    "while_cap_exceeded",
    "while_cap_continue",
    "while_cap_hybrid",
    "while_cap_chain",
    "masked_local_chain",
    "masked_bool_chain",
    "unknown_name",
    "invalid_index",
    "zero_step",
    "missing_return",
    "missing_return_ok",
    "missing_return_mixed",
    "missing_return_loop",
    "missing_return_loop_ok",
    "missing_return_hybrid",
    "missing_return_hybrid_ok",
]
TYPES = ("bool", "int32", "int64", "float32", "float64")
CASES += [f"convert_{input_type}_{output_type}" for input_type in TYPES for output_type in TYPES]


def corpus_source(case):
    return CORPUS / ("identity.dsl" if case.startswith("convert_") else f"{case}.dsl")


def typed_values(values, dtype):
    # NumPy treats nonempty strings as True; fixture Boolean tokens are 0/1.
    return np.array([int(value) for value in values] if dtype == np.dtype("bool") else values, dtype=dtype)


def assert_typed_result(actual, expected, *, exact=False):
    assert actual.dtype == expected.dtype
    if expected.dtype.kind in "biu" or exact:
        np.testing.assert_array_equal(actual, expected)
    if expected.dtype.kind == "f":
        if not exact:
            tolerance = 1e-6 if expected.dtype == np.dtype("float32") else 1e-12
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
        zeros = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("jit", [False, True])
def test_native_source_corpus(case, jit, monkeypatch):
    if case.startswith("while_cap_"):
        monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    if not CORPUS.is_dir():
        pytest.skip("Set MINIEXPR_PORTABLE_CORPUS to the native miniexpr fixture directory")
    source = corpus_source(case).read_text()
    words = (CORPUS / f"{case}.txt").read_text().split()
    outcome, input_type, output_type = words[:3]
    count, nvars = map(int, words[3:5])
    names = words[5 : 5 + nvars]
    rows = np.array(words[5 + nvars :]).reshape(count, nvars + 1)
    input_dtype, output_dtype = np.dtype(input_type), np.dtype(output_type)
    expected = typed_values(rows[:, -1], output_dtype)
    kernel = DSLKernel.from_source(source)
    by_name = {name: typed_values(rows[:, index], input_dtype) for index, name in enumerate(names)}
    inputs = [by_name[name] for name in kernel.input_names]
    if outcome in {"ok", "ok_exact"}:
        result = blosc2.lazyudf(kernel, inputs, dtype=output_dtype, jit=jit)[:]
        assert_typed_result(result, expected, exact=outcome == "ok_exact")
    else:
        with pytest.raises((NotImplementedError, RuntimeError)):
            blosc2.lazyudf(kernel, inputs, dtype=output_dtype, jit=jit)[:]
    assert kernel.dsl_source == source
    assert kernel.func is None

    runner = os.environ.get("MINIEXPR_PORTABLE_RUNNER")
    if runner:
        completed = subprocess.run(
            [
                runner,
                str(corpus_source(case)),
                str(CORPUS / f"{case}.txt"),
                "on" if jit else "off",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        if outcome == "compile_error":
            assert lines == [outcome]
        else:
            assert lines[0] == f"jit={int(jit)}"
            if outcome == "eval_error":
                assert lines[1:] == [outcome]
            else:
                assert_typed_result(
                    typed_values(lines[1:], output_dtype), expected, exact=outcome == "ok_exact"
                )


def test_from_source_never_executes_python(tmp_path):
    target = tmp_path / "executed"
    with pytest.raises(ValueError, match="start with a function"):
        DSLKernel.from_source(f"open({str(target)!r}, 'w')\ndef kernel(x):\n    return x\n")
    with pytest.raises(ValueError, match="plain positional"):
        DSLKernel.from_source(f"def kernel(x=open({str(target)!r}, 'w')):\n    return x\n")
    assert not target.exists()


@pytest.mark.parametrize(
    "source", ["", "def broken(", "def k(x, x):\n    return x", "def k(x):\n    return x\x00"]
)
def test_from_source_invalid_header(source):
    with pytest.raises(ValueError):
        DSLKernel.from_source(source)


def test_from_source_native_body_and_no_python_fallback():
    source = "# me:compiler=tcc\ndef kernel(x):\n    return !x\n"
    kernel = DSLKernel.from_source(source)
    assert kernel.dsl_source == source
    assert kernel.input_names == ["x"]
    with pytest.raises(RuntimeError, match="cannot execute as a Python"):
        kernel((np.array([1.0]),), np.empty(1))


@pytest.mark.parametrize("jit", [False, True])
def test_from_source_without_external_corpus(jit):
    source = "# me:compiler=tcc\ndef kernel(x):\n    return !x\n"
    kernel = blosc2.DSLKernel.from_source(source)
    result = blosc2.lazyudf(kernel, (np.array([0.0, 1.0, -1.0]),), dtype=np.bool_, jit=jit)[:]
    np.testing.assert_array_equal(result, [True, False, False])


def test_from_source_invalid_native_body():
    kernel = DSLKernel.from_source("def kernel(x):\n    return x[0]\n")
    with pytest.raises((NotImplementedError, RuntimeError)):
        blosc2.lazyudf(kernel, (np.arange(4, dtype=np.float64),), dtype=np.float64)[:]


@pytest.fixture
def portable_validator():
    info = blosc2.validate_portable_dsl("def k(x):\n    return x\n", {"x": "float64"}, "float64")
    if info["status"] == "runtime_unsupported":
        pytest.skip("Rebuild with miniexpr portable validation support")
    assert info == {"valid": True, "status": "success", "line": 0, "column": 0, "error": None}
    return blosc2.validate_portable_dsl


@pytest.mark.parametrize("case", CASES)
def test_portable_profile_corpus(portable_validator, case):
    if not CORPUS.is_dir():
        pytest.skip("Set MINIEXPR_PORTABLE_CORPUS to the native miniexpr fixture directory")
    words = (CORPUS / f"{case}.txt").read_text().split()
    outcome, input_type, output_type = words[:3]
    nvars = int(words[4])
    inputs = dict.fromkeys(words[5 : 5 + nvars], input_type)
    info = portable_validator(corpus_source(case).read_text(), inputs, output_type)
    assert info["valid"] == (outcome != "compile_error"), info


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ("return sum(x)", "unsupported_feature"),
        ("return np.sin(x)", "unsupported_feature"),
        ("return x.lower()", "unsupported_feature"),
        ("return x[0]", "unsupported_feature"),
        ("return _flat_idx + x", "unsupported_feature"),
        ("return callback(x)", "unsupported_feature"),
        ("return CAPTURE + x", "invalid_source"),
        ("return sin()", "invalid_source"),
    ],
)
def test_portable_profile_rejections(portable_validator, body, status):
    info = portable_validator(f"def k(x):\n    {body}\n", {"x": "float64"}, "float64")
    assert not info["valid"]
    assert info["status"] == status
    assert info["error"]
    assert info["line"] == 2
    assert info["column"] > 0


def test_portable_profile_signature_and_version(portable_validator):
    source = "def k(x, y):\n    return x + y\n"
    inputs = {"y": "float64", "x": "float64"}
    assert portable_validator(source, inputs, "float64")["valid"]
    assert portable_validator(source, inputs, "float64", language_version="99")["status"] == (
        "unsupported_version"
    )
    assert portable_validator(source, {"x": "float64"}, "float64")["status"] == "invalid_signature"
    assert portable_validator(source, {"x": "float64", "y": "int32"}, "float64")["status"] == (
        "unsupported_feature"
    )
    assert portable_validator(source, inputs, "complex128")["status"] == "unsupported_feature"


def test_portable_profile_never_executes_or_compiles_jit(portable_validator, monkeypatch, tmp_path):
    cache = tmp_path / "jit-cache"
    target = tmp_path / "executed"
    monkeypatch.setenv("CC", "/no/compiler")
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc")
    monkeypatch.setenv("ME_DSL_JIT", "1")
    monkeypatch.setenv("ME_DSL_JIT_CACHE_DIR", str(cache))
    # This would reach the iteration cap if executed. Validation must still succeed.
    source = "# me:compiler=cc\ndef k(x):\n    while 1:\n        pass\n    return x\n"
    assert portable_validator(source, {"x": "float64"}, "float64")["valid"]
    source = f"def k(x):\n    return open({str(target)!r}, 'w')\n"
    assert not portable_validator(source, {"x": "float64"}, "float64")["valid"]
    assert not target.exists()
    assert not cache.exists()


def test_portable_profile_missing_native_support(monkeypatch):
    from blosc2 import blosc2_ext

    monkeypatch.delattr(blosc2_ext, "validate_portable_dsl_source", raising=False)
    info = blosc2.validate_portable_dsl("def k(x):\n    return x\n", {"x": "float64"}, "float64")
    assert not info["valid"]
    assert info["status"] == "runtime_unsupported"


@pytest.mark.parametrize(
    ("source", "inputs", "version"),
    [
        ("def k(x):\n    return x\n\x00", {"x": "float64"}, "0.1"),
        ("def k(x):\n    return x\n", {"x\x00": "float64"}, "0.1"),
        ("def k(x):\n    return x\n", {"x": "float64"}, "0.1\x00"),
    ],
)
def test_portable_profile_rejects_nul(source, inputs, version):
    with pytest.raises(ValueError, match="NUL"):
        blosc2.validate_portable_dsl(source, inputs, "float64", language_version=version)


@pytest.mark.parametrize("literal", ["9007199254740993", "9_007_199_254_740_993", "0x20000000000001"])
def test_portable_integer_literal_bound_is_exact(portable_validator, literal):
    info = portable_validator(f"def k(x):\n    return x + {literal}\n", {"x": "int64"}, "int64")
    assert not info["valid"]
    assert info["status"] == "unsupported_feature"
    assert "portable range" in info["error"]
