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


def typed_values(values, dtype):
    # NumPy treats nonempty strings as True; fixture Boolean tokens are 0/1.
    return np.array([int(value) for value in values] if dtype == np.dtype("bool") else values, dtype=dtype)


def assert_typed_result(actual, expected):
    assert actual.dtype == expected.dtype
    if expected.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, expected)
    else:
        tolerance = 1e-6 if expected.dtype == np.dtype("float32") else 1e-12
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance, equal_nan=True)
        zeros = expected == 0
        np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("jit", [False, True])
def test_native_source_corpus(case, jit):
    if not CORPUS.is_dir():
        pytest.skip("Set MINIEXPR_PORTABLE_CORPUS to the native miniexpr fixture directory")
    source = (CORPUS / f"{case}.dsl").read_text()
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
    if outcome == "ok":
        result = blosc2.lazyudf(kernel, inputs, dtype=output_dtype, jit=jit)[:]
        assert_typed_result(result, expected)
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
                str(CORPUS / f"{case}.dsl"),
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
                assert_typed_result(typed_values(lines[1:], output_dtype), expected)


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
