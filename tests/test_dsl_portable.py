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


@pytest.mark.parametrize("case", ["affine", "chains", "loops"])
@pytest.mark.parametrize("jit", [False, True])
def test_native_source_corpus(case, jit):
    if not CORPUS.is_dir():
        pytest.skip("Set MINIEXPR_PORTABLE_CORPUS to the native miniexpr fixture directory")
    source = (CORPUS / f"{case}.dsl").read_text()
    words = (CORPUS / f"{case}.txt").read_text().split()
    count, nvars = map(int, words[:2])
    names = words[2 : 2 + nvars]
    rows = np.array(words[2 + nvars :], dtype=np.float64).reshape(count, nvars + 1)
    kernel = DSLKernel.from_source(source)
    by_name = dict(zip(names, rows[:, :-1].T, strict=True))
    inputs = [by_name[name].copy() for name in kernel.input_names]
    result = blosc2.lazyudf(kernel, inputs, dtype=np.float64, jit=jit)[:]
    np.testing.assert_allclose(result, rows[:, -1], rtol=1e-12, atol=1e-12)
    assert kernel.dsl_source == source
    assert kernel.func is None

    runner = os.environ.get("MINIEXPR_PORTABLE_RUNNER")
    if runner:
        completed = subprocess.run(
            [runner, str(CORPUS / f"{case}.dsl"), str(CORPUS / f"{case}.txt"), "on" if jit else "off"],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = completed.stdout.splitlines()
        assert lines[0] == f"jit={int(jit)}"
        np.testing.assert_allclose(np.array(lines[1:], dtype=float), rows[:, -1], rtol=1e-12, atol=1e-12)


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
