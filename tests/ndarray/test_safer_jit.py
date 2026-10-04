"""Best-effort native JIT without filesystem requirements or unsolicited output."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import blosc2
from blosc2.lazyexpr import _jit_from_env

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM or sys.platform == "win32", reason="POSIX native JIT subprocess tests"
)
HERE = Path(__file__).resolve().parent


def run_probe(tmp_path, backend="tcc", mode="on", trace=False, **overrides):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(("ME_DSL_", "BLOSC_ME_JIT")) or name in ("CC", "CFLAGS"):
            env.pop(name)
    env.update(PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(tmp_path), ME_DSL_TRACE="1" if trace else "0")
    env.update(overrides)
    return subprocess.run(
        [sys.executable, str(HERE / "safer_jit_probe.py"), backend, mode],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )


@pytest.fixture
def denied_libtcc(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("Test double needs a C compiler")
    library = tmp_path / ("denied.dylib" if sys.platform == "darwin" else "denied.so")
    subprocess.run(
        [
            cc,
            "-dynamiclib" if sys.platform == "darwin" else "-shared",
            "-fPIC",
            str(HERE / "safer_jit_libtcc.c"),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return library


def test_tcc_no_cache_directory(tmp_path):
    result = run_probe(tmp_path, trace=True, CC="/no/compiler", PATH="")
    assert "compiler=tcc" in result.stderr
    assert "jit runtime built:" in result.stderr
    assert "source=disk-cache" not in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_tcc_invalid_tmpdir(tmp_path):
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("not a cache")
    result = run_probe(tmp_path, trace=True, TMPDIR=str(invalid), CC="/no/compiler", PATH="")
    assert "jit runtime built:" in result.stderr
    assert list(tmp_path.iterdir()) == [invalid]


@pytest.mark.parametrize("trace", [False, True])
def test_tcc_denied_mapping_is_quiet(tmp_path, denied_libtcc, trace):
    result = run_probe(tmp_path, trace=trace, ME_DSL_JIT_LIBTCC_PATH=str(denied_libtcc))
    assert result.stdout == ""
    if trace:
        assert "jit runtime fallback: interpreter" in result.stderr
        assert "executable mapping failed: Permission denied" in result.stderr
        assert "jit runtime built:" not in result.stderr
    else:
        assert result.stderr == ""
    assert list(tmp_path.iterdir()) == [denied_libtcc]


@pytest.mark.parametrize("trace", [False, True])
def test_tcc_compile_failure_is_quiet(tmp_path, trace):
    result = run_probe(tmp_path, trace=trace, ME_DSL_JIT_TCC_OPTIONS="-invalid-safer-jit-option")
    assert result.stdout == ""
    if trace:
        assert "jit runtime fallback: interpreter" in result.stderr
        assert "invalid-safer-jit-option" in result.stderr
    else:
        assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


def test_tcc_explicit_missing_library(tmp_path):
    result = run_probe(tmp_path, trace=True, ME_DSL_JIT_LIBTCC_PATH=str(tmp_path / "missing"))
    assert "failed to load libtcc" in result.stderr
    assert "jit runtime fallback: interpreter" in result.stderr
    assert "jit runtime built:" not in result.stderr


@pytest.mark.parametrize("backend", ["tcc", "cc"])
def test_jit_off_does_not_compile(tmp_path, backend):
    result = run_probe(tmp_path, backend=backend, mode="off", trace=True)
    assert "jit runtime built:" not in result.stderr
    assert "reason=jit_mode=off" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_runtime_disable_wins_over_jit_true(tmp_path):
    result = run_probe(tmp_path, trace=True, ME_DSL_JIT="0")
    assert "jit runtime built:" not in result.stderr
    assert "disabled by environment" in result.stderr
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("trace", [False, True])
def test_cc_missing_compiler_is_quiet(tmp_path, trace):
    result = run_probe(tmp_path, backend="cc", trace=trace, CC="/no/compiler")
    assert result.stdout == ""
    if trace:
        assert "c compiler unavailable" in result.stderr
        assert "jit runtime fallback: interpreter" in result.stderr
    else:
        assert result.stderr == ""


@pytest.mark.parametrize("failure", ["compile", "load"])
@pytest.mark.parametrize("trace", [False, True])
def test_cc_failure_is_quiet(tmp_path, failure, trace):
    compiler = tmp_path / "test-cc"
    compiler.write_text(
        "#!/bin/sh\n"
        + (
            "echo test-compiler-error >&2\nexit 1\n"
            if failure == "compile"
            else 'while [ "$#" -gt 0 ]; do\n'
            '  if [ "$1" = "-o" ]; then shift; echo invalid-library > "$1"; exit 0; fi\n'
            "  shift\ndone\nexit 1\n"
        )
    )
    compiler.chmod(0o700)
    result = run_probe(tmp_path, backend="cc", trace=trace, CC=str(compiler))
    assert result.stdout == ""
    assert "test-compiler-error" not in result.stderr
    if trace:
        assert "jit runtime fallback: interpreter" in result.stderr
        assert ("compilation failed" if failure == "compile" else "load failed") in result.stderr
    else:
        assert result.stderr == ""


def test_cc_reuses_disk_cache_in_fresh_process(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("System compiler unavailable")
    cold = run_probe(tmp_path, backend="cc", trace=True)
    warm = run_probe(tmp_path, backend="cc", trace=True)
    assert "jit runtime built:" in cold.stderr
    assert "source=disk-cache" in warm.stderr
    assert "jit runtime built:" not in warm.stderr
    assert list((tmp_path / "miniexpr-jit").glob("kernel_*.meta"))


def test_cc_compiler_output_is_opt_in(tmp_path):
    compiler = tmp_path / "test-cc"
    compiler.write_text("#!/bin/sh\necho test-compiler-diagnostic >&2\nexit 1\n")
    compiler.chmod(0o700)
    result = run_probe(tmp_path, backend="cc", CC=str(compiler), ME_DSL_JIT_DEBUG_CC="1")
    assert "test-compiler-diagnostic" in result.stderr
    assert "jit runtime fallback:" not in result.stderr  # Runtime tracing is independently disabled.


def test_blosc_me_jit_environment_values(monkeypatch):
    monkeypatch.setenv("BLOSC_ME_JIT", "cc")
    assert _jit_from_env(False, "tcc") == (True, "cc")
    monkeypatch.setenv("BLOSC_ME_JIT", "1")
    assert _jit_from_env(False, "tcc") == (True, "tcc")
    monkeypatch.setenv("BLOSC_ME_JIT", "0")
    assert _jit_from_env(True, "tcc") == (True, "tcc")
