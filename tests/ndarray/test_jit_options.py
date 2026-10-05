"""Global and scoped JIT policies, including native compiler configuration."""

import asyncio
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def restore_options(monkeypatch):
    previous = blosc2.get_jit_options()
    for name in (
        "CC",
        "CFLAGS",
        "ME_DSL_TRACE",
        "ME_DSL_JIT",
        "ME_DSL_JIT_COMPILER",
        "ME_DSL_JIT_DEBUG_CC",
        "ME_DSL_JIT_CACHE_DIR",
        "BLOSC_ME_JIT",
        "BLOSC_ME_JIT_TRACE",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    blosc2.set_jit_options(**previous)


def test_set_get_and_atomic_validation():
    original = blosc2.get_jit_options()
    assert blosc2.set_jit_options(jit=True, jit_backend="cc") == original
    result = blosc2.get_jit_options()
    assert result["jit"] is True
    assert result["jit_backend"] == "cc"
    result["jit"] = False
    assert blosc2.get_jit_options()["jit"] is True
    with pytest.raises(TypeError):
        blosc2.set_jit_options(jit=False, trace="yes")
    assert blosc2.get_jit_options()["jit"] is True
    blosc2.set_jit_options(jit=None, jit_backend=None)
    assert blosc2.get_jit_options() == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"jit": 1},
        {"jit_backend": "invalid"},
        {"fp_accuracy": 1},
        {"compiler": ""},
        {"cflags": "\0"},
        {"cache_dir": ""},
    ],
)
def test_invalid_options(kwargs):
    with pytest.raises((TypeError, ValueError)):
        blosc2.set_jit_options(**kwargs)
    with pytest.raises((TypeError, ValueError)):
        with blosc2.jit_options(**kwargs):
            pass


def test_nested_context_and_exception():
    blosc2.set_jit_options(jit=True, jit_backend="cc")
    with blosc2.jit_options(jit=False):
        assert blosc2.get_jit_options()["jit_backend"] == "cc"

        def fail():
            with blosc2.jit_options(jit=None, jit_backend="tcc"):
                assert blosc2.get_jit_options()["jit"] is None
                raise RuntimeError("restore")

        with pytest.raises(RuntimeError):
            fail()
        assert blosc2.get_jit_options()["jit"] is False
    assert blosc2.get_jit_options()["jit"] is True


def test_context_isolation():
    async def worker(backend):
        with blosc2.jit_options(jit_backend=backend):
            await asyncio.sleep(0)
            return blosc2.get_jit_options()["jit_backend"]

    async def run():
        return await asyncio.gather(worker("cc"), worker("tcc"))

    assert asyncio.run(run()) == ["cc", "tcc"]
    with blosc2.jit_options(jit_backend="cc"), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(blosc2.get_jit_options).result()["jit_backend"] is None


@pytest.mark.parametrize("method", ["mean", "var", "std"])
@pytest.mark.parametrize("axis", [None, 0])
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_statistical_reductions_forward_execution_options(
    method, axis, keepdims, masked, tmp_path, monkeypatch
):
    data = np.arange(48, dtype=np.float64).reshape(8, 6) / 4
    expr = blosc2.asarray(data) + 1
    mask = (data % 3) != 0 if masked else None
    options = {
        "jit": False,
        "jit_backend": "cc",
        "trace": False,
        "compiler": "/no/compiler",
        "cflags": "-O1",
        "cache_dir": str(tmp_path / "cache"),
        "compiler_output": False,
    }
    observed = []
    original = blosc2.LazyExpr.sum

    def capture(self, *args, **kwargs):
        observed.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.LazyExpr, "sum", capture)
    statistic_kwargs = {"ddof": 1} if method != "mean" else {}
    storage_kwargs = {"urlpath": str(tmp_path / "result.b2nd"), "mode": "w"} if axis == 0 else {}
    result = getattr(expr, method)(
        axis=axis,
        keepdims=keepdims,
        where=mask,
        fp_accuracy=blosc2.FPAccuracy.MEDIUM,
        **statistic_kwargs,
        **options,
        **storage_kwargs,
    )
    expected = getattr(np, method)(
        data + 1, axis=axis, keepdims=keepdims, where=True if mask is None else mask, **statistic_kwargs
    )
    if axis == 0:
        assert isinstance(result, blosc2.NDArray)
        assert (tmp_path / "result.b2nd").exists()
        result = result[:]
    np.testing.assert_allclose(result, expected)
    assert observed
    for kwargs in observed:
        assert {name: kwargs.get(name) for name in options} == options
        assert kwargs["fp_accuracy"] == blosc2.FPAccuracy.MEDIUM
        assert "urlpath" not in kwargs
        assert "mode" not in kwargs


@pytest.mark.parametrize("method", ["mean", "var", "std"])
def test_statistical_reductions_execution_options_with_out(method):
    data = np.arange(24, dtype=np.float64).reshape(4, 6)
    expr = blosc2.asarray(data) + 1
    out = blosc2.empty((6,), dtype=np.float64)
    result = getattr(expr, method)(axis=0, out=out, jit=True, trace=False)
    assert result is out
    np.testing.assert_allclose(out[:], getattr(np, method)(data + 1, axis=0))


@pytest.mark.parametrize("method", ["mean", "var", "std"])
def test_statistical_reductions_reject_invalid_execution_options(method):
    expr = blosc2.asarray(np.arange(12, dtype=np.float64)) + 1
    with pytest.raises(TypeError, match="jit must be bool"):
        getattr(expr, method)(jit="invalid")


@pytest.mark.parametrize("method", ["mean", "var", "std"])
@pytest.mark.parametrize("axis", [None, 0])
def test_statistical_reductions_slice_execution_options(method, axis, monkeypatch):
    data = np.arange(48, dtype=np.float64).reshape(8, 6)
    expr = blosc2.asarray(data) + 1
    observed = []
    original = blosc2.LazyExpr.compute

    def capture(self, *args, **kwargs):
        observed.append(kwargs.copy())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.LazyExpr, "compute", capture)
    result = getattr(expr, method)(
        axis=axis, item=slice(1, 5), jit=False, trace=False, fp_accuracy=blosc2.FPAccuracy.MEDIUM
    )
    np.testing.assert_allclose(result, getattr(np, method)(data[1:5] + 1, axis=axis))
    assert observed
    for kwargs in observed:
        assert kwargs["jit"] is False
        assert kwargs["trace"] is False
        assert kwargs["fp_accuracy"] == blosc2.FPAccuracy.MEDIUM


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("shape", [None, (0, 2)])
@pytest.mark.parametrize("jit", [False, True, None])
def test_empty_ramps_accept_execution_options(constructor, dtype, shape, jit, tmp_path, capfd):
    options = {
        "jit": jit,
        "jit_backend": "cc",
        "fp_accuracy": blosc2.FPAccuracy.HIGH,
        "trace": True,
        "compiler": "/no/compiler",
        "cflags": "-invalid-unused-option",
        "cache_dir": tmp_path / "unused-cache",
        "compiler_output": True,
    }
    urlpath = tmp_path / "empty.b2nd"
    args = (0,) if constructor == "arange" else (0, 1, 0)
    result = getattr(blosc2, constructor)(
        *args, dtype=dtype, shape=shape, urlpath=urlpath, mode="w", **options
    )
    assert result.shape == ((0,) if shape is None else shape)
    assert result.dtype == dtype
    assert result[:].size == 0
    assert urlpath.exists()
    assert not (tmp_path / "unused-cache").exists()
    assert capfd.readouterr() == ("", "")


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_empty_ramps_accept_inherited_execution_options(constructor):
    options = dict.fromkeys(blosc2.get_jit_options())
    args = (0,) if constructor == "arange" else (0, 1, 0)
    with blosc2.jit_options(jit=True, trace=True):
        result = getattr(blosc2, constructor)(*args, **options)
    assert result.shape == (0,)


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
@pytest.mark.parametrize(
    ("options", "error"),
    [
        ({"jit": "invalid"}, TypeError),
        ({"trace": "invalid"}, TypeError),
        ({"jit_backend": "invalid"}, ValueError),
        ({"fp_accuracy": 1}, TypeError),
        ({"cflags": "\0"}, ValueError),
    ],
)
def test_empty_ramps_validate_execution_options(constructor, options, error):
    args = (0,) if constructor == "arange" else (0, 1, 0)
    with pytest.raises(error):
        getattr(blosc2, constructor)(*args, **options)


@pytest.mark.parametrize("constructor", ["arange", "linspace"])
def test_ramp_metadata_shortcut_strips_execution_options(constructor, monkeypatch):
    monkeypatch.setattr(sys.modules["blosc2.ndarray"], "is_inside_new_expr", lambda: True)
    args = (3,) if constructor == "arange" else (0, 1, 3)
    result = getattr(blosc2, constructor)(*args, jit=True, jit_backend="tcc", trace=True)
    assert result.shape == (3,)
    np.testing.assert_array_equal(result[:], 0)


@blosc2.dsl_kernel
def kernel(x):
    return x * 2 + 1


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native miniexpr prefilter observation")
def test_fp_accuracy_precedence(monkeypatch):
    observed = []
    original = blosc2.NDArray._set_pref_expr

    def capture(self, *args, **kwargs):
        observed.append(kwargs.get("fp_accuracy", args[2] if len(args) > 2 else None))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", capture)
    arr = blosc2.asarray(np.arange(32, dtype=np.float64))
    blosc2.set_jit_options(fp_accuracy=blosc2.FPAccuracy.HIGH)
    expr = arr + 1
    expr.compute()
    assert observed[-1] == blosc2.FPAccuracy.HIGH
    with blosc2.jit_options(fp_accuracy=blosc2.FPAccuracy.MEDIUM):
        expr.compute(fp_accuracy=blosc2.FPAccuracy.HIGH)
        assert observed[-1] == blosc2.FPAccuracy.HIGH
        udf = blosc2.lazyudf(kernel, (arr,), dtype=arr.dtype)
        udf.compute(fp_accuracy=blosc2.FPAccuracy.HIGH)
        assert observed[-1] == blosc2.FPAccuracy.HIGH


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="Native POSIX TCC execution")
def test_evaluation_time_and_udf_policy(capfd):
    arr = blosc2.asarray(np.arange(32, dtype=np.float64))
    expr = arr + 1
    udf = blosc2.lazyudf(kernel, (arr,), dtype=arr.dtype, jit=False)
    with blosc2.jit_options(jit=True, jit_backend="tcc", trace=True):
        np.testing.assert_array_equal(expr.compute()[:], np.arange(32) + 1)
        assert "jit runtime built:" in capfd.readouterr().err
        np.testing.assert_array_equal(udf[:], np.arange(32) * 2 + 1)
        assert "reason=jit_mode=off" in capfd.readouterr().err
    assert blosc2.get_jit_options()["trace"] is False


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="Native POSIX CC backend")
def test_probe_uses_configured_backend(tmp_path, capfd):
    with blosc2.jit_options(jit_backend="cc", compiler="/no/compiler", cache_dir=tmp_path, trace=True):
        status = blosc2.validate_dsl_jit(kernel, (np.float64,), np.float64)
    assert status["compiled"]
    assert not status["jit"]
    assert "c compiler unavailable" in capfd.readouterr().err


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="Native POSIX CC backend")
def test_native_cc_options_and_cache_identity(tmp_path, capfd):
    if shutil.which("cc") is None:
        pytest.skip("CC unavailable")
    before = dict(os.environ)
    cache = tmp_path / "direct cache's $data"
    with blosc2.jit_options(jit_backend="cc", compiler="cc", cflags="-O1", cache_dir=cache, trace=True):
        blosc2.linspace(0, 9, 64)
        cold = capfd.readouterr()
        assert cold.out == ""
        assert "compiler=cc" in cold.err
        assert "jit runtime built:" in cold.err
        assert list(cache.glob("kernel_*.meta"))
        assert not (cache / "miniexpr-jit").exists()
        blosc2.linspace(0, 9, 64)
        assert "jit runtime hit:" in capfd.readouterr().err
        blosc2.linspace(0, 9, 64, trace=False)
        assert capfd.readouterr() == ("", "")
        assert len(list(cache.glob("kernel_*.meta"))) == 1
        blosc2.linspace(0, 9, 64, cflags="-O2")
        assert "jit runtime built:" in capfd.readouterr().err
        assert len(list(cache.glob("kernel_*.meta"))) == 2
        blosc2.linspace(0, 9, 64, compiler="/no/compiler")
        assert "c compiler unavailable" in capfd.readouterr().err
        second = tmp_path / "second-cache"
        blosc2.linspace(0, 9, 64, cache_dir=second)
        assert "jit runtime built:" in capfd.readouterr().err
        assert list(second.glob("kernel_*.meta"))
    assert dict(os.environ) == before


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="Native POSIX TCC execution")
def test_environment_overrides_and_tcc_ignores_cc_settings(tmp_path, capfd, monkeypatch):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "tcc")
    with blosc2.jit_options(
        jit_backend="cc", compiler="/no/compiler", cache_dir=tmp_path / "cache", trace=True
    ):
        blosc2.arange(64)
    assert "compiler=tcc" in capfd.readouterr().err
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("ME_DSL_TRACE", "0")
    with blosc2.jit_options(trace=True):
        blosc2.arange(67)
    assert capfd.readouterr() == ("", "")


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="POSIX compiler-output injection")
def test_compiler_output(tmp_path, capfd):
    compiler = tmp_path / "bad cc's executable"
    compiler.write_text("#!/bin/sh\necho scoped-compiler-output >&2\nexit 1\n")
    compiler.chmod(0o700)
    with blosc2.jit_options(
        jit_backend="cc", compiler=compiler, cache_dir=tmp_path / "cache", compiler_output=True
    ):
        blosc2.linspace(0, 3, 71)
    captured = capfd.readouterr()
    assert "scoped-compiler-output" in captured.err
    assert "jit runtime fallback:" not in captured.err


@pytest.mark.skipif(blosc2.IS_WASM or os.name == "nt", reason="Native POSIX CC backend")
@pytest.mark.parametrize(
    ("variable", "value", "reason"),
    [
        ("CC", "/no/env/compiler", "c compiler unavailable"),
        ("CFLAGS", "-invalid-blosc2-option", "compilation failed"),
    ],
)
def test_cc_environment_precedence(tmp_path, capfd, monkeypatch, variable, value, reason):
    monkeypatch.setenv(variable, value)
    monkeypatch.setenv("ME_DSL_JIT_CACHE_DIR", str(tmp_path / "env-cache"))
    with blosc2.jit_options(
        jit_backend="cc", compiler="cc", cflags="-O1", cache_dir=tmp_path / "python-cache", trace=True
    ):
        blosc2.linspace(0, 11, 79)
    assert reason in capfd.readouterr().err
    assert (tmp_path / "env-cache").exists()
    assert not (tmp_path / "python-cache").exists()
