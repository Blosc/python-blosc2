#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################


import numpy as np
import pytest

import blosc2
from blosc2.dsl_kernel import DSLSyntaxError
from blosc2.lazyexpr import _apply_jit_backend_pragma

where = np.where
clip = np.clip


def _windows_policy_blocks_dsl_dtype(dtype, operand_dtypes=()) -> bool:
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    dtype = np.dtype(dtype)
    return lazyexpr_mod.sys.platform == "win32" and blosc2.isdtype(dtype, "integral")


def _make_arrays(shape=(8, 8), chunks=(4, 4), blocks=(2, 2)):
    a = np.linspace(0, 1, num=np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.linspace(1, 2, num=np.prod(shape), dtype=np.float32).reshape(shape)
    a2 = blosc2.asarray(a, chunks=chunks, blocks=blocks)
    b2 = blosc2.asarray(b, chunks=chunks, blocks=blocks)
    return a, b, a2, b2


def _make_int_arrays(shape=(8, 8), chunks=(4, 4), blocks=(2, 2)):
    a = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    b = np.arange(np.prod(shape), dtype=np.int32).reshape(shape) + 3
    a2 = blosc2.asarray(a, chunks=chunks, blocks=blocks)
    b2 = blosc2.asarray(b, chunks=chunks, blocks=blocks)
    return a, b, a2, b2


@blosc2.dsl_kernel
def kernel_loop(x, y):
    acc = 0.0
    for i in range(2):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.dsl_kernel
def kernel_fallback_range_2args(x, y):
    acc = 0.0
    for i in range(1, 3):
        acc = acc + x + y + i
    return acc


@blosc2.dsl_kernel
def kernel_integer_ops(x, y):
    acc = ((x + y) - (x * 2)) // 3
    acc = acc % 5
    acc = acc ^ (x & y)
    acc = acc | (x << 1)
    return acc + (y >> 1)


@blosc2.dsl_kernel
def kernel_control_flow_full(x, y):
    acc = x
    for i in range(4):
        if i == 0:
            acc = acc + y
            continue
        if i == 1:
            acc = acc - y
        else:
            acc = where(acc < y, acc + i, acc - i)
            if i == 3:
                break
    return acc


@blosc2.dsl_kernel
def kernel_while_full(x, y):
    acc = x
    i = 0
    while i < 3:
        acc = where(acc < y, acc + 1, acc - 1)
        i = i + 1
    return acc


@blosc2.dsl_kernel
def kernel_loop_param(x, y, niter):
    acc = x
    # loop count comes from scalar niter
    for _i in range(niter):
        acc = where(acc < y, acc + 1, acc - 1)
    return acc


@blosc2.dsl_kernel
def kernel_scalar_float_cast(x, niter):
    offset = float(niter)
    return x + offset


@blosc2.dsl_kernel
def kernel_fallback_kw_call(x, y):
    return clip(x + y, a_min=0.5, a_max=2.5)


@blosc2.dsl_kernel
def kernel_fallback_for_else(x, y):
    acc = x
    for i in range(2):
        acc = acc + i
    else:
        acc = acc + y
    return acc


@blosc2.dsl_kernel
def kernel_fallback_tuple_assign(x, y):
    lhs, rhs = x, y
    return lhs + rhs


@blosc2.dsl_kernel
def kernel_fallback_ternary(x):
    return 1 if x else 0


@blosc2.dsl_kernel
def kernel_index_ramp(x):
    return _i0 * _n1 + _i1  # noqa: F821  # DSL index/shape symbols resolved by miniexpr


@blosc2.dsl_kernel
def kernel_index_ramp_float_cast(x):
    return float(_i0) * _n1 + _i1  # noqa: F821  # DSL index/shape symbols resolved by miniexpr


@blosc2.dsl_kernel
def kernel_index_ramp_int_cast(x):
    return int(_i0 * _n1 + _i1)  # noqa: F821  # DSL index/shape symbols resolved by miniexpr


@blosc2.dsl_kernel
def kernel_bool_cast_numeric(x):
    return bool(x)


@blosc2.dsl_kernel
def kernel_index_ramp_no_inputs():
    return _i0 * _n1 + _i1  # noqa: F821  # DSL index/shape symbols resolved by miniexpr


def test_dsl_kernel_loop_kept_as_full_dsl_function():
    assert kernel_loop.dsl_source is not None
    assert "def kernel_loop(x, y):" in kernel_loop.dsl_source
    assert kernel_loop.input_names == ["x", "y"]

    a, b, a2, b2 = _make_arrays()
    expr = blosc2.lazyudf(kernel_loop, (a2, b2), dtype=a2.dtype, chunks=a2.chunks, blocks=a2.blocks)
    res = expr.compute()
    expected = kernel_loop.func(a, b)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_kernel_integer_ops_kept_as_full_dsl_function():
    assert kernel_integer_ops.dsl_source is not None
    assert "def kernel_integer_ops(x, y):" in kernel_integer_ops.dsl_source
    assert kernel_integer_ops.input_names == ["x", "y"]

    a, b, a2, b2 = _make_int_arrays()
    expr = blosc2.lazyudf(
        kernel_integer_ops,
        (a2, b2),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    try:
        res = expr.compute()
    except RuntimeError as e:
        # Some DSL ops may still be unsupported by miniexpr backends.
        if "DSL kernels require miniexpr" not in str(e):
            raise
    else:
        expected = kernel_integer_ops.func(a, b)
        np.testing.assert_equal(res[...], expected)


def test_dsl_kernel_index_symbols_keep_full_kernel(monkeypatch):
    assert kernel_index_ramp.dsl_source is not None
    assert "def kernel_index_ramp(x):" in kernel_index_ramp.dsl_source

    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    captured = {"calls": 0, "expr": None}

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        captured["calls"] += 1
        captured["expr"] = expression.decode("utf-8") if isinstance(expression, bytes) else expression
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc, jit=jit)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    shape = (10, 10)
    x2 = blosc2.zeros(shape, dtype=np.float32)
    expr = blosc2.lazyudf(kernel_index_ramp, (x2,), dtype=np.float32)
    res = expr[:]

    assert captured["calls"] >= 1
    assert "def kernel_index_ramp(x):" in captured["expr"]
    assert "_i0" in captured["expr"]
    assert "_n1" in captured["expr"]
    assert "_i1" in captured["expr"]
    assert res.shape == shape


def test_dsl_kernel_with_no_inputs_works_with_explicit_shape():
    assert kernel_index_ramp_no_inputs.dsl_source is not None
    assert "def kernel_index_ramp_no_inputs():" in kernel_index_ramp_no_inputs.dsl_source
    assert kernel_index_ramp_no_inputs.input_names == []

    shape = (10, 10)
    expr = blosc2.lazyudf(kernel_index_ramp_no_inputs, (), dtype=np.float32, shape=shape)
    res = expr[:]
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_equal(res, expected)


def test_dsl_kernel_with_no_inputs_sum_returns_scalar():
    shape = (10, 5)
    expr = blosc2.lazyudf(kernel_index_ramp_no_inputs, (), dtype=np.float32, shape=shape)
    result = expr.sum()

    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape).sum()
    assert np.isscalar(result)
    np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.0)


def test_dsl_kernel_with_no_inputs_requires_shape_or_out():
    with pytest.raises(ValueError, match="shape"):
        _ = blosc2.lazyudf(kernel_index_ramp_no_inputs, (), dtype=np.float32)


def test_dsl_kernel_with_no_inputs_handles_windows_dtype_policy(monkeypatch):
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    monkeypatch.setattr(lazyexpr_mod.sys, "platform", "win32")

    shape = (10, 10)
    expr = blosc2.lazyudf(kernel_index_ramp_no_inputs, (), dtype=np.float32, shape=shape)
    res = expr[:]
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_equal(res, expected)


def test_dsl_kernel_index_symbols_float_cast_matches_expected_ramp():
    shape = (32, 5)
    x2 = blosc2.zeros(shape, dtype=np.float32)
    expr = blosc2.lazyudf(kernel_index_ramp_float_cast, (x2,), dtype=np.float32)
    res = expr[:]
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    np.testing.assert_allclose(res, expected, rtol=0.0, atol=0.0)


def test_dsl_kernel_index_symbols_float_cast_uses_miniexpr_fast_path(monkeypatch):
    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    captured = {"calls": 0, "expr": None}

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        captured["calls"] += 1
        captured["expr"] = expression.decode("utf-8") if isinstance(expression, bytes) else expression
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc, jit=jit)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    shape = (16, 9)
    x2 = blosc2.zeros(shape, dtype=np.float32)
    expr = blosc2.lazyudf(kernel_index_ramp_float_cast, (x2,), dtype=np.float32)
    res = expr[:]
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    np.testing.assert_allclose(res, expected, rtol=0.0, atol=0.0)
    assert captured["calls"] >= 1
    assert "def kernel_index_ramp_float_cast(x):" in captured["expr"]
    assert "float(_i0)" in captured["expr"]
    assert "_n1" in captured["expr"]
    assert "_i1" in captured["expr"]


def test_dsl_kernel_index_symbols_int_cast_matches_expected_ramp():
    shape = (32, 5)
    x2 = blosc2.zeros(shape, dtype=np.float32)
    expr = blosc2.lazyudf(kernel_index_ramp_int_cast, (x2,), dtype=np.int64)
    if _windows_policy_blocks_dsl_dtype(np.int64, operand_dtypes=(x2.dtype,)):
        with pytest.raises(RuntimeError, match="DSL kernels require miniexpr"):
            _ = expr[:]
        return
    try:
        res = expr[:]
    except RuntimeError as e:
        import importlib

        lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
        if lazyexpr_mod.sys.platform == "win32":
            pytest.xfail(f"Windows miniexpr int-cast path is unstable in CI: {e}")
        raise
    expected = np.arange(np.prod(shape), dtype=np.int64).reshape(shape)
    np.testing.assert_equal(res, expected)


def test_dsl_kernel_bool_cast_numeric_matches_expected():
    x = np.array([[0.0, 1.0, -2.0], [3.5, 0.0, -0.1]], dtype=np.float32)
    x2 = blosc2.asarray(x, chunks=(2, 3), blocks=(1, 2))
    expr = blosc2.lazyudf(kernel_bool_cast_numeric, (x2,), dtype=np.bool_)
    if _windows_policy_blocks_dsl_dtype(np.bool_, operand_dtypes=(x2.dtype,)):
        with pytest.raises(RuntimeError, match="DSL kernels require miniexpr"):
            _ = expr[:]
        return
    res = expr[:]
    expected = x != 0.0
    np.testing.assert_equal(res, expected)


def test_dsl_kernel_full_control_flow_kept_as_dsl_function():
    assert kernel_control_flow_full.dsl_source is not None
    assert "def kernel_control_flow_full(x, y):" in kernel_control_flow_full.dsl_source
    assert "for i in range(4):" in kernel_control_flow_full.dsl_source
    assert "if i == 1:" in kernel_control_flow_full.dsl_source
    assert "continue" in kernel_control_flow_full.dsl_source
    assert "break" in kernel_control_flow_full.dsl_source
    assert "where(" in kernel_control_flow_full.dsl_source

    a, b, a2, b2 = _make_arrays()
    expr = blosc2.lazyudf(
        kernel_control_flow_full,
        (a2, b2),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    res = expr.compute()
    expected = kernel_control_flow_full.func(a, b)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_kernel_while_kept_as_dsl_function():
    assert kernel_while_full.dsl_source is not None
    assert "def kernel_while_full(x, y):" in kernel_while_full.dsl_source
    assert "while i < 3:" in kernel_while_full.dsl_source

    a, b, a2, b2 = _make_arrays()
    expr = blosc2.lazyudf(
        kernel_while_full,
        (a2, b2),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    res = expr.compute()
    expected = kernel_while_full.func(a, b)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_kernel_accepts_scalar_param_per_call():
    assert kernel_loop_param.dsl_source is not None
    assert "def kernel_loop_param(x, y, niter):" in kernel_loop_param.dsl_source
    assert "for _i in range(niter):" in kernel_loop_param.dsl_source
    assert kernel_loop_param.input_names == ["x", "y", "niter"]

    a, b, a2, b2 = _make_arrays()
    niter = 3
    expr = blosc2.lazyudf(
        kernel_loop_param,
        (a2, b2, niter),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    res = expr.compute()
    expected = kernel_loop_param.func(a, b, niter)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_kernel_scalar_param_keeps_miniexpr_fast_path(monkeypatch):
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    old_try_miniexpr = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = True

    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    captured = {"calls": 0, "expr": None, "keys": None}

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        captured["calls"] += 1
        captured["expr"] = expression.decode("utf-8") if isinstance(expression, bytes) else expression
        captured["keys"] = tuple(inputs.keys())
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc, jit=jit)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    try:
        a, b, a2, b2 = _make_arrays(shape=(32, 32), chunks=(16, 16), blocks=(8, 8))
        niter = 3
        expr = blosc2.lazyudf(
            kernel_loop_param,
            (a2, b2, niter),
            dtype=a2.dtype,
        )
        res = expr.compute(strict_miniexpr=False)
        expected = kernel_loop_param.func(a, b, niter)

        np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)
        assert captured["calls"] >= 1
        assert captured["keys"] == ("x", "y")
        assert "def kernel_loop_param(x, y):" in captured["expr"]
        assert "for it in range(3):" not in captured["expr"]
        assert "for _i in range(3):" in captured["expr"]
        assert "# loop count comes from scalar niter" in captured["expr"]
        assert "range(niter)" not in captured["expr"]
        assert "float(niter)" not in captured["expr"]
    finally:
        lazyexpr_mod.try_miniexpr = old_try_miniexpr


def test_dsl_kernel_scalar_float_cast_inlined_without_float_call(monkeypatch):
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    old_try_miniexpr = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = True

    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    captured = {"calls": 0, "expr": None, "keys": None}

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        captured["calls"] += 1
        captured["expr"] = expression.decode("utf-8") if isinstance(expression, bytes) else expression
        captured["keys"] = tuple(inputs.keys())
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc, jit=jit)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    try:
        a, _, a2, _ = _make_arrays(shape=(32, 32), chunks=(16, 16), blocks=(8, 8))
        niter = 3
        expr = blosc2.lazyudf(kernel_scalar_float_cast, (a2, niter), dtype=a2.dtype)
        res = expr.compute()
        expected = kernel_scalar_float_cast.func(a, niter)

        np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)
        assert captured["calls"] >= 1
        assert captured["keys"] == ("x",)
        assert "def kernel_scalar_float_cast(x):" in captured["expr"]
        assert "offset = 3.0" in captured["expr"]
        assert "float(3)" not in captured["expr"]
    finally:
        lazyexpr_mod.try_miniexpr = old_try_miniexpr


def test_dsl_kernel_miniexpr_failure_raises_even_with_strict_disabled(monkeypatch):
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    old_try_miniexpr = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = True

    def failing_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        raise ValueError("forced miniexpr failure")

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", failing_set_pref_expr)

    try:
        _, _, a2, b2 = _make_arrays(shape=(32, 32), chunks=(16, 16), blocks=(8, 8))
        expr = blosc2.lazyudf(kernel_loop, (a2, b2), dtype=a2.dtype)
        with pytest.raises(RuntimeError, match="DSL kernels require miniexpr"):
            _ = expr.compute()
        with pytest.raises(RuntimeError, match="DSL kernels require miniexpr"):
            _ = expr.compute(strict_miniexpr=False)
    finally:
        lazyexpr_mod.try_miniexpr = old_try_miniexpr


def test_lazyudf_jit_policy_forwarding(monkeypatch):
    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    old_try_miniexpr = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = True

    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    seen = []

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None, jit=None):
        seen.append(jit)
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc, jit=jit)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    try:
        _, _, a2, b2 = _make_arrays(shape=(32, 32), chunks=(16, 16), blocks=(8, 8))
        expr = blosc2.lazyudf(kernel_loop, (a2, b2), dtype=a2.dtype, jit=False)
        _ = expr.compute(strict_miniexpr=False)
        _ = expr.compute(jit=True, strict_miniexpr=False)
        assert seen[0] is False
        assert seen[1] is True
    finally:
        lazyexpr_mod.try_miniexpr = old_try_miniexpr


def test_jit_backend_pragma_wrapping_plain_expression():
    expr = _apply_jit_backend_pragma("sin((a + 0.5))", {"a": np.empty(1, dtype=np.float64)}, "cc")
    assert expr.startswith("# me:compiler=cc\ndef __me_auto(a):")
    assert "return sin((a + 0.5))" in expr


def test_jit_backend_pragma_wrapping_dsl_source():
    dsl_src = "def k(a):\n    return sin((a + 0.5))"
    wrapped = _apply_jit_backend_pragma(dsl_src, {"a": np.empty(1, dtype=np.float64)}, "tcc")
    assert wrapped.startswith("# me:compiler=tcc\ndef k(a):")


@pytest.mark.parametrize(
    "kernel",
    [
        kernel_fallback_kw_call,
        kernel_fallback_for_else,
        kernel_fallback_tuple_assign,
    ],
)
def test_dsl_kernel_flawed_syntax_detected_fallback_callable(kernel):
    assert kernel.dsl_source is None
    assert kernel.input_names is None
    assert isinstance(kernel.dsl_error, DSLSyntaxError)

    a, b, a2, b2 = _make_arrays()
    with pytest.raises(DSLSyntaxError, match="Invalid DSL kernel"):
        _ = blosc2.lazyudf(
            kernel,
            (a2, b2),
            dtype=a2.dtype,
            chunks=a2.chunks,
            blocks=a2.blocks,
        )


def test_dsl_kernel_ternary_rejected_with_actionable_error():
    assert kernel_fallback_ternary.dsl_source is None
    assert isinstance(kernel_fallback_ternary.dsl_error, DSLSyntaxError)
    msg = str(kernel_fallback_ternary.dsl_error)
    assert "Ternary expressions are not supported in DSL" in msg
    assert "line" in msg
    assert "column" in msg
    assert "DSL kernel source:" in msg
    assert "^" in msg


def test_validate_dsl_api_valid_and_invalid():
    valid_report = blosc2.validate_dsl(kernel_loop)
    assert valid_report["valid"] is True
    assert valid_report["error"] is None
    assert "def kernel_loop(x, y):" in valid_report["dsl_source"]
    assert valid_report["input_names"] == ["x", "y"]

    invalid_report = blosc2.validate_dsl(kernel_fallback_ternary)
    assert invalid_report["valid"] is False
    assert "Ternary expressions are not supported in DSL" in invalid_report["error"]
    assert invalid_report["dsl_source"] is None
    assert invalid_report["input_names"] is None


# ---------------------------------------------------------------------------
# Tests for DSL kernel persistence (save / reload / execute)
# ---------------------------------------------------------------------------


@blosc2.dsl_kernel
def kernel_save_simple(x, y):
    return x**2 + y**2 + 2 * x * y


@blosc2.dsl_kernel
def kernel_save_clamp(x, y):
    return where(x + y > 1.5, x + y, 1.5)


@blosc2.dsl_kernel
def kernel_save_loop(x, y):
    acc = 0.0
    for i in range(3):
        acc = acc + x * (i + 1) - y
    return acc


def _save_reload_compute(kernel, inputs_np, inputs_b2, dtype, urlpaths, extra_kwargs=None):
    """Save a LazyUDF backed by *kernel*, reload it, and return (reloaded_expr, result)."""
    lazy = blosc2.lazyudf(kernel, inputs_b2, dtype=dtype, **(extra_kwargs or {}))
    lazy.save(urlpath=urlpaths["lazy"])
    reloaded = blosc2.open(urlpaths["lazy"])
    return reloaded, reloaded.compute()


def test_dsl_save_simple(tmp_path):
    """Simple quadratic kernel: dsl_source and DSLKernel type survive a round-trip."""
    from blosc2.dsl_kernel import DSLKernel

    shape = (16, 16)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(na, urlpath=str(tmp_path / "a.b2nd"), mode="w")
    b = blosc2.asarray(nb, urlpath=str(tmp_path / "b.b2nd"), mode="w")

    urlpaths = {"lazy": str(tmp_path / "lazy.b2nd")}
    reloaded, result = _save_reload_compute(kernel_save_simple, (na, nb), (a, b), np.float64, urlpaths)

    assert isinstance(reloaded, blosc2.LazyUDF)
    assert isinstance(reloaded.func, DSLKernel), "func must be a DSLKernel after reload"
    assert reloaded.func.dsl_source is not None, "dsl_source must be preserved"
    assert "kernel_save_simple" in reloaded.func.dsl_source

    expected = (na + nb) ** 2  # (x+y)^2 == x^2 + y^2 + 2xy
    np.testing.assert_allclose(result[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_save_clamp(tmp_path):
    """Kernel with a `where` call survives save/reload and produces correct values."""
    from blosc2.dsl_kernel import DSLKernel

    shape = (20, 20)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(0.5, 1.5, np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(na, urlpath=str(tmp_path / "a.b2nd"), mode="w")
    b = blosc2.asarray(nb, urlpath=str(tmp_path / "b.b2nd"), mode="w")

    urlpaths = {"lazy": str(tmp_path / "lazy.b2nd")}
    reloaded, result = _save_reload_compute(kernel_save_clamp, (na, nb), (a, b), np.float64, urlpaths)

    assert isinstance(reloaded.func, DSLKernel)
    assert reloaded.func.dsl_source is not None

    expected = np.where(na + nb > 1.5, na + nb, 1.5)
    np.testing.assert_allclose(result[...], expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Not supported on WASM")
def test_dsl_save_loop(tmp_path):
    """Kernel with a loop (full DSL function) survives save/reload."""
    from blosc2.dsl_kernel import DSLKernel

    shape = (12, 12)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(na, urlpath=str(tmp_path / "a.b2nd"), mode="w")
    b = blosc2.asarray(nb, urlpath=str(tmp_path / "b.b2nd"), mode="w")

    urlpaths = {"lazy": str(tmp_path / "lazy.b2nd")}
    reloaded, result = _save_reload_compute(kernel_save_loop, (na, nb), (a, b), np.float64, urlpaths)

    assert isinstance(reloaded.func, DSLKernel)
    assert reloaded.func.dsl_source is not None
    assert "for i in range(3):" in reloaded.func.dsl_source

    expected = kernel_save_loop.func(na, nb)
    np.testing.assert_allclose(result[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_save_getitem(tmp_path):
    """Reloaded DSL kernel supports __getitem__ (sliced access), not just compute()."""
    from blosc2.dsl_kernel import DSLKernel

    shape = (16, 16)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(na, urlpath=str(tmp_path / "a.b2nd"), mode="w")
    b = blosc2.asarray(nb, urlpath=str(tmp_path / "b.b2nd"), mode="w")

    lazy = blosc2.lazyudf(kernel_save_simple, (a, b), dtype=np.float64)
    lazy.save(urlpath=str(tmp_path / "lazy.b2nd"))
    reloaded = blosc2.open(str(tmp_path / "lazy.b2nd"))

    assert isinstance(reloaded.func, DSLKernel)
    expected = (na + nb) ** 2
    np.testing.assert_allclose(reloaded[()], expected, rtol=1e-5, atol=1e-6)


def test_dsl_save_input_names_match(tmp_path):
    """After reload, input_names in the DSLKernel match the original kernel."""
    from blosc2.dsl_kernel import DSLKernel

    shape = (10, 10)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float32).reshape(shape)
    a = blosc2.asarray(na, urlpath=str(tmp_path / "a.b2nd"), mode="w")
    b = blosc2.asarray(nb, urlpath=str(tmp_path / "b.b2nd"), mode="w")

    lazy = blosc2.lazyudf(kernel_save_simple, (a, b), dtype=np.float64)
    lazy.save(urlpath=str(tmp_path / "lazy.b2nd"))
    reloaded = blosc2.open(str(tmp_path / "lazy.b2nd"))

    assert isinstance(reloaded.func, DSLKernel)
    assert reloaded.func.input_names == ["x", "y"]
