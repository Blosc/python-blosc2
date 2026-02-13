#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


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
            tmp = np.where(x < y, y + i, x - i)
        else:
            tmp = np.where(x > y, x + i, y - i)
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
            acc = np.where(acc < y, acc + i, acc - i)
            if i == 3:
                break
    return acc


@blosc2.dsl_kernel
def kernel_while_full(x, y):
    acc = x
    i = 0
    while i < 3:
        acc = np.where(acc < y, acc + 1, acc - 1)
        i = i + 1
    return acc


@blosc2.dsl_kernel
def kernel_loop_param(x, y, niter):
    acc = x
    for _i in range(niter):
        acc = np.where(acc < y, acc + 1, acc - 1)
    return acc


@blosc2.dsl_kernel
def kernel_fallback_kw_call(x, y):
    return np.clip(x + y, a_min=0.5, a_max=2.5)


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


def test_dsl_kernel_reduced_expr():
    assert kernel_loop.dsl_source is not None
    assert "def " not in kernel_loop.dsl_source
    assert kernel_loop.input_names == ["x", "y"]

    a, b, a2, b2 = _make_arrays()
    expr = blosc2.lazyudf(kernel_loop, (a2, b2), dtype=a2.dtype, chunks=a2.chunks, blocks=a2.blocks)
    res = expr.compute()
    expected = kernel_loop.func(a, b)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)


def test_dsl_kernel_integer_ops_reduced_expr():
    assert kernel_integer_ops.dsl_source is not None
    assert "def " not in kernel_integer_ops.dsl_source
    assert kernel_integer_ops.input_names == ["x", "y"]

    a, b, a2, b2 = _make_int_arrays()
    expr = blosc2.lazyudf(
        kernel_integer_ops,
        (a2, b2),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    res = expr.compute()
    expected = kernel_integer_ops.func(a, b)

    np.testing.assert_equal(res[...], expected)


def test_dsl_kernel_full_control_flow_kept_as_dsl_function():
    assert kernel_control_flow_full.dsl_source is not None
    assert "def kernel_control_flow_full(x, y):" in kernel_control_flow_full.dsl_source
    assert "for i in range(4):" in kernel_control_flow_full.dsl_source
    assert "elif (i == 1):" in kernel_control_flow_full.dsl_source
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
    assert "while (i < 3):" in kernel_while_full.dsl_source

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
    if blosc2.IS_WASM:
        pytest.skip("miniexpr fast path is not available on WASM")

    import importlib

    lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
    old_try_miniexpr = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = True

    original_set_pref_expr = blosc2.NDArray._set_pref_expr
    captured = {"calls": 0, "expr": None, "keys": None}

    def wrapped_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc=None):
        captured["calls"] += 1
        captured["expr"] = expression.decode("utf-8") if isinstance(expression, bytes) else expression
        captured["keys"] = tuple(inputs.keys())
        return original_set_pref_expr(self, expression, inputs, fp_accuracy, aux_reduc)

    monkeypatch.setattr(blosc2.NDArray, "_set_pref_expr", wrapped_set_pref_expr)

    try:
        a, b, a2, b2 = _make_arrays(shape=(32, 32), chunks=(16, 16), blocks=(8, 8))
        niter = 3
        expr = blosc2.lazyudf(
            kernel_loop_param,
            (a2, b2, niter),
            dtype=a2.dtype,
        )
        res = expr.compute()
        expected = kernel_loop_param.func(a, b, niter)

        np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)
        assert captured["calls"] >= 1
        assert captured["keys"] == ("x", "y")
        assert "def kernel_loop_param(x, y):" in captured["expr"]
        assert "for it in range(3):" not in captured["expr"]
        assert "for _i in range(3):" in captured["expr"]
        assert "range(niter)" not in captured["expr"]
        assert "float(niter)" not in captured["expr"]
    finally:
        lazyexpr_mod.try_miniexpr = old_try_miniexpr


@pytest.mark.parametrize(
    "kernel",
    [
        kernel_fallback_range_2args,
        kernel_fallback_kw_call,
        kernel_fallback_for_else,
        kernel_fallback_tuple_assign,
    ],
)
def test_dsl_kernel_flawed_syntax_detected_fallback_callable(kernel):
    assert kernel.dsl_source is None
    assert kernel.input_names is None

    a, b, a2, b2 = _make_arrays()
    expr = blosc2.lazyudf(
        kernel,
        (a2, b2),
        dtype=a2.dtype,
        chunks=a2.chunks,
        blocks=a2.blocks,
    )
    res = expr.compute()
    expected = kernel.func(a, b)

    np.testing.assert_allclose(res[...], expected, rtol=1e-5, atol=1e-6)
