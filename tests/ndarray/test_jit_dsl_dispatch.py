#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


def _mandel_numpy(cr, ci, max_iter):
    zr = np.zeros_like(cr)
    zi = np.zeros_like(ci)
    n = np.zeros(cr.shape, dtype=np.int64)
    active = np.ones(cr.shape, dtype=bool)
    for _ in range(max_iter):
        mag = zr * zr + zi * zi
        active = active & ~(active & (mag > 4.0))
        new_zr = zr * zr - zi * zi + cr
        new_zi = 2 * zr * zi + ci
        zr = np.where(active, new_zr, zr)
        zi = np.where(active, new_zi, zi)
        n = np.where(active, n + 1, n)
    return n


def _mandel_grid():
    h, w = 12, 16
    cr = np.linspace(-2, 1, w).astype(np.float64)[None, :] * np.ones((h, 1))
    ci = np.linspace(-1, 1, h).astype(np.float64)[:, None] * np.ones((1, w))
    return cr, ci


def test_jit_control_flow_dispatches_to_dsl_and_matches_numpy(monkeypatch, capsys):
    @blosc2.jit
    def mandel(cr, ci, max_iter):
        zr = 0.0
        zi = 0.0
        n = 0
        for _i in range(max_iter):
            if zr * zr + zi * zi > 4.0:
                break
            new_zr = zr * zr - zi * zi + cr
            zi = 2 * zr * zi + ci
            zr = new_zr
            n = n + 1
        return n

    cr, ci = _mandel_grid()
    monkeypatch.setenv("BLOSC_ME_JIT_TRACE", "1")
    res = mandel(cr, ci, 30)
    captured = capsys.readouterr()
    assert "engine=miniexpr" in captured.out
    assert "def mandel" in captured.out
    np.testing.assert_array_equal(res, _mandel_numpy(cr, ci, 30))


def test_jit_control_flow_with_default_argument():
    @blosc2.jit
    def mandel(cr, ci, max_iter=30):
        zr = 0.0
        zi = 0.0
        n = 0
        for _i in range(max_iter):
            if zr * zr + zi * zi > 4.0:
                break
            new_zr = zr * zr - zi * zi + cr
            zi = 2 * zr * zi + ci
            zr = new_zr
            n = n + 1
        return n

    cr, ci = _mandel_grid()
    res = mandel(cr, ci)
    np.testing.assert_array_equal(res, _mandel_numpy(cr, ci, 30))


def test_jit_elementwise_function_still_traces(monkeypatch):
    calls = []
    real_lazyudf = blosc2.lazyudf

    def spy_lazyudf(*args, **kwargs):
        calls.append((args, kwargs))
        return real_lazyudf(*args, **kwargs)

    monkeypatch.setattr(blosc2, "lazyudf", spy_lazyudf)

    @blosc2.jit
    def elemwise(a, b):
        return a * 2.0 + b

    a = np.arange(100, dtype=np.float64)
    b = np.arange(100, dtype=np.float64) * 0.5
    res = elemwise(a, b)
    np.testing.assert_allclose(res, a * 2.0 + b)
    assert calls == []  # no control flow -> never routed through the DSL/lazyudf path


def test_jit_strict_true_on_elementwise_dsl_valid_function_uses_dsl(monkeypatch):
    calls = []
    import blosc2.proxy as proxy_mod

    real_wrapper = proxy_mod._jit_dsl_wrapper

    def spy(*args, **kwargs):
        calls.append(True)
        return real_wrapper(*args, **kwargs)

    monkeypatch.setattr(proxy_mod, "_jit_dsl_wrapper", spy)

    @blosc2.jit(strict=True)
    def elemwise(a, b):
        return a * 2.0 + b

    a = np.arange(100, dtype=np.float64)
    b = np.arange(100, dtype=np.float64) * 0.5
    res = elemwise(a, b)
    np.testing.assert_allclose(res, a * 2.0 + b)
    assert calls  # dispatched through the DSL wrapper


def test_jit_strict_true_on_non_dsl_function_raises_at_decoration_time():
    with pytest.raises(Exception, match="axis"):

        @blosc2.jit(strict=True)
        def bad(a):
            return np.sum(a, axis=1)


def test_jit_strict_false_on_control_flow_traces():
    @blosc2.jit(strict=False)
    def cf_func(a, b):
        if True:
            return a + b
        return a - b

    a = np.arange(100, dtype=np.float64)
    b = np.arange(100, dtype=np.float64) * 0.5
    res = cf_func(a, b)
    np.testing.assert_allclose(res, a + b)


def test_jit_control_flow_on_python_scalar_flag_still_traces():
    @blosc2.jit
    def scalar_flag(a, b, flag):
        if flag:
            return a + b
        return a - b

    a = np.arange(100, dtype=np.float64)
    b = np.arange(100, dtype=np.float64) * 0.5
    np.testing.assert_allclose(scalar_flag(a, b, True), a + b)
    np.testing.assert_allclose(scalar_flag(a, b, False), a - b)


def test_jit_dsl_route_rejects_broadcasting():
    @blosc2.jit
    def kernel(a, b, n):
        acc = 0.0
        for _i in range(n):
            acc = acc + a + b
        return acc

    with pytest.raises(TypeError, match="broadcasting"):
        kernel(np.zeros((10,)), np.zeros((20,)), 2)


def _kernel_src(a, b, n):
    acc = 0.0
    for _i in range(n):
        acc = acc + a + b
    return acc


def test_jit_dsl_route_out_numpy_c_contiguous_filled_in_place():
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5
    out = np.empty(1000, dtype=np.float64)
    jit_f = blosc2.jit(out=out)(_kernel_src)
    res = jit_f(a, b, 3)
    assert res is out
    np.testing.assert_allclose(out, (a + b) * 3)


def test_jit_dsl_route_out_numpy_non_contiguous_uses_copyto_fallback():
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5
    out = np.empty(2000, dtype=np.float64)[::2]
    assert not out.flags.c_contiguous
    jit_f = blosc2.jit(out=out)(_kernel_src)
    res = jit_f(a, b, 3)
    assert res is out
    np.testing.assert_allclose(out, (a + b) * 3)


def test_jit_dsl_route_out_mismatched_shape_or_dtype_raises_typeerror():
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5

    with pytest.raises(TypeError, match="shape"):
        blosc2.jit(out=np.empty(500, dtype=np.float64))(_kernel_src)(a, b, 3)

    with pytest.raises(TypeError, match="dtype"):
        blosc2.jit(out=np.empty(1000, dtype=np.float32))(_kernel_src)(a, b, 3)


def test_jit_dsl_route_ndarray_out_raises_not_implemented_mentioning_urlpath():
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5
    nd_out = blosc2.zeros((1000,), dtype=np.float64)
    with pytest.raises(NotImplementedError, match="urlpath"):
        blosc2.jit(out=nd_out)(_kernel_src)(a, b, 3)


def test_jit_dsl_route_compute_urlpath_persists_result(tmp_path):
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5
    urlpath = str(tmp_path / "persisted.b2nd")
    jit_f = blosc2.jit(urlpath=urlpath, mode="w")(_kernel_src)
    res = jit_f(a, b, 3)
    assert isinstance(res, blosc2.NDArray)
    reopened = blosc2.open(urlpath)
    np.testing.assert_allclose(reopened[:], (a + b) * 3)


def test_jit_dsl_route_ndarray_operands_match_numpy_operands():
    a = np.arange(1000, dtype=np.float64)
    b = np.arange(1000, dtype=np.float64) * 0.5
    na = blosc2.asarray(a)
    nb = blosc2.asarray(b)
    jit_f = blosc2.jit()(_kernel_src)
    res_numpy = jit_f(a, b, 3)
    res_ndarray = jit_f(na, nb, 3)
    np.testing.assert_array_equal(res_numpy, res_ndarray)
