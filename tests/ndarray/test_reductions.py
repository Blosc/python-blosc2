#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import blosc2
import numexpr as ne
import numpy as np
import pytest

NITEMS_SMALL = 1000
NITEMS = 10_000


@pytest.fixture(params=[np.float32, np.float64])
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,), (NITEMS,), (NITEMS // 100, 100)])
def shape_fixture(request):
    return request.param


@pytest.fixture()
def array_fixture(dtype_fixture, shape_fixture):
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    # For full generality, use different chunks and blocks
    # chunks = [c // 17 for c in na1.shape]
    # blocks = [c // 19 for c in na1.shape]
    # chunks1 = [c // 23 for c in na1.shape]
    # blocks1 = [c // 29 for c in na1.shape]
    chunks = [c // 4 for c in na1.shape]
    blocks = [c // 8 for c in na1.shape]
    chunks1 = [c // 10 for c in na1.shape]
    blocks1 = [c // 30 for c in na1.shape]
    a1 = blosc2.asarray(na1, chunks=chunks, blocks=blocks)
    na2 = np.copy(na1)
    a2 = blosc2.asarray(na2, chunks=chunks, blocks=blocks)
    na3 = np.copy(na1)
    # Let other operands have chunks1 and blocks1
    a3 = blosc2.asarray(na3, chunks=chunks1, blocks=blocks1)
    na4 = np.copy(na1)
    a4 = blosc2.asarray(na4, chunks=chunks1, blocks=blocks1)
    return a1, a2, a3, a4, na1, na2, na3, na4


@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all"])
def test_reduce_bool(array_fixture, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 > a3 * a4
    nres = ne.evaluate("na1 + na2 > na3 * na4")
    res = getattr(expr, reduce_op)()
    nres = getattr(nres, reduce_op)()
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


@pytest.mark.parametrize("reduce_op", ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all"])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype_out", [np.int16, np.float64])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"cparams": blosc2.CParams(clevel=1, filters=[blosc2.Filter.BITSHUFFLE], filters_meta=[0])}],
)
def test_reduce_params(array_fixture, axis, keepdims, dtype_out, reduce_op, kwargs):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and np.isscalar(axis) and len(a1.shape) >= axis:
        return
    if isinstance(axis, tuple) and len(a1.shape) < len(axis):
        return
    if reduce_op == "prod":
        # To avoid overflow, create a1 and a2 with small values
        na1 = np.linspace(0, 0.1, np.prod(a1.shape), dtype=np.float32).reshape(a1.shape)
        a1 = blosc2.asarray(na1)
        na2 = np.linspace(0, 0.5, np.prod(a1.shape), dtype=np.float32).reshape(a1.shape)
        a2 = blosc2.asarray(na2)
        expr = a1 + a2 - 0.2
        nres = eval("na1 + na2 - .2")
    else:
        expr = a1 + a2 - a3 * a4
        nres = eval("na1 + na2 - na3 * na4")
    if reduce_op in ("sum", "prod", "mean", "std"):
        if reduce_op in ("mean", "std") and dtype_out == np.int16:
            # mean and std need float dtype as output
            dtype_out = np.float64
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out, **kwargs)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out)
    else:
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims, **kwargs)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    if kwargs != {}:
        if not np.isscalar(res):
            assert isinstance(res, blosc2.NDArray)
        np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)
    else:
        np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


# TODO: "prod" is not supported here because it overflows with current values
@pytest.mark.parametrize("reduce_op", ["sum", "min", "max", "mean", "std", "var", "any", "all"])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_reduce_expr_arr(array_fixture, axis, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and len(a1.shape) >= axis:
        return
    expr = a1 + a2 - a3 * a4
    nres = eval("na1 + na2 - na3 * na4")
    res = getattr(expr, reduce_op)(axis=axis) + getattr(a1, reduce_op)(axis=axis)
    nres = getattr(nres, reduce_op)(axis=axis) + getattr(na1, reduce_op)(axis=axis)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


# Test broadcasting
@pytest.mark.parametrize("reduce_op", ["sum", "mean", "std", "var", "min", "max", "any", "all"])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize(
    "shapes",
    [
        ((5, 5, 5), (5, 5), (5,)),
        ((10, 10, 10), (10, 10), (10,)),
        ((100, 100, 100), (100, 100), (100,)),
    ],
)
def test_broadcast_params(axis, keepdims, reduce_op, shapes):
    na1 = np.linspace(0, 1, np.prod(shapes[0])).reshape(shapes[0])
    na2 = np.linspace(1, 2, np.prod(shapes[1])).reshape(shapes[1])
    na3 = np.linspace(2, 3, np.prod(shapes[2])).reshape(shapes[2])
    a1 = blosc2.asarray(na1)
    a2 = blosc2.asarray(na2)
    a3 = blosc2.asarray(na3)

    expr1 = a1 + a2 - a3
    assert expr1.shape == shapes[0]
    expr2 = a1 * a2 + 1
    assert expr2.shape == shapes[0]
    res = expr1 - getattr(expr2, reduce_op)(axis=axis, keepdims=keepdims)
    assert res.shape == shapes[0]
    # print(f"res: {res.shape} expr1: {expr1.shape} expr2: {expr2.shape}")
    nres = eval(f"na1 + na2 - na3 - (na1 * na2 + 1).{reduce_op}(axis={axis}, keepdims={keepdims})")

    tol = 1e-14 if a1.dtype == "float64" else 1e-5
    np.testing.assert_allclose(res[:], nres, atol=tol, rtol=tol)


# Test reductions with item parameter
@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stripes", ["rows", "columns"])
@pytest.mark.parametrize("stripe_len", [2, 10, 15, 100])
@pytest.mark.parametrize("shape", [(10, 30), (30, 10), (50, 50)])
@pytest.mark.parametrize("chunks", [None, (10, 15), (20, 30)])
def test_reduce_item(reduce_op, dtype, stripes, stripe_len, shape, chunks):
    na = np.linspace(0, 1, num=np.prod(shape), dtype=dtype).reshape(shape)
    a = blosc2.asarray(na, chunks=chunks)
    tol = 1e-6 if dtype == np.float32 else 1e-15
    for i in range(0, a.shape[0], stripe_len):
        if stripes == "rows":
            _slice = (slice(i, i + stripe_len), slice(None))
        else:
            _slice = (slice(None), slice(i, i + stripe_len))
        slice_ = na[_slice]
        if slice_.size == 0 and reduce_op not in ("sum", "prod"):
            # For mean, std, and var, numpy just raises a warning, so don't check
            if reduce_op in ("min", "max"):
                # Check that a ValueError is raised when the slice is empty
                with pytest.raises(ValueError):
                    getattr(a, reduce_op)(item=_slice)
                with pytest.raises(ValueError):
                    getattr(na[_slice], reduce_op)()
        else:
            res = getattr(a, reduce_op)(item=_slice)
            nres = getattr(na[_slice], reduce_op)()
            np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


# Test fast path for reductions
@pytest.mark.parametrize(
    "chunks, blocks",
    [
        ((20, 50, 100), (10, 50, 100)),
        ((10, 25, 70), (10, 25, 50)),
        ((10, 50, 100), (6, 25, 75)),
        ((15, 30, 75), (7, 20, 50)),
        ((20, 50, 100), (10, 50, 60)),
    ],
)
@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var"])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
def test_fast_path(chunks, blocks, disk, fill_value, reduce_op, axis):
    shape = (20, 50, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
    if disk:
        a = blosc2.open(urlpath)
    na = a[:]

    res = getattr(a, reduce_op)(axis=axis)
    nres = getattr(na[:], reduce_op)(axis=axis)

    assert np.allclose(res, nres)

@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var"])
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_save(disk, fill_value, reduce_op, axis):
    shape = (20, 50, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, urlpath=urlpath, mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, urlpath=urlpath, mode="w")
    if disk:
        a = blosc2.open(urlpath)
    na = a[:]

    expr = f"a + a.{reduce_op}(axis={axis})"
    lexpr = blosc2.lazyexpr(expr, operands={"a": a})
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = na + getattr(na[()], reduce_op)(axis=axis)
    assert np.allclose(res[()], nres)

    # A expression with a single operand that is reduced should be supported as well
    expr = f"a.{reduce_op}(axis={axis})"
    lexpr = blosc2.lazyexpr(expr, operands={"a": a})
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = getattr(na[()], reduce_op)(axis=axis)

    assert np.allclose(res[()], nres)
