#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import math

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

NITEMS_SMALL = 1000
NITEMS = 10_000


@pytest.fixture(params=[np.float32, np.float64])
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,), (NITEMS,), (NITEMS // 100, 100)])
def shape_fixture(request):
    return request.param


@pytest.fixture
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


# @pytest.mark.parametrize("reduce_op", ["sum"])
@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all", "argmax", "argmin"])
def test_reduce_bool(array_fixture, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = (a1 + a2) > (a3 * a4)
    nres = ne_evaluate("(na1 + na2) > (na3 * na4)")
    res = getattr(expr, reduce_op)()
    # res = expr.sum()
    # print("res:", res)
    nres = getattr(nres, reduce_op)()
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "mean", "std", "var", "min", "max", "any", "all", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype_out", [np.int16, np.float64])
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"cparams": blosc2.CParams(clevel=1, filters=[blosc2.Filter.BITSHUFFLE], filters_meta=[0])}],
)
@pytest.mark.heavy
def test_reduce_params(array_fixture, axis, keepdims, dtype_out, reduce_op, kwargs):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and np.isscalar(axis) and len(a1.shape) >= axis:
        return
    if isinstance(axis, tuple) and (len(a1.shape) < len(axis) or reduce_op in ("argmax", "argmin")):
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
@pytest.mark.parametrize(
    "reduce_op", ["sum", "min", "max", "mean", "std", "var", "any", "all", "argmax", "argmin"]
)
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
@pytest.mark.parametrize(
    "reduce_op", ["sum", "mean", "std", "var", "min", "max", "any", "all", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, (0, 1), None])
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
    if isinstance(axis, tuple) and (reduce_op in ("argmax", "argmin")):
        axis = 1
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
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stripes", ["rows", "columns"])
@pytest.mark.parametrize("stripe_len", [2, 10, 15, 100])
@pytest.mark.parametrize("shape", [(10, 30), (30, 10), (50, 50)])
@pytest.mark.parametrize("chunks", [None, (10, 15), (20, 30)])
@pytest.mark.heavy
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
            if reduce_op in ("min", "max", "argmin", "argmax"):
                # Check that a ValueError is raised when the slice is empty
                with pytest.raises(ValueError):
                    getattr(a, reduce_op)(item=_slice)
                with pytest.raises(ValueError):
                    getattr(na[_slice], reduce_op)()
        else:
            res = getattr(a, reduce_op)(item=_slice)
            nres = getattr(na[_slice], reduce_op)()
            np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
def test_reduce_slice(reduce_op):
    shape = (8, 12, 5)
    na = np.linspace(0, 1, num=np.prod(shape)).reshape(shape)
    a = blosc2.asarray(na, chunks=(2, 5, 1))
    tol = 1e-6 if na.dtype == np.float32 else 1e-15
    _slice = (slice(1, 2, 1), slice(3, 7, 1))
    res = getattr(a, reduce_op)(item=_slice, axis=-1)
    nres = getattr(na[_slice], reduce_op)(axis=-1)
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)

    # Test reductions with slices and strides
    _slice = (slice(1, 2, 1), slice(1, 9, 2))
    res = getattr(a, reduce_op)(item=_slice, axis=1)
    nres = getattr(na[_slice], reduce_op)(axis=1)
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)

    # Test reductions with ints
    _slice = (0, slice(1, 9, 1))
    res = getattr(a, reduce_op)(item=_slice, axis=1)
    nres = getattr(na[_slice], reduce_op)(axis=1)
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)

    _slice = (0, slice(1, 9, 2))
    res = getattr(a, reduce_op)(item=_slice, axis=1)
    nres = getattr(na[_slice], reduce_op)(axis=1)
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)


# Test fast path for reductions
@pytest.mark.parametrize(
    ("chunks", "blocks"),
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
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, 1, None])
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
    nres = getattr(na, reduce_op)(axis=axis)

    assert np.allclose(res, nres)


@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_save_version1(disk, fill_value, reduce_op, axis):
    shape = (20, 50, 100)
    if isinstance(axis, tuple) and (reduce_op in ("argmax", "argmin")):
        axis = 1
        shape = (20, 20, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, urlpath=urlpath, mode="w")
        b = blosc2.full(shape, fill_value - 0.1, urlpath="b.b2nd", mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, urlpath=urlpath, mode="w")
        b = blosc2.zeros(shape, dtype=np.float64, urlpath="b.b2nd", mode="w") - 0.1
    if disk:
        a = blosc2.open(urlpath)
        b = blosc2.open("b.b2nd")
    na = a[:]
    nb = b[:]

    # A reduction in the back
    expr = f"a + {reduce_op}(b, axis={axis}) + 1"
    lexpr = blosc2.lazyexpr(expr)
    assert lexpr.shape == a.shape
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = na + getattr(nb[()], reduce_op)(axis=axis) + 1
    assert np.allclose(res[()], nres)

    if disk:
        blosc2.remove_urlpath("a1.b2nd")
        blosc2.remove_urlpath("b.b2nd")
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_save_version2(disk, fill_value, reduce_op, axis):
    shape = (20, 50, 100)
    if isinstance(axis, tuple) and (reduce_op in ("argmax", "argmin")):
        axis = 1
        shape = (20, 20, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, urlpath=urlpath, mode="w")
        b = blosc2.full(shape, fill_value - 0.1, urlpath="b.b2nd", mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, urlpath=urlpath, mode="w")
        b = blosc2.zeros(shape, dtype=np.float64, urlpath="b.b2nd", mode="w") - 0.1
    if disk:
        a = blosc2.open(urlpath)
        b = blosc2.open("b.b2nd")
    na = a[:]
    nb = b[:]

    # A reduction in front
    expr = f"a.{reduce_op}(axis={axis}) + b"
    lexpr = blosc2.lazyexpr(expr, operands={"a": a, "b": b})
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = getattr(na[()], reduce_op)(axis=axis) + nb
    assert np.allclose(res[()], nres)

    if disk:
        blosc2.remove_urlpath("a1.b2nd")
        blosc2.remove_urlpath("b.b2nd")
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_save_version3(disk, fill_value, reduce_op, axis):
    shape = (20, 50, 100)
    if isinstance(axis, tuple) and (reduce_op in ("argmax", "argmin")):
        axis = 1
        shape = (20, 20, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, urlpath=urlpath, mode="w")
        b = blosc2.full(shape, fill_value - 0.1, urlpath="b.b2nd", mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, urlpath=urlpath, mode="w")
        b = blosc2.zeros(shape, dtype=np.float64, urlpath="b.b2nd", mode="w") - 0.1
    if disk:
        a = blosc2.open(urlpath)
        b = blosc2.open("b.b2nd")
    na = a[:]
    nb = b[:]

    # A reduction as a function
    expr = f"{reduce_op}(a, axis={axis}) + b"
    lexpr = blosc2.lazyexpr(expr, operands={"a": a, "b": b})
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = getattr(na[()], reduce_op)(axis=axis) + nb
    assert np.allclose(res[()], nres)

    if disk:
        blosc2.remove_urlpath("a1.b2nd")
        blosc2.remove_urlpath("b.b2nd")
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("fill_value", [0, 1, 0.32])
@pytest.mark.parametrize(
    "reduce_op", ["sum", "prod", "min", "max", "any", "all", "mean", "std", "var", "argmax", "argmin"]
)
@pytest.mark.parametrize("axis", [0, (0, 1), None])
def test_save_version4(disk, fill_value, reduce_op, axis):
    if isinstance(axis, tuple) and (reduce_op in ("argmax", "argmin")):
        axis = 1
    shape = (20, 50, 100)
    urlpath = "a1.b2nd" if disk else None
    if fill_value != 0:
        a = blosc2.full(shape, fill_value, urlpath=urlpath, mode="w")
        b = blosc2.full(shape, fill_value - 0.1, urlpath="b.b2nd", mode="w")
    else:
        a = blosc2.zeros(shape, dtype=np.float64, urlpath=urlpath, mode="w")
        b = blosc2.zeros(shape, dtype=np.float64, urlpath="b.b2nd", mode="w") - 0.1
    if disk:
        a = blosc2.open(urlpath)
        b = blosc2.open("b.b2nd")
    na = a[:]

    # Just a single reduction
    expr = f"a.{reduce_op}(axis={axis})"
    lexpr = blosc2.lazyexpr(expr, operands={"a": a})
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    res = lexpr.compute()
    nres = getattr(na[()], reduce_op)(axis=axis)
    assert np.allclose(res[()], nres)

    if disk:
        blosc2.remove_urlpath("a1.b2nd")
        blosc2.remove_urlpath("b.b2nd")
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("compute", [True, False])
def test_save_constructor_reduce(shape, disk, compute):
    lshape = math.prod(shape)
    urlpath_a = "a.b2nd" if disk else None
    urlpath_b = "b.b2nd" if disk else None
    a = blosc2.arange(lshape, shape=shape, urlpath=urlpath_a, mode="w")
    b = blosc2.ones(shape, urlpath=urlpath_b, mode="w")
    expr = f"arange({lshape}).sum() + a + ones({shape}).sum() + b + 1"
    lexpr = blosc2.lazyexpr(expr)
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    if compute:
        res = lexpr.compute()
        res = res[()]  # for later comparison with nres
    else:
        res = lexpr[()]
    na = np.arange(lshape).reshape(shape).sum()
    nb = np.ones(shape).sum()
    nres = na + a[:] + nb + b[:] + 1
    assert np.allclose(res[()], nres)
    if disk:
        blosc2.remove_urlpath(urlpath_a)
        blosc2.remove_urlpath(urlpath_b)
        blosc2.remove_urlpath("out.b2nd")


@pytest.mark.parametrize("shape", [(10,), (10, 10), (10, 10, 10)])
@pytest.mark.parametrize("disk", [True, False])
@pytest.mark.parametrize("compute", [True, False])
def test_save_constructor_reduce2(shape, disk, compute):
    lshape = math.prod(shape)
    urlpath_a = "a.b2nd" if disk else None
    urlpath_b = "b.b2nd" if disk else None
    a = blosc2.arange(lshape, shape=shape, urlpath=urlpath_a, mode="w")
    b = blosc2.ones(shape, urlpath=urlpath_b, mode="w")
    expr = "sum(a + 1) + (b + 2).sum() + 3"
    lexpr = blosc2.lazyexpr(expr)
    if disk:
        lexpr.save("out.b2nd")
        lexpr = blosc2.open("out.b2nd")
    if compute:
        res = lexpr.compute()
        res = res[()]  # for later comparison with nres
    else:
        res = lexpr[()]
    na = np.arange(lshape).reshape(shape)
    nb = np.ones(shape)
    nres = np.sum(na + 1) + (nb + 2).sum() + 3
    assert np.allclose(res, nres)
    assert res.dtype == nres.dtype
    if disk:
        blosc2.remove_urlpath(urlpath_a)
        blosc2.remove_urlpath(urlpath_b)
        blosc2.remove_urlpath("out.b2nd")


def test_reduction_index():
    shape = (20, 20)
    a = blosc2.linspace(0, 20, num=np.prod(shape), shape=shape)
    arr = blosc2.lazyexpr("sum(a, axis=0)", {"a": a})
    newarr = arr.compute()
    assert arr[:10].shape == (10,)
    assert arr[0].shape == ()
    assert arr.shape == newarr.shape

    a = blosc2.ones(shape=(0, 0))
    with pytest.raises(np.exceptions.AxisError):
        arr = blosc2.lazyexpr("sum(a, axis=(0, 1, 2))", {"a": a})
    with pytest.raises(ValueError):
        arr = blosc2.lazyexpr("sum(a, axis=(0, 0))", {"a": a})


@pytest.mark.parametrize("idx", [0, 1, (0,), slice(1, 2), (slice(0, 1),), slice(0, 4), (0, 2)])
def test_reduction_index2(idx):
    N = 10
    shape = (N, N, N)
    a = blosc2.linspace(0, 1, num=np.prod(shape), shape=(N, N, N))
    expr = blosc2.lazyexpr("a.sum(axis=1)")
    out = expr[idx]
    na = blosc2.asarray(a)
    nout = na.sum(axis=1)[idx]
    assert out.shape == nout.shape
    assert np.allclose(out, nout)


def test_slice_lazy():
    shape = (20, 20)
    a = blosc2.linspace(0, 20, num=np.prod(shape), shape=shape)
    arr = blosc2.lazyexpr("anarr.slice(slice(10,15)) + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], a.slice(slice(10, 15))[:] + 1)


def test_slicebrackets_lazy():
    shape = (20, 20)
    a = blosc2.linspace(0, 20, num=np.prod(shape), shape=shape)
    arr = blosc2.lazyexpr("sum(anarr[10:15], axis=0) + anarr[10:15] + arange(20) + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], np.sum(a[10:15], axis=0) + a[10:15] + np.arange(20) + 1)

    # Try with getitem
    a = blosc2.linspace(0, 20, num=np.prod(shape), shape=shape)
    arr = blosc2.lazyexpr("sum(anarr[10:15], axis=0) + anarr[10:15] + arange(20) + 1", {"anarr": a})
    newarr = arr[:3]
    res = np.sum(a[10:15], axis=0) + a[10:15] + np.arange(20) + 1
    np.testing.assert_allclose(newarr, res[:3])

    # Test other cases
    arr = blosc2.lazyexpr("anarr[10:15, 2:9] + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], a[10:15, 2:9] + 1)

    arr = blosc2.lazyexpr("anarr[10:15][2:9] + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], a[10:15][2:9] + 1)

    arr = blosc2.lazyexpr("sum(anarr[10:15], axis=1) + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], np.sum(a[10:15], axis=1) + 1)

    arr = blosc2.lazyexpr("anarr[10] + 1", {"anarr": a})
    newarr = arr.compute()
    np.testing.assert_allclose(newarr[:], a[10] + 1)

    arr = blosc2.lazyexpr("anarr[10, 1] + 1", {"anarr": a})
    newarr = arr[:]
    np.testing.assert_allclose(newarr, a[10, 1] + 1)


def test_reduce_string():
    shape = (10, 10, 2)

    # Create a NDArray from a NumPy array
    npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
    npc = npa**2 + npb**2 + 2 * npa * npb + 1

    a = blosc2.asarray(npa)
    b = blosc2.asarray(npb)

    # Get a LazyExpr instance
    c = a**2 + b**2 + 2 * a * b + 1
    # Evaluate: output is a NDArray
    d = blosc2.lazyexpr("sl + c.sum() + a.std()", operands={"a": a, "c": c, "sl": a.slice((1, 1))})
    sum = d.compute()[()]
    npsum = npa[1, 1] + np.sum(npc) + np.std(npa)
    assert np.allclose(sum, npsum)
