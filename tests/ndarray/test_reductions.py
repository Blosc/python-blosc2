#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numexpr as ne
import numpy as np
import pytest

import blosc2

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


@pytest.mark.parametrize("reduce_op", ["sum", "prod", "min", "max", "any", "all"])
def test_reduce_bool(array_fixture, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 > a3 * a4
    nres = ne.evaluate("na1 + na2 > na3 * na4")
    res = getattr(expr, reduce_op)()
    nres = getattr(nres, reduce_op)()
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)


@pytest.mark.parametrize("reduce_op", ["sum", "prod", "mean", "min", "max", "any", "all"])
@pytest.mark.parametrize("axis", [0, 1, (0, 1), None])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype_out", [np.int16, np.float64])
def test_reduce_params(array_fixture, axis, keepdims, dtype_out, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and np.isscalar(axis) and len(a1.shape) >= axis:
        return
    if type(axis) == tuple and len(a1.shape) < len(axis):
        return
    if reduce_op == "prod":
        # To avoid overflow
        expr = a1 - a2 + 1
        nres = eval("na1 - na2 + 1")
    else:
        expr = a1 + a2 - a3 * a4
        nres = eval("na1 + na2 - na3 * na4")
    if reduce_op in ("sum", "prod", "mean"):
        if reduce_op == "mean" and dtype_out == np.int16:
            dtype_out = np.float64
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out)
    else:
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)


# TODO: "any" and "all" are not supported yet because:
# ne.evaluate('(o0 + o1)', local_dict = {'o0': np.array(True), 'o1': np.array(True)})
# is not supported by NumExpr
@pytest.mark.parametrize("reduce_op", ["sum", "min", "max"])
@pytest.mark.parametrize("axis", [0, 1, None])
def test_reduce_expr_arr(array_fixture, axis, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and len(a1.shape) >= axis:
        return
    expr = a1 + a2 - a3 * a4
    nres = eval("na1 + na2 - na3 * na4")
    res = getattr(expr, reduce_op)(axis=axis) + getattr(a1, reduce_op)(axis=axis)
    print(f"res: {res}")
    res = res[()]
    nres = getattr(nres, reduce_op)(axis=axis) + getattr(na1, reduce_op)(axis=axis)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)
