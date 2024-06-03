#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numexpr as ne
import numpy as np
import pytest

import blosc2

NITEMS_SMALL = 1_000
NITEMS = 50_000

# SUB_URL = 'http://localhost:8002/'
# ROOT = 'foo'
# DIR = 'operands/'
SUB_URL = 'https://demo.caterva2.net/'
ROOT = 'b2tests'
DIR = 'expr/'

# resp = httpx.post(f'{SUB_URL}auth/jwt/login',
#                   data=dict(username='user@example.com', password='foobar'))
# resp.raise_for_status()
# AUTH_COOKIE = '='.join(list(resp.cookies.items())[0])

@pytest.fixture(params=[np.float64])
def dtype_fixture(request):
    return request.param


@pytest.fixture(params=[(NITEMS_SMALL,),
                        (NITEMS,),
                        ])
def shape_fixture(request):
    return request.param


@pytest.fixture(params=[
    None,
    # AUTH_COOKIE,
])
def auth_cookie(request):
    return request.param


@pytest.fixture
def array_fixture(dtype_fixture, shape_fixture, auth_cookie):
    chunks_blocks_fixture = (True, False)
    nelems = np.prod(shape_fixture)
    na1 = np.linspace(0, 10, nelems, dtype=dtype_fixture).reshape(shape_fixture)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a1-{shape_fixture}d.b2nd'
    path = pathlib.PosixPath(f'{ROOT}/{DIR + urlpath}')
    a1 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a2-{shape_fixture}d.b2nd'
    path = pathlib.PosixPath(f'{ROOT}/{DIR + urlpath}')
    a2 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    # Let other operands have chunks1 and blocks1
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a3-{shape_fixture}d.b2nd'
    path = pathlib.PosixPath(f'{ROOT}/{DIR + urlpath}')
    a3 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)
    urlpath = f'ds-0-10-linspace-{dtype_fixture.__name__}-{chunks_blocks_fixture}-a4-{shape_fixture}d.b2nd'
    path = pathlib.PosixPath(f'{ROOT}/{DIR + urlpath}')
    a4 = blosc2.C2Array(path, sub_url=SUB_URL, auth_cookie=auth_cookie)

    return a1, a2, a3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1)


@pytest.mark.parametrize("reduce_op", ["sum", "max", "all"])
def test_reduce_bool(array_fixture, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    expr = a1 + a2 > a3 * a4
    nres = ne.evaluate("na1 + na2 > na3 * na4")
    res = getattr(expr, reduce_op)()
    nres = getattr(nres, reduce_op)()
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)


@pytest.mark.parametrize("reduce_op", ["prod", "mean", "var", "min", "any"])
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype_out", [np.int16])
def test_reduce_params(array_fixture, axis, keepdims, dtype_out, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and np.isscalar(axis) and len(a1.shape) >= axis:
        return
    if type(axis) == tuple and len(a1.shape) < len(axis):
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
@pytest.mark.parametrize("reduce_op", ["max", "mean", "var"])
@pytest.mark.parametrize("axis", [0])
def test_reduce_expr_arr(array_fixture, axis, reduce_op):
    a1, a2, a3, a4, na1, na2, na3, na4 = array_fixture
    if axis is not None and len(a1.shape) >= axis:
        return
    expr = a1 + a2 - a3 * a4
    nres = eval("na1 + na2 - na3 * na4")
    res = getattr(expr, reduce_op)(axis=axis) + getattr(a1, reduce_op)(axis=axis)
    # print(f"res: {res}")
    res = res[()]
    nres = getattr(nres, reduce_op)(axis=axis) + getattr(na1, reduce_op)(axis=axis)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res, nres, atol=tol, rtol=tol)
