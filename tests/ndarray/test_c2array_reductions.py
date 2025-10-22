#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import pathlib

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

pytestmark = pytest.mark.network

NITEMS_SMALL = 1_000
ROOT = "@public"
DIR = "expr/"


def get_arrays(shape, chunks_blocks):
    dtype = np.float64
    nelems = np.prod(shape)
    na1 = np.linspace(0, 10, nelems, dtype=dtype).reshape(shape)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a1-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a1 = blosc2.C2Array(path)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a2-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a2 = blosc2.C2Array(path)
    # Let other operands have chunks1 and blocks1
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a3-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a3 = blosc2.C2Array(path)
    urlpath = f"ds-0-10-linspace-{dtype.__name__}-{chunks_blocks}-a4-{shape}d.b2nd"
    path = pathlib.Path(f"{ROOT}/{DIR + urlpath}").as_posix()
    a4 = blosc2.C2Array(path)
    assert isinstance(a1, blosc2.C2Array)
    assert isinstance(a2, blosc2.C2Array)
    assert isinstance(a3, blosc2.C2Array)
    assert isinstance(a4, blosc2.C2Array)
    return a1, a2, a3, a4, na1, np.copy(na1), np.copy(na1), np.copy(na1)


@pytest.mark.parametrize("reduce_op", ["sum", pytest.param("all", marks=pytest.mark.heavy)])
def test_reduce_bool(reduce_op, cat2_context):
    shape = (NITEMS_SMALL,)
    chunks_blocks = "default"
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
    expr = a1 + a2 > a3 * a4
    nres = ne_evaluate("na1 + na2 > na3 * na4")
    res = getattr(expr, reduce_op)()
    nres = getattr(nres, reduce_op)()
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "chunks_blocks",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    "reduce_op",
    [pytest.param("prod", marks=pytest.mark.heavy), "min", pytest.param("any", marks=pytest.mark.heavy)],
)
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("keepdims", [True, False])
@pytest.mark.parametrize("dtype_out", [np.int16])
def test_reduce_params(chunks_blocks, axis, keepdims, dtype_out, reduce_op, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
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
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims, dtype=dtype_out)
    else:
        res = getattr(expr, reduce_op)(axis=axis, keepdims=keepdims)
        nres = getattr(nres, reduce_op)(axis=axis, keepdims=keepdims)
    tol = 1e-15 if a1.dtype == "float64" else 1e-6
    np.testing.assert_allclose(res[()], nres, atol=tol, rtol=tol)


# TODO: "any" and "all" are not supported yet because:
# ne_evaluate('(o0 + o1)', local_dict = {'o0': np.array(True), 'o1': np.array(True)})
# is not supported by NumExpr
@pytest.mark.parametrize(
    "chunks_blocks",
    [
        pytest.param((True, True), marks=pytest.mark.heavy),
        (True, False),
        (False, True),
        (False, False),
    ],
)
@pytest.mark.parametrize(
    "reduce_op",
    [
        pytest.param("max", marks=pytest.mark.heavy),
        "mean",
        pytest.param("var", marks=pytest.mark.heavy),
    ],
)
@pytest.mark.parametrize("axis", [0])
def test_reduce_expr_arr(chunks_blocks, axis, reduce_op, cat2_context):
    shape = (60, 60)
    a1, a2, a3, a4, na1, na2, na3, na4 = get_arrays(shape, chunks_blocks)
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
