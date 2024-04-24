#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest
import numba as nb

import blosc2


@nb.jit(nopython=True, parallel=True)
def numba1p(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    output[:] = x + 1


@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        # Test different shapes with and without padding
        ((10, 10), (10, 10), (10, 10),),
        ((20, 20), (10, 10), (10, 10),),
        ((20, 20), (10, 10), (5, 5),),
        ((13, 13), (10, 10), (10, 10),),
        ((13, 13), (10, 10), (5, 5),),
        ((10, 10), (10, 10), (4, 4),),
        ((13, 13), (10, 10), (4, 4),),
    ],
)
def test_lazyexpr_udf_1p(shape, chunks, blocks):
    npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    npc = npa + 1

    expr = blosc2.expr_from_udf(numba1p, ((npa, npa.dtype), ), npa.dtype, chunks=chunks, blocks=blocks)
    res = expr.eval()

    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)


@nb.jit(nopython=True, parallel=True)
def numba2p(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    y = inputs_tuple[1]
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            output[i, j] = x[i, j] ** 2 + y[i, j] ** 2 + 2 * x[i, j] * y[i, j] + 1


@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ((20, 20), (10, 10), (5, 5),),
        ((13, 13, 10), (10, 10, 5), (5, 5, 3),),
        ((13, 13), (10, 10), (5, 5),),
    ],
)
def test_lazyexpr_udf_2p(shape, chunks, blocks):
    npa = np.arange(0, np.prod(shape)).reshape(shape)
    npb = np.arange(1, np.prod(shape) + 1).reshape(shape)
    npc = npa**2 + npb**2 + 2 * npa * npb + 1

    b = blosc2.asarray(npb)
    expr = blosc2.expr_from_udf(numba2p, ((npa, npa.dtype), (b, b.dtype)), npa.dtype, chunks=chunks, blocks=blocks)
    res = expr.eval()

    np.testing.assert_allclose(res[...], npc)


@nb.jit(nopython=True, parallel=True)
def numba1dim(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    y = inputs_tuple[1]
    z = inputs_tuple[2]
    output[:] = x + y + z


# Test with np.ndarray, blosc2.SChunk and python scalar operands
@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ((20, ), (10, ), (5, ),),
        ((23, ), (10, ), (3, ),),
    ],
)
def test_lazyexpr_udf_1dim(shape, chunks, blocks):
    npa = np.arange(start=0, stop=np.prod(shape)).reshape(shape)
    npb = np.arange(start=1, stop=np.prod(shape) + 1).reshape(shape)
    py_scalar = np.e
    npc = npa + npb + py_scalar

    b = blosc2.SChunk(data=npb)
    expr = blosc2.expr_from_udf(numba1dim, ((npa, npa.dtype), (b, npb.dtype), (py_scalar, np.float64)),
                                np.float64, chunks=chunks, blocks=blocks)
    res = expr.eval()

    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)


def test_lazyexpr_udf_params():
    shape = (23, )
    npa = np.arange(start=0, stop=np.prod(shape)).reshape(shape)
    py_scalar = np.e
    a = blosc2.asarray(npa)
    schunk = blosc2.SChunk(data=npa)

    # Assert that shape is computed correctly
    npc = npa + 1
    cparams = {'nthreads': 4}
    expr = blosc2.expr_from_udf(numba1p, ((schunk, npa.dtype), ), np.float64, cparams=cparams)
    res = expr.eval()
    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)
    assert res.shape == npa.shape
    assert res.schunk.cparams['nthreads'] == cparams['nthreads']

    expr = blosc2.expr_from_udf(numba1p, ((py_scalar, np.float64), ), np.float64)
    res = expr.eval()
    npc = py_scalar + 1
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)

    npc = np.full(shape, py_scalar, np.float64) + 1
    expr = blosc2.expr_from_udf(numba1p, ((py_scalar, np.float64), ), np.float64, shape)
    res = expr.eval()
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)
    assert res.shape == npc.shape
