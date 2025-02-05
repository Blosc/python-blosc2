#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pytest

import blosc2

import numpy as np

###### General expressions

def expr_nojit(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

@blosc2.jit
def expr_jit(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

# Define the parameters
test_params = [
    ((10, 100), (10, 100,), "float32"),
    ((10, 100), (100,), "float64"),  # using broadcasting
]

@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_expr(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)

    d_jit = expr_jit(a, b, c)
    d_nojit = expr_nojit(a, b, c)

    np.testing.assert_equal(d_jit[...], d_nojit[...])


@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_expr_out(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)
    d_nojit = expr_nojit(a, b, c)

    # Testing jit decorator with an out param
    out = blosc2.zeros(shape, dtype=np.bool_)

    @blosc2.jit(out=out)
    def expr_jit_out(a, b, c):
        return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

    d_jit = expr_jit_out(a, b, c)
    np.testing.assert_equal(d_jit[...], d_nojit[...])
    np.testing.assert_equal(out[...], d_nojit[...])

@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_expr_kwargs(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)
    d_nojit = expr_nojit(a, b, c)

    # Testing jit decorator with kwargs
    cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE])

    @blosc2.jit(**{"cparams": cparams})
    def expr_jit_cparams(a, b, c):
        return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

    d_jit = expr_jit_cparams(a, b, c)
    np.testing.assert_equal(d_jit[...], d_nojit[...])
    assert d_jit.schunk.cparams.clevel == 1
    assert d_jit.schunk.cparams.codec == blosc2.Codec.LZ4
    assert d_jit.schunk.cparams.filters == [blosc2.Filter.BITSHUFFLE] + [blosc2.Filter.NOFILTER] * 5


###### Reductions

def reduc_nojit(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

@blosc2.jit
def reduc_jit(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_reduc(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)

    d_jit = reduc_jit(a, b, c)
    d_nojit = reduc_nojit(a, b, c)

    np.testing.assert_equal(d_jit[...], d_nojit[...])

@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_reduc_out(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)
    d_nojit = reduc_nojit(a, b, c)

    # Testing jit decorator with an out param via the reduction function
    out = np.zeros((shape[0],), dtype=np.int64)

    # Note that out does not work with reductions as the last function call
    @blosc2.jit
    def reduc_jit_out(a, b, c):
        return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1, out=out)

    d_jit = reduc_jit_out(a, b, c)
    np.testing.assert_equal(d_jit[...], d_nojit[...])
    np.testing.assert_equal(out[...], d_nojit[...])

@pytest.mark.parametrize("shape, cshape, dtype", test_params)
def test_reduc_kwargs(shape, cshape, dtype):
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = blosc2.linspace(1, 2, shape[0] * shape[1], dtype=dtype, shape=shape)
    c = blosc2.linspace(-10, 10, cshape[0], dtype=dtype, shape=cshape)
    d_nojit = reduc_nojit(a, b, c)

    # Testing jit decorator with kwargs via an out param in the reduction function
    cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE])
    out = blosc2.zeros((shape[0],), dtype=np.int64, cparams=cparams)

    @blosc2.jit
    def reduc_jit_cparams(a, b, c):
        return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1, out=out)

    d_jit = reduc_jit_cparams(a, b, c)
    np.testing.assert_equal(d_jit[...], d_nojit[...])
    assert d_jit.schunk.cparams.clevel == 1
    assert d_jit.schunk.cparams.codec == blosc2.Codec.LZ4
    assert d_jit.schunk.cparams.filters == [blosc2.Filter.BITSHUFFLE] + [blosc2.Filter.NOFILTER] * 5
