#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2
from blosc2.lazyexpr import ne_evaluate

###### General expressions

# Define the parameters
test_params = [
    ((10, 100), (10, 100), "float32"),
    ((10, 100), (100,), "float64"),  # using broadcasting
]


@pytest.fixture(params=test_params)
def sample_data(request):
    shape, cshape, dtype = request.param
    # The jit decorator can work with any numpy or NDArray params in functions
    a = blosc2.linspace(0, 1, shape[0] * shape[1], dtype=dtype, shape=shape)
    b = np.linspace(1, 2, shape[0] * shape[1], dtype=dtype).reshape(shape)
    c = blosc2.linspace(-10, 10, np.prod(cshape), dtype=dtype, shape=cshape)
    return a, b, c, shape


def test_expr(sample_data):
    a, b, c, shape = sample_data
    d_blosc2 = blosc2.evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)")
    d_numexpr = ne_evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)")
    np.testing.assert_equal(d_blosc2, d_numexpr)


# skip this test for WASM for now
@pytest.mark.skipif(blosc2.IS_WASM, reason="Skip test for WASM")
def test_expr_out(sample_data):
    a, b, c, shape = sample_data
    # Testing with an out param
    out = blosc2.zeros(shape, dtype="bool")
    d_blosc2 = blosc2.evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)", out=out)
    out2 = np.zeros(shape, dtype=np.bool_)
    d_numexpr = ne_evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)", out=out2)
    np.testing.assert_equal(d_blosc2, d_numexpr)
    np.testing.assert_equal(out, out2)


def test_expr_optimization(sample_data):
    a, b, c, shape = sample_data
    d_blosc2 = blosc2.evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)", optimization="none")
    d_numexpr = ne_evaluate("((a**3 + sin(a * 2)) < c) & (b > 0)", optimization="none")
    np.testing.assert_equal(d_blosc2, d_numexpr)


###### Reductions


def test_reduc(sample_data):
    a, b, c, shape = sample_data
    d_blosc2 = blosc2.evaluate("sum(((a**3 + sin(a * 2)) < c) & (b > 0), axis=1)")
    a = a[:]
    b = b[:]
    c = c[:]  # ensure that all operands are numpy arrays
    d_numpy = np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1)
    np.testing.assert_equal(d_blosc2, d_numpy)


def test_reduc_out(sample_data):
    a, b, c, shape = sample_data
    # Testing with an out param
    out = blosc2.zeros(shape[0], dtype=np.int64)
    # Both versions below should work
    d_blosc2 = blosc2.evaluate("sum(((a**3 + sin(a * 2)) < c) & (b > 0), axis=1)", out=out)
    out2 = out[:]
    d_blosc2_ = blosc2.evaluate("sum(((a**3 + sin(a * 2)) < c) & (b > 0), axis=1, out=out2)")
    a = a[:]
    b = b[:]
    c = c[:]  # ensure that all operands are numpy arrays
    out3 = out[:]
    d_numpy = np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1, out=out3)
    np.testing.assert_equal(d_blosc2, d_numpy)
    np.testing.assert_equal(d_blosc2_, d_numpy)
    np.testing.assert_equal(out, out2)
    np.testing.assert_equal(out, out3)


###### NumPy functions


@pytest.mark.parametrize("func", ["cumsum", "cumulative_sum", "cumprod"])
def test_numpy_funcs(sample_data, func):
    a, b, c, shape = sample_data
    try:
        npfunc = getattr(np, func)
        d_blosc2 = blosc2.evaluate(f"{func}(((a**3 + sin(a * 2)) < c) & (b > 0), axis=0)")
        a = a[:]
        b = b[:]
        c = c[:]  # ensure that all operands are numpy arrays
        d_numpy = npfunc(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)
        np.testing.assert_equal(d_blosc2, d_numpy)
    except AttributeError:
        pytest.skip("NumPy version has no cumulative_sum function.")
