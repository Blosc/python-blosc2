#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark to evaluate expressions with numba and NDArray instances as operands.
# As numba takes a while to compile the first time, we use cached functions, so
# make sure to run the script at least a couple of times.

from time import time

import numba as nb
import numexpr as ne
import numpy as np

import blosc2


shape = (5000, 10_000)
chunks = [500, 10_000]
blocks = [20, 10_000]
dtype = np.float32

# Expression to evaluate
exprs = ("x + 1",
         "x**2 + y**2 + 2 * x * y + 1",
         "sin(x)**3 + cos(y)**2 + cos(x) * sin(y) + z",
         )


# Create input arrays
npx = np.linspace(0, 1, np.prod(shape)).reshape(shape)
npy = np.linspace(-1, 1, np.prod(shape)).reshape(shape)
npz = np.linspace(0, 10, np.prod(shape)).reshape(shape)
vardict = {"x": npx, "y": npy, "z": npz, "np": np}
x = blosc2.asarray(npx, chunks=chunks, blocks=blocks)
y = blosc2.asarray(npy, chunks=chunks, blocks=blocks)
z = blosc2.asarray(npz, chunks=chunks, blocks=blocks)
b2vardict = {"x": x, "y": y, "z": z, "blosc2": blosc2}

# Define the functions to evaluate the expressions
# First the pure numba+numpy version
@nb.jit(nopython=True, parallel=True, cache=True)
def func_numba(x, y, z, expr):
    output = np.empty(x.shape, x.dtype)
    if expr == exprs[0]:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j] + 1
    elif expr == exprs[1]:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j]**2 + y[i, j]**2 + 2 * x[i, j] * y[i, j] + 1
    elif expr == exprs[2]:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = np.sin(x[i, j])**3 + np.cos(y[i, j])**2 + np.cos(x[i, j]) * np.sin(y[i, j]) + z[i, j]
    return output


# Now, the numba+blosc2 version using an udf
@nb.jit(nopython=True, parallel=True, cache=True)
def udf_numba(inputs, output, offset):
    icount = len(inputs)
    x = inputs[0]
    if icount == 1:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j] + 1
    elif icount == 2:
        y = inputs[1]
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j]**2 + y[i, j]**2 + 2 * x[i, j] * y[i, j] + 1
    elif icount == 3:
        y = inputs[1]
        z = inputs[2]
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = np.sin(x[i, j])**3 + np.cos(y[i, j])**2 + np.cos(x[i, j]) * np.sin(y[i, j]) + z[i, j]


for expr in exprs:
    print(f"*** Evaluating expression: {expr} ...")

    # Evaluate the expression with NumPy/numexpr
    npexpr = expr.replace("sin", "np.sin").replace("cos", "np.cos")
    t0 = time()
    npres = eval(npexpr, vardict)
    print("NumPy took %.3f s" % (time() - t0))
    # ne.set_num_threads(1)
    # nb.set_num_threads(1)  # this does not work that well; better use the NUMBA_NUM_THREADS env var
    t0 = time()
    ne.evaluate(expr, vardict, out=np.empty_like(npx))
    print("NumExpr took %.3f s" % (time() - t0))

    # Evaluate the expression with Blosc2+numexpr
    blosc2.cparams_dflts["codec"] = blosc2.Codec.LZ4
    blosc2.cparams_dflts["clevel"] = 5
    b2expr = expr.replace("sin", "blosc2.sin").replace("cos", "blosc2.cos")
    c = eval(b2expr, b2vardict)
    t0 = time()
    d = c.eval()
    print("Blosc2+numexpr+eval took %.3f s" % (time() - t0))
    # Check
    np.testing.assert_allclose(d[:], npres)
    t0 = time()
    d = c[:]
    print("Blosc2+numexpr+getitem took %.3f s" % (time() - t0))
    # Check
    np.testing.assert_allclose(d[:], npres)

    # nb.set_num_threads(1)
    t0 = time()
    res = func_numba(npx, npy, npz, expr)
    print("Numba took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res, npres)

    if expr == exprs[0]:
        inputs = (npx,)
    elif expr == exprs[1]:
        inputs = (npx, npy)
    elif expr == exprs[2]:
        inputs = (npx, npy, npz)

    expr_ = blosc2.lazyudf(udf_numba, inputs, npx.dtype,
                           chunks=chunks, blocks=blocks)
    # actual benchmark
    # eval() uses the udf function as a prefilter
    t0 = time()
    res = expr_.eval()
    print("Blosc2+numba+eval took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres)
    # getitem uses the same compiled function but as a postfilter
    t0 = time()
    res = expr_[:]
    print("Blosc2+numba+getitem took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres)
