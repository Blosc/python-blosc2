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
blocks = [4, 10_000]
# Comment out the next line to force chunks and blocks above
chunks, blocks = None, None
# Check with fast compression
cparams = dict(clevel=1, codec=blosc2.Codec.BLOSCLZ)

dtype = np.float32
rtol = 1e-6 if dtype == np.float32 else 1e-17
atol = 1e-6 if dtype == np.float32 else 1e-17

# Expression to evaluate
exprs = ("x + 1",
         "x**2 + y**2 + 2 * x * y + 1",
         "sin(x)**3 + cos(y)**2 + cos(x) * sin(y) + z",
         )


# Create input arrays
npx = np.linspace(0, 1, np.prod(shape), dtype=dtype).reshape(shape)
npy = np.linspace(-1, 1, np.prod(shape), dtype=dtype).reshape(shape)
npz = np.linspace(0, 10, np.prod(shape), dtype=dtype).reshape(shape)
vardict = {"x": npx, "y": npy, "z": npz, "np": np}
x = blosc2.asarray(npx, chunks=chunks, blocks=blocks, cparams=cparams)
y = blosc2.asarray(npy, chunks=chunks, blocks=blocks, cparams=cparams)
z = blosc2.asarray(npz, chunks=chunks, blocks=blocks, cparams=cparams)
b2vardict = {"x": x, "y": y, "z": z, "blosc2": blosc2}

print(f"shape: {x.shape}, chunks: {x.chunks}, blocks: {x.blocks}, cratio: {x.schunk.cratio:.2f}")

# Define the functions to evaluate the expressions
# First the pure numba+numpy version
@nb.jit(parallel=True, cache=True)
def func_numba(x, y, z, n):
    output = np.empty(x.shape, x.dtype)
    if n == 0:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j] + 1
    elif n == 1:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = x[i, j]**2 + y[i, j]**2 + 2 * x[i, j] * y[i, j] + 1
    elif n == 2:
        for i in nb.prange(x.shape[0]):
            for j in nb.prange(x.shape[1]):
                output[i, j] = np.sin(x[i, j])**3 + np.cos(y[i, j])**2 + np.cos(x[i, j]) * np.sin(y[i, j]) + z[i, j]
    return output


# Now, the numba+blosc2 version using an udf
@nb.jit(parallel=True, cache=True)
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


for n, expr in enumerate(exprs):
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
    print("LazyExpr+eval took %.3f s" % (time() - t0))
    # Check
    np.testing.assert_allclose(d[:], npres, rtol=rtol, atol=atol)
    t0 = time()
    d = c[:]
    print("LazyExpr+getitem took %.3f s" % (time() - t0))
    # Check
    np.testing.assert_allclose(d[:], npres, rtol=rtol, atol=atol)

    # nb.set_num_threads(1)
    t0 = time()
    res = func_numba(npx, npy, npz, n)
    print("Numba took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res, npres, rtol=rtol, atol=atol)

    inputs = (x,)
    if n == 1:
        inputs = (x, y)
    elif n == 2:
        inputs = (x, y, z)

    expr_ = blosc2.lazyudf(udf_numba, inputs, npx.dtype, chunked_eval=False,
                           chunks=chunks, blocks=blocks, cparams=cparams)
    # actual benchmark
    # eval() uses the udf function as a prefilter
    t0 = time()
    res = expr_.eval()
    print("LazyUDF+eval took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres, rtol=rtol, atol=atol)
    # getitem uses the same compiled function but as a postfilter
    t0 = time()
    res = expr_[:]
    print("LazyUDF+getitem took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres, rtol=rtol, atol=atol)

    expr_ = blosc2.lazyudf(udf_numba, inputs, npx.dtype, chunked_eval=True,
                           chunks=chunks, blocks=blocks, cparams=cparams)
    # getitem but using chunked evaluation
    t0 = time()
    res = expr_.eval()
    print("LazyUDF+chunked_eval took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres, rtol=rtol, atol=atol)
    t0 = time()
    res = expr_[:]
    print("LazyUDF+getitem+chunked_eval took %.3f s" % (time() - t0))
    np.testing.assert_allclose(res[...], npres, rtol=rtol, atol=atol)
