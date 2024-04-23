#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark to evaluate expressions with numba and NDArray instances as operands.
# As numba takes a while to compile the first time, a warm-up is done before the
# actual benchmark.

from time import time

import numba as nb
import numexpr as ne
import numpy as np

import blosc2

shape = (4000, 10_000)
chunks = [200, 10_000]
blocks = [20, 10_000]
dtype = np.float64

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
npc = npa + 1

t0 = time()
npc = npa + 1
print("NumPy took %.3f s" % (time() - t0))

ne.set_num_threads(1)
ne.evaluate("npa + 1", out=np.empty_like(npa))
t0 = time()
ne.evaluate("npa + 1", out=np.empty_like(npa))
print("NumExpr took %.3f s" % (time() - t0))

# a = blosc2.SChunk()
# a.append_data(npa)

blosc2.cparams_dflts["codec"] = blosc2.Codec.BLOSCLZ
blosc2.cparams_dflts["clevel"] = 1
a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

# Get a LazyExpr instance
c = a + 1
# Evaluate!  Output is a NDArray
d = c.evaluate()
t0 = time()
d = c.evaluate()
print("Blosc2+numexpr took %.3f s" % (time() - t0))
# Check
assert np.allclose(d[:], npc)


@nb.jit(nopython=True, parallel=True)
def func_numba(x):
    # return x + 1
    out = np.empty(x.shape, x.dtype)
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            out[i, j] = x[i, j] + 1
    return out


nb_res = func_numba(npa)
t0 = time()
nb_res = func_numba(npa)
print("Numba took %.3f s" % (time() - t0))


@nb.jit(nopython=True, parallel=True)
def udf_numba(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    # output[:] = x + 1
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            output[i, j] = x[i, j] + 1

# warm up
expr = blosc2.expr_from_udf(udf_numba, ((npa, npa.dtype),), npa.dtype, chunks=chunks, blocks=blocks)
res = expr.eval()
expr = blosc2.expr_from_udf(udf_numba, ((npa, npa.dtype),), npa.dtype, chunks=chunks, blocks=blocks)
# actual benchmark
t0 = time()
res = expr.eval()
#res = expr[:]  # 1.6x slower with prefilters; will try with postfilters in the future
print("Blosc2+numba took %.3f s" % (time() - t0))
# print(res.info)


# assert np.allclose(res[...], npc)
tol = 1e-5 if dtype is np.float32 else 1e-14
if dtype in (np.float32, np.float64):
    np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)
