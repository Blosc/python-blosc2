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

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)

t0 = time()
npc = npa + 1
print("NumPy took %.3f s" % (time() - t0))

# ne.set_num_threads(1)
# nb.set_num_threads(1)  # this does not work that well; better use the NUMBA_NUM_THREADS env var
t0 = time()
ne.evaluate("npa + 1", out=np.empty_like(npa))
print("NumExpr took %.3f s" % (time() - t0))

blosc2.cparams_dflts["codec"] = blosc2.Codec.LZ4
blosc2.cparams_dflts["clevel"] = 5
a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

# Get a LazyExpr instance
c = a + 1
t0 = time()
d = c.eval()
print("Blosc2+numexpr+eval took %.3f s" % (time() - t0))
# Check
np.testing.assert_allclose(d[:], npc)
t0 = time()
d = c[:]
print("Blosc2+numexpr+getitem took %.3f s" % (time() - t0))
# Check
np.testing.assert_allclose(d[:], npc)


@nb.jit(nopython=True, parallel=True, cache=True)
def func_numba(x):
    # return x + 1
    out = np.empty(x.shape, x.dtype)
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            out[i, j] = x[i, j] + 1
    return out


# nb.set_num_threads(1)
t0 = time()
nb_res = func_numba(npa)
print("Numba took %.3f s" % (time() - t0))


@nb.jit(nopython=True, parallel=True, cache=True)
def udf_numba(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    # output[:] = x + 1
    # return
    for i in nb.prange(x.shape[0]):
        for j in nb.prange(x.shape[1]):
            output[i, j] = x[i, j] + 1


expr = blosc2.expr_from_udf(udf_numba, ((npa, npa.dtype),), npa.dtype,
                            chunks=chunks, blocks=blocks)
# actual benchmark
t0 = time()
res = expr.eval()
print("Blosc2+numba+eval took %.3f s" % (time() - t0))
expr = blosc2.expr_from_udf(udf_numba, ((npa, npa.dtype),), npa.dtype,
                            chunks=chunks, blocks=blocks)
# getitem uses the same compiled function but as a postfilter
t0 = time()
res = expr[:]
print("Blosc2+numba+getitem took %.3f s" % (time() - t0))

np.testing.assert_allclose(res[...], npc)
