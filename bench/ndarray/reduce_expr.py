#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark to compute expressions with numba and NDArray instances as operands.
# As numba takes a while to compile the first time, we use cached functions, so
# make sure to run the script at least a couple of times.

from time import time

import numexpr as ne
import numpy as np

import blosc2

shape = (100, 100, 10_000)
chunks = [10, 100, 10_000]
blocks = [4, 10, 1_000]
# Comment out the next line to force chunks and blocks above
chunks, blocks = None, None
dtype = np.float32
rtol = 1e-5 if dtype == np.float32 else 1e-16
atol = 1e-5 if dtype == np.float32 else 1e-16

# Axis to reduce
laxis = (None, 0, 1, 2, (0, 2))

# cparams defaults
blosc2.cparams_dflts["codec"] = blosc2.Codec.LZ4
blosc2.cparams_dflts["clevel"] = 5

# Create input arrays
npx = np.linspace(0, 1, np.prod(shape), dtype=dtype).reshape(shape)
npy = np.linspace(-1, 1, np.prod(shape), dtype=dtype).reshape(shape)
npz = np.linspace(0, 10, np.prod(shape), dtype=dtype).reshape(shape)
vardict = {"x": npx, "y": npy, "z": npz, "np": np}
x = blosc2.asarray(npx, chunks=chunks, blocks=blocks)
y = blosc2.asarray(npy, chunks=chunks, blocks=blocks)
z = blosc2.asarray(npz, chunks=chunks, blocks=blocks)
print(f"*** cratios: x={x.schunk.cratio:.2f}x, y={y.schunk.cratio:.2f}x, z={z.schunk.cratio:.2f}x")

expr = "(x**2 + y**2 * z** 2) < 1"


for axis in laxis:
    print(f"*** Computing expression on axis: {axis} ...")

    # Compute the reduction with NumPy/numexpr
    npexpr = expr.replace("sin", "np.sin").replace("cos", "np.cos")
    t0 = time()
    npres = eval(npexpr, vardict).sum(axis=axis)
    tref = time() - t0
    print("NumPy took %.3f s" % tref)
    # ne.set_num_threads(1)
    # nb.set_num_threads(1)  # this does not work that well; better use the NUMBA_NUM_THREADS env var
    t0 = time()
    out = ne.evaluate(expr, vardict).sum(axis=axis)
    t1 = time() - t0
    print(f"NumExpr took {t1:.3f} s; {tref / t1:.1f}x wrt NumPy")

    # Reduce with Blosc2
    c = eval(expr)
    t0 = time()
    d = c.compute()
    d = d.sum(axis=axis)
    t1 = time() - t0
    print(f"LazyExpr+compute took {t1:.3f} s; {tref / t1:.1f}x wrt NumPy")
    # Check
    np.testing.assert_allclose(d[()], npres, rtol=rtol, atol=atol)
    t0 = time()
    d = c[:]
    d = d.sum(axis=axis)
    t1 = time() - t0
    print(f"LazyExpr+getitem took {t1:.3f} s; {tref / t1:.1f}x wrt NumPy")
    # Check
    np.testing.assert_allclose(d[()], npres, rtol=rtol, atol=atol)
