#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compare arange constructor times wrt DSL kernels.

from time import time
import numpy as np

import blosc2

dtype = np.int64
shape = (10_000, 10_000)
start, stop = 1, 2
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

@blosc2.dsl_kernel
def kernel_ramp(start, stop, nitems):  # noqa
    step = (float(stop) - float(start)) / float(nitems - 1)
    return float(start) + _flat_idx * step  # noqa: F821  # DSL index/shape symbols resolved by miniexpr

t0 = time()
npa = np.linspace(start, stop, np.prod(shape), dtype=dtype).reshape(shape)
print("NumPy arange:", round(time() - t0, 3), "s")
#print(npa)

t0 = time()
a1 = blosc2.linspace(start, stop, np.prod(shape), dtype=dtype, shape=shape, cparams=cparams)
print("Blosc2 arange:", round(time() - t0, 3), "s")

np.testing.assert_array_equal(a1, npa)

t0 = time()
a2 = blosc2.lazyudf(kernel_ramp, (start, stop, np.prod(shape)), dtype=dtype, shape=shape)
# a2 = blosc2.lazyudf(kernel_ramp, (0, ), dtype=dtype, shape=shape, jit_backend="cc")
a3 = a2.compute(cparams=cparams)
print("Blosc2 with DSL kernel (tcc jit backend):", round(time() - t0, 3), "s")

np.testing.assert_array_equal(a3, npa)

t0 = time()
a2 = blosc2.lazyudf(kernel_ramp, (start, stop, np.prod(shape)), dtype=dtype, shape=shape, jit_backend="cc")
a3 = a2.compute(cparams=cparams)
print("Blosc2 with DSL kernel (cc jit backend):", round(time() - t0, 3), "s")

np.testing.assert_array_equal(a3, npa)
