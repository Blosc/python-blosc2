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

dtype = np.float64
shape = (10_000, 10_000)

@blosc2.dsl_kernel
def kernel_ramp(start):
    return start + _flat_idx  # noqa: F821  # DSL index/shape symbols resolved by miniexpr

t0 = time()
npa = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
print("NumPy arange:", round(time() - t0, 3), "s")

t0 = time()
a1 = blosc2.arange(0, np.prod(shape), dtype=dtype, shape=shape)
print("Blosc2 arange:", round(time() - t0, 3), "s")

t0 = time()
a2 = blosc2.lazyudf(kernel_ramp, (0, ), dtype=dtype, shape=shape)
# a2 = blosc2.lazyudf(kernel_ramp, (0, ), dtype=dtype, shape=shape, jit_backend="cc")
a3 = a2.compute(cparams=dict(clevel=1, codec=blosc2.Codec.LZ4))
print("Blosc2 with DSL kernel (tcc jit backend):", round(time() - t0, 3), "s")

np.testing.assert_array_equal(npa, a3)

t0 = time()
a2 = blosc2.lazyudf(kernel_ramp, (0, ), dtype=dtype, shape=shape, jit_backend="cc")
a3 = a2.compute(cparams=dict(clevel=1, codec=blosc2.Codec.LZ4))
print("Blosc2 with DSL kernel (cc jit backend):", round(time() - t0, 3), "s")

np.testing.assert_array_equal(npa, a3)
