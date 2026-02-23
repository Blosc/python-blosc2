#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compare reduction performance on DSL kernels.
# This uses the special _global_linear_idx var.

from time import time
import numpy as np

import blosc2

dtype = np.float64
shape = (10_000, 10_000)

@blosc2.dsl_kernel
def kernel_ramp():
    # return _i0 * _n1 + _i1  # noqa: F821  # DSL index/shape symbols resolved by miniexpr
    return _global_linear_idx  # noqa: F821  # DSL index/shape symbols resolved by miniexpr

print(kernel_ramp.dsl_source)
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
npa = a.compute(cparams=dict(clevel=1, codec=blosc2.Codec.LZ4))
t0 = time()
result = npa.sum()
# print(result)
print("Blosc2 sum over NDArray:", round(time() - t0, 3), "s")

t0 = time()
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
result = a.sum()
# print(result)
print("Blosc2 sum over LazyArray:", round(time() - t0, 3), "s")

t0 = time()
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
# result = a.compute(cparams=dict(clevel=1, codec=blosc2.Codec.LZ4)).sum()
result = a.compute().sum()
# print(result)
print("(with a prior .compute):", round(time() - t0, 3), "s")

t0 = time()
npa = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
result = npa.sum()
# print(result)
print("NumPy arange + sum:", round(time() - t0, 3), "s")
