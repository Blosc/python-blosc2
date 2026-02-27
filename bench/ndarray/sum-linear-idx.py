#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compare reduction performance on DSL kernels.
# This uses the special _flat_idx var.

from time import time
import numpy as np

import blosc2

dtype = np.int64
shape = (10_000, 10_000)
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

@blosc2.dsl_kernel
def kernel_ramp():
    # return _i0 * _n1 + _i1  # noqa: F821  # DSL index/shape symbols resolved by miniexpr
    return _flat_idx  # noqa: F821  # DSL index/shape symbols resolved by miniexpr

print(kernel_ramp.dsl_source)
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
npa = a.compute(cparams=cparams)
t0 = time()
result = npa.sum()
# print(result)
print("Blosc2 sum over NDArray:", round(time() - t0, 3), "s")

t0 = time()
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
result = a.sum(cparams=cparams)
# print(result)
print("Blosc2 sum over LazyArray:", round(time() - t0, 3), "s")

t0 = time()
a = blosc2.lazyudf(kernel_ramp, (), dtype=dtype, shape=shape)
result = a.compute(cparams=cparams).sum()
# print(result)
print("(with a prior .compute):", round(time() - t0, 3), "s")

t0 = time()
a = blosc2.arange(np.prod(shape), dtype=dtype, shape=shape, cparams=cparams)
result = a.sum()
# print(result)
print("Blosc2 arange + sum:", round(time() - t0, 3), "s")

t0 = time()
npa = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
result = npa.sum()
# print(result)
print("NumPy arange + sum:", round(time() - t0, 3), "s")
