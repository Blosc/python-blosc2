#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate expressions with NDArray instances as operands.

import numpy as np

import blosc2

shape = (10, 10, 2)

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa**2 + npb**2 + 2 * npa * npb + 1

a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

# Get a LazyExpr instance
c = a**2 + b**2 + 2 * a * b + 1
# Evaluate: output is a NDArray
# d = c.sum(axis=1)
# d = blosc2.sum(c, axis=1)
# d = blosc2.sum(c) + blosc2.mean(a)
# d = blosc2.sum(c, axis=1) + blosc2.mean(a, axis=0)
# d = blosc2.sum(c, axis=(0, 2)) + blosc2.mean(a, axis=(0, 2))
# d = blosc2.sum(c) + blosc2.std(a, axis=1)
d = blosc2.any(c, axis=(0, 2)) < b.slice((0, slice(0, 10), 0))
print(d, d.shape, d.dtype)
# print(d.expression, d.operands)
e = d.compute()
# print(e)
assert isinstance(d, blosc2.LazyExpr)

# Check
assert isinstance(e, blosc2.NDArray)
sum = e[()]
print("Reduction with Blosc2:\n", sum)
# npsum = npc.sum(axis=1)
# npsum = np.sum(npc, axis=1)
# npsum = np.sum(npc) + np.mean(npa)
# npsum = np.sum(npc, axis=1) + np.mean(npa, axis=0)
# npsum = np.sum(npc, axis=(0, 2)) + np.mean(npa, axis=(0, 2))
# npsum = np.sum(npc) + np.std(npa)
npsum = np.any(npc, axis=(0, 2)) < npb[0, :, 0]
print("Reduction with NumPy:\n", npsum)
# npsum = np.sum(npc, axis=(0,2)) + np.std(npa, axis=(0, 2))
assert np.allclose(sum, npsum)

# # Evaluate a slice: output is a NumPy array
npd = d[()]
# # Check
assert np.allclose(npd, npsum)

print("NDArray expression evaluated correctly in-memory!")
