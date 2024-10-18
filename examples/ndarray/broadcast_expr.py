#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate expressions with NDArray instances having different shapes as operands.
# The broadcast is done internally and tries to mimic NumPy behavior.

import numpy as np

import blosc2

# Two operands with the next shapes should be supported
# shape1, shape2 = (2, 1, 3, 2), (3, 3, 2)
# shape1, shape2 = (2, 1, 3, 2), (3, 1, 2)
shape1, shape2 = (2, 1, 1, 1), (3, 2, 2)

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape1), dtype=np.float32).reshape(shape1)
npb = np.linspace(1, 2, np.prod(shape2), dtype=np.float64).reshape(shape2)
npc = npa + npb
npres = npa + npb
print("Broadcast with NumPy:\n", npres)

a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

# Get a LazyExpr instance
c = a + b
# Evaluate: output is a NDArray
# d = a + blosc2.mean(a, axis=0)
# d = a + np.mean(npa, axis=0)
d = a + b
# print(d, d.shape, d.dtype)
# print(d.expression, d.operands)
assert isinstance(d, blosc2.LazyExpr)
e = d.compute()
print(e)
assert isinstance(d, blosc2.LazyExpr)
# Check
assert isinstance(e, blosc2.NDArray)
res = e[:]
print("Broadcast with Blosc2:\n", res)

assert np.allclose(res, npres)

# # Evaluate a slice: output is a NumPy array
npd = d[:]
# # Check
assert np.allclose(npd, npres)

print("NDArray expression evaluated correctly in-memory!")
