#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how values in expressions support broadcast with NDArray/NumPy arrays as operands

import numpy as np

import blosc2

shape = (2, 5)

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa + npb.mean(axis=0)
# npres = npa + np.mean(npa, axis=0)
npres = npa + np.mean(npc, axis=0)
print("Broadcast with NumPy:\n", npres)

a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

# Get a LazyExpr instance
c = a + b.mean(axis=0)
# Evaluate: output is a NDArray
# d = a + blosc2.mean(a, axis=0)
# d = a + np.mean(npa, axis=0)
d = a + blosc2.mean(c, axis=0)
# print(d, d.shape, d.dtype)
# print(d.expression, d.operands)
assert isinstance(d, blosc2.LazyExpr)
e = d.eval()
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
