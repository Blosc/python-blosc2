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

shape = (50, 50)

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
d = (blosc2.sum(c, axis=1) + blosc2.sum(a, axis=0)).eval()
# Check
assert isinstance(d, blosc2.NDArray)
sum = d[()]
print(sum)
# npsum = npc.sum(axis=1)
# npsum = np.sum(npc, axis=1)
npsum = np.sum(npc, axis=1) + np.sum(npa, axis=0)
assert np.allclose(sum, npsum)

# # Evaluate a slice: output is a NumPy array
# npd = c[:]
# # Check
# assert isinstance(npd, np.ndarray)
# assert np.allclose(npd, npc)
#
# print("NDArray expression evaluated correctly in-memory!")
