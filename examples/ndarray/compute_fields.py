#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate expressions with NDField instances as operands.

import numpy as np

import blosc2

shape = (50, 50)

# Create a structured NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa**2 + npb**2 > 2 * npa * npb + 1
nps = np.empty(shape, dtype=[("a", npa.dtype), ("b", npb.dtype)])
nps["a"] = npa
nps["b"] = npb

s = blosc2.asarray(nps)
a = blosc2.NDField(s, "a")
b = blosc2.NDField(s, "b")

# Get a LazyExpr instance
c = a**2 + b**2 > 2 * a * b + 1

# Evaluate: output is a NDArray
d = c.compute()
# Check
assert isinstance(d, blosc2.NDArray)
assert np.allclose(d[:], npc)

# Evaluate the whole slice: output is a NumPy array
npd = c[:]
# Check
assert isinstance(npd, np.ndarray)
assert np.allclose(npd, npc)

# Evaluate a partial slice: output is a NumPy array
npd = c[1:10]
# Check
assert isinstance(npd, np.ndarray)
assert np.allclose(npd, npc[1:10])

print("Expression with NDField operands evaluated correctly!")
