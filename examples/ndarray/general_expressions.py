#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to build expressions with a general mix of NDArray and NumPy operands.

import numpy as np

import blosc2

shape = (50, 50)

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa**2 + npb**2 + 2 * npa * npb + 1

a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

# Get a LazyExpr instance with all NDArray operands
c = blosc2.lazyexpr("a**2 + b**2 + 2 * a * b + 1", {"a": a, "b": b})
d = c.compute()
assert np.allclose(d[:], npc)

# A LazyExpr instance with a mix of NDArray and NumPy operands
c = blosc2.lazyexpr("a**2 + b**2 + 2 * a * b + 1", {"a": npa, "b": b})
d = c.compute()
assert np.allclose(d[:], npc)

# A LazyExpr instance with a all NumPy operands
c = blosc2.lazyexpr("a**2 + b**2 + 2 * a * b + 1", {"a": npa, "b": npb})
d = c.compute()
assert np.allclose(d[:], npc)

# Evaluate partial slices
npd = c[1]
# Check
assert np.allclose(npd, npc[1])

npd = c[1:10]
# Check
assert np.allclose(npd, npc[1:10])

print(d.info)

print("Lazy expression evaluated correctly in-memory!")
