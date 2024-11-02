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
d = blosc2.lazyexpr("sl + c.sum() + a.std()", operands={"a": a, "c": c, "sl": a.slice((1, 1))})
print(f"Expression: {d.expression}")
print(f"Operands: {d.operands}")
assert isinstance(d, blosc2.LazyExpr)
e = d.compute()
assert isinstance(d, blosc2.LazyExpr)
# Check
assert isinstance(e, blosc2.NDArray)
sum = e[()]
print("Reduction with Blosc2:\n", sum)
npsum = npa[1, 1] + np.sum(npc) + np.std(npa)
print("Reduction with NumPy:\n", npsum)
# npsum = np.sum(npc, axis=(0,2)) + np.std(npa, axis=(0, 2))
assert np.allclose(sum, npsum)

print("NDArray expression evaluated correctly in-memory!")
