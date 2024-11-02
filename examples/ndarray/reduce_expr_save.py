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

shape = (10, 1, 2)

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa**2 + npb**2 + 2 * npa * npb + 1

a = blosc2.asarray(npa, urlpath="a.b2nd", mode="w")
b = blosc2.asarray(npb, urlpath="b.b2nd", mode="w")

# Get a LazyExpr instance
c = a**2 + b**2 + 2 * a * b + 1
c.save(urlpath="c.b2nd")
c = blosc2.open("c.b2nd")
# Evaluate: output is a NDArray
d = blosc2.lazyexpr("a + c.sum() + a.std()", operands={"a": a, "c": c})
d.save(urlpath="lazy-d.b2nd")

# Load the expression from disk
d = blosc2.open("lazy-d.b2nd")
print(f"Expression: {d}")
assert isinstance(d, blosc2.LazyExpr)
e = d.compute()
assert isinstance(e, blosc2.NDArray)
sum = e[()]
print("Reduction with Blosc2:\n", sum[1])
npsum = npa + np.sum(npc) + np.std(npa)
print("Reduction with NumPy:\n", npsum[1])
assert np.allclose(sum, npsum)

print("NDArray expression evaluated correctly in-memory!")
