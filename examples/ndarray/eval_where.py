#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate expressions in combination with the where() functionality.

import numpy as np

import blosc2

shape = (50, 50)
chunks = (10, 10)
blocks = (5, 5)

# Create a structured NumPy array
npa = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
npc = npa**2 + npb**2 > 2 * npa * npb + 1
nps = np.empty(shape, dtype=[("a", npa.dtype), ("b", npb.dtype)])
nps["a"] = npa
nps["b"] = npb

s = blosc2.asarray(nps, chunks=chunks, blocks=blocks)
a = blosc2.NDField(s, "a")
b = blosc2.NDField(s, "b")

# Get a LazyExpr instance
c = a**2 + b**2 > 2 * a * b + 1


# Simple where() method
d = c.where(0, 1)
# print(d[:])
np.testing.assert_allclose(d[:], np.where(npc, 0, 1))

d = blosc2.where(c, 0, 1)
# print(d[:])
np.testing.assert_allclose(d[:], np.where(npc, 0, 1))

d = blosc2.lazyexpr(c, where=(0, 1))
# print(d[:])
np.testing.assert_allclose(d[:], np.where(npc, 0, 1))


# Not sure if a decorator like this is a good idea, but it works
@blosc2.lazywhere(0, 1)
def myexpr(a, b):
    return a**2 + b**2 > 2 * a * b + 1


d = myexpr(a, b)
# print(d[:])
np.testing.assert_allclose(d[:], np.where(npc, 0, 1))

# where accepts only a single `x` parameter (not directly supported by NumPy)
d = c.where(s)
npd = d[:]
# print(npd)
np.testing.assert_allclose(npd["a"], nps[npc]["a"])
np.testing.assert_allclose(npd["b"], nps[npc]["b"])


# Decorator version
@blosc2.lazywhere(s)
def myexpr2(a, b):
    return a**2 + b**2 > 2 * a * b + 1


d = myexpr2(a, b)
npd = d[:]
# print(npd)
np.testing.assert_allclose(npd["a"], nps[npc]["a"])
np.testing.assert_allclose(npd["b"], nps[npc]["b"])


# # TODO: Test with no parameters
# d = c.where()
# print(d[:])
# np.testing.assert_allclose(d[:], npc.nonzero())

# NDArray.__getitem__ with LazyExpr (converted into c.where(s) behind the scenes)
d = s[a**2 + b**2 > 2 * a * b + 1]
npd = d[:]
# print(npd)
np.testing.assert_allclose(npd["a"], nps[npc]["a"])
np.testing.assert_allclose(npd["b"], nps[npc]["b"])

# NDArray.__getitem__ with a string expression
d = s["a**2 + b**2 > 2 * a * b + 1"]
npd = d[:]
print(npd)
np.testing.assert_allclose(npd["a"], nps[npc]["a"])
np.testing.assert_allclose(npd["b"], nps[npc]["b"])

# Combined with reductions
d = blosc2.where(c, 0, 1).sum(axis=1)
print(d[...])
np.testing.assert_allclose(d[...], np.where(npc, 0, 1).sum(axis=1))

print("blosc2.where is working correctly!")
