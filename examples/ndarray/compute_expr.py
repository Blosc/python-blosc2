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

print("NDArray expression evaluated correctly in-memory!")

# Now, evaluate the expression from operands in disk
# TODO: when doing a copy, mode should be 'w' by default?
da = a.copy(urlpath="a.b2nd", mode="w")
db = b.copy(urlpath="b.b2nd", mode="w")

# Get a LazyExpr instance
(da**2 + db**2 + 2 * da * db + 1).save(urlpath="c.b2nd")
dc = blosc2.open("c.b2nd")

# Evaluate: output is a NDArray
dc2 = dc.compute()
# Check
assert isinstance(dc2, blosc2.NDArray)
assert np.allclose(dc2[:], npc)
print("NDArray expression evaluated correctly on-disk!")
