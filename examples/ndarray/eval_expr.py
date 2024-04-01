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
dtype = np.float64

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
npb = np.linspace(1, 2, np.prod(shape)).reshape(shape)
npc = npa**2 + npb**2 + 2 * npa * npb + 1

a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

# Get a LazyExpr instance
c = a**2 + b**2 + 2 * a * b + 1
# Evaluate!  Output is a NDArray
d = c.evaluate()
# Check
assert isinstance(d, blosc2.NDArray)
assert np.allclose(d[:], npc)

# Evaluate an slice!  Output is a NumPy array
npd = c[:]
# Check
assert isinstance(npd, np.ndarray)
assert np.allclose(npd, npc)

# Evaluate an slice!  Output is a NumPy array
npd = c[1:10]
# Check
assert isinstance(npd, np.ndarray)
assert np.allclose(npd, npc[1:10])
