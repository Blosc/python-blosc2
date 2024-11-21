#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This example shows how to use the `linspace()` constructor to create a blosc2 array.

from time import time

import numpy as np

import blosc2

N = 1_000_000

t0 = time()
a = blosc2.linspace(0, 10, N)
print(f"Time to create blosc2 array: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)")
cratio = a.schunk.nbytes / a.schunk.cbytes
print(f"storage required by arr: {a.schunk.cbytes / 1e6:.2f} MB ({cratio:.2f}x)")

# You can create ndim arrays too
t0 = time()
b = blosc2.linspace(0, 1, N, dtype=np.float32, shape=(5, N // 5))
print(f"Time to create blosc2 array2: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)")
cratio = b.schunk.nbytes / b.schunk.cbytes
print(f"storage required by array: {a.schunk.cbytes / 1e6:.2f} MB ({cratio:.2f}x)")

# For reference, let's compare with numpy
t0 = time()
na = np.linspace(0, 10, N)
print(f"Time to create numpy array: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)")
print(f"storage required by numpy array: {na.nbytes / 1e6:.2f} MB")
np.testing.assert_allclose(a[:], na)

# Create an NDArray from a numpy array
t0 = time()
c = blosc2.asarray(na)
print(
    f"Time to create blosc2 array from numpy array: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
)
cratio = c.schunk.nbytes / c.schunk.cbytes
print(f"storage required by array: {c.schunk.cbytes / 1e6:.2f} MB ({cratio:.2f}x)")
np.testing.assert_allclose(c[:], a[:])

# In conclusion, you can use blosc2 linspace() to create blosc2 arrays requiring much less storage
# than numpy arrays.  If speed is important, and you can afford the extra memory, you can create
# blosc2 arrays much faster straight from numpy arrays as well.
