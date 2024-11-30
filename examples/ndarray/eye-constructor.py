#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This example shows how to use the `eye()` constructor to create a blosc2 array.

import math
from time import time

import numpy as np

import blosc2

N = 20_000

shape = (N, N)
print(f"*** Creating a blosc2 eye array with shape: {shape} ***")
t0 = time()
a = blosc2.eye(*shape, dtype=np.int8)
cratio = a.schunk.nbytes / a.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({math.prod(shape) / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {a.schunk.cbytes / 1e6:.2f} MB (cratio: {cratio:.2f}x)"
)
print(f"Last 3 elements:\n{a[-3:]}")

# You can create rectangular arrays too
shape = (N, N * 5)
print(f"*** Creating a blosc2 eye array with shape: {shape} ***")
t0 = time()
a = blosc2.eye(*shape, dtype=np.int8)
cratio = a.schunk.nbytes / a.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({math.prod(shape) / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {a.schunk.cbytes / 1e6:.2f} MB (cratio: {cratio:.2f}x)"
)
print(f"First 3 elements:\n{a[:3]}")


# In conclusion, you can use blosc2 eye() to create blosc2 arrays requiring much less storage
# than numpy arrays.
