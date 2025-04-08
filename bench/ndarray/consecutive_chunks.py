#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for comparing speeds of NDArray.slice() when using
# different slices containing consecutive and non-consecutive chunks.

import math
from time import time
import numpy as np
import blosc2

# Dimensions and type properties for the arrays
shape = (50, 100, 300)
chunks = (5, 25, 50)
blocks = (1, 5, 10)
dtype = np.dtype(np.int32)

# Consecutive slices
c_slices = [
    (slice(0, 50), slice(0, 100), slice(0, 300)),
    (slice(0, 10), slice(0, 100), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 50)),
    ]
# Non-consecutive slices
nc_slices = [
    (slice(0, 50), slice(0, 100), slice(0, 300-1)),
    (slice(0, 10), slice(0, 100-1), slice(0, 300)),
    (slice(0, 5-1), slice(0, 25), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 50-1)),
    ]

print("Creating array with shape:", shape)
arr = blosc2.arange(math.prod(shape), dtype=dtype, shape=shape, chunks=chunks, blocks=blocks)

t0 = time()
for s in c_slices:
    arr2 = arr.slice(s)
    # print(arr2.shape, arr[s].shape)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
t1 = time() - t0
print(f"Time to get consecutive slices: {t1:.5f}")

t0 = time()
for s in nc_slices:
    arr2 = arr.slice(s)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
t1 = time() - t0
print(f"Time to get non-consecutive slices: {t1:.5f}")
