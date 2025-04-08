#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for comparing speeds of NDArray.slice() when using
# different slices containing consecutive and non-consecutive chunks,
# as well as aligned and unaligned.

import math
from time import time
import numpy as np
import blosc2

# Dimensions and type properties for the arrays
shape = (50, 100, 300)
chunks = (5, 25, 50)
blocks = (1, 5, 10)
dtype = np.dtype(np.int32)

# Non-consecutive slices
nc_slices = [
    (slice(0, 50), slice(0, 100), slice(0, 300-1)),
    (slice(0, 10), slice(0, 100-1), slice(0, 300)),
    (slice(0, 5-1), slice(0, 25), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 50-1)),
    ]
# Consecutive slices
c_slices = [
    (slice(0, 50), slice(0, 100), slice(0, 300)),
    (slice(0, 10), slice(0, 100), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 300)),
    (slice(0, 5), slice(0, 25), slice(0, 50)),
    ]
# Non-aligned slices
na_slices = [
    (slice(10, 50-1), slice(25, 100), slice(50, 300)),
    (slice(10, 40), slice(25, 75-1), slice(100, 200)),
    (slice(20, 35), slice(50, 75), slice(100, 300-1)),
    (slice(20+1, 25), slice(25, 50), slice(50, 100)),
    ]
# Aligned slices
a_slices = [
    (slice(10, 50), slice(25, 100), slice(50, 300)),
    (slice(10, 40), slice(25, 75), slice(100, 200)),
    (slice(20, 35), slice(50, 75), slice(100, 300)),
    (slice(20, 25), slice(25, 50), slice(50, 100)),
    ]

print("Creating array with shape:", shape)
t0 = time()
arr = blosc2.arange(math.prod(shape), dtype=dtype, shape=shape, chunks=chunks, blocks=blocks)
print(f"Time to create array: {time() - t0 : .5f}")

print("Timing non-consecutive slices...")
nc_times = []
t0 = time()
for s in nc_slices:
    t1 = time()
    arr2 = arr.slice(s)
    nc_times.append(time() - t1)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
print(f"Time to get non-consecutive slices: {time() - t0 : .5f}")

print("Timing consecutive slices...")
c_times = []
c_speedup = []
t0 = time()
for i, s in enumerate(c_slices):
    t1 = time()
    arr2 = arr.slice(s)
    c_times.append(time() - t1)
    c_speedup.append(nc_times[i] / c_times[i])
    # print(arr2.shape, arr[s].shape)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
print(f"Time to get consecutive slices: {time() - t0 : .5f}")
print(f"Speedups for consecutive slices: ", [f"{s:.2f}x" for s in c_speedup])

print("Timing non-aligned slices...")
na_times = []
t0 = time()
for i, s in enumerate(na_slices):
    t1 = time()
    arr2 = arr.slice(s)
    na_times.append(time() - t1)
    # print(arr2.shape, arr[s].shape)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
print(f"Time to get non-aligned slices: {time() - t0 : .5f}")

print("Timing aligned slices...")
a_times = []
a_speedup = []
t0 = time()
for i, s in enumerate(a_slices):
    t1 = time()
    arr2 = arr.slice(s)
    a_times.append(time() - t1)
    a_speedup.append(na_times[i] / a_times[i])
    # print(arr2.shape, arr[s].shape)
    # print(arr2.schunk.nbytes, arr[s].nbytes)
    # np.testing.assert_array_equal(arr2[:], arr[s])
print(f"Time to get aligned slices: {time() - t0 : .5f}")
print(f"Speedups for aligned slices: ", [f"{s:.2f}x" for s in a_speedup])
