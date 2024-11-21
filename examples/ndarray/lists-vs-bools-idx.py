#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This example shows how to use the `indices()` method to get the indices an expression
# and compare this with the bools version of the index.

"""
The output of this script is:
```
Time to create blosc2 array (UDF): 1.337 s
storage required by arr: 673.80 MB (2.26x)
Time to get values: 1.144 s
vals: [(205058, 2.0505828e-05, 1.75294661e-05)
 (283791, 2.8379178e-05, 2.55440616e-05)
 (351524, 3.5152421e-05, 4.65315200e-06)], len: 499774
Time to get list indices: 0.963 s
storage required by indices: 0.81 MB (7.36x)
Time to get values using list: 0.352 s
Time to get bool indices: 0.366 s
storage required by bools idx: 2.23 MB (42.68x)
Time to get values using bools: 0.351 s
```
"""

from time import time

import numpy as np

import blosc2

N = 100_000_000
reduc = 0.01

dt = np.dtype([("a", "i4"), ("b", "f4"), ("c", "f8")])

# # Create a numpy structured array with 3 fields and N elements
# t0 = time()
# nsa = np.empty((N,), dtype=dt)
# nsa["a"][:] = np.arange(N, dtype="i4")
# nsa["b"][:] = np.linspace(0, 1, N, dtype="f4")
# rng = np.random.default_rng(42)  # to get reproducible results
# nsa["c"][:] = rng.random(N)
# print(f"Time to create numpy array: {time() - t0:.3f} s")
#
# # Get the blosc2 array
# t0 = time()
# arr = blosc2.asarray(nsa)
# print(f"Time to create blosc2 array: {time() - t0:.3f} s")


# Create a blosc2 array with a UDF (User Defined Function)
# This emulates the creation of a blosc2 array above
def fill_chunk(inputs_tuple, output, offset):
    lout = len(output)
    off = offset[0]
    output["a"][:] = np.arange(off, off + lout, dtype="i4")
    start = off / N * reduc
    stop = (off + lout) / N * reduc
    output["b"][:] = np.linspace(start, stop, lout, dtype="f4")
    rng = inputs_tuple[0]
    output["c"][:] = rng.random(len(output))


t0 = time()
rng = np.random.default_rng(42)  # to get reproducible results
lazyarray = blosc2.lazyudf(fill_chunk, (rng,), dtype=dt, shape=(N,))
# print(lazyarray.info)
arr = lazyarray.compute()
print(f"Time to create blosc2 array (UDF): {time() - t0:.3f} s")
print(f"storage required by arr: {arr.schunk.cbytes / 2**20:.2f} MB ({arr.schunk.cratio:.2f}x)")
# print(f"arr: {arr[:3]}, len: {len(arr)}")
# print(arr.info)

# Get the values for the expression "b >= c"
t0 = time()
vals = arr["b >= c"].compute()
print(f"Time to get values: {time() - t0:.3f} s")
print(f"vals: {vals[:3]}, len: {len(vals)}")

# Get the list of indices for the expression "b >= c"
t0 = time()
indices = arr["b >= c"].indices().compute()
print(f"Time to get list indices: {time() - t0:.3f} s")
print(f"storage required by indices: {indices.schunk.cbytes / 2**20:.2f} MB ({indices.schunk.cratio:.2f}x)")
# print(f"indices: {indices[:10]}, len: {len(indices)}")

# Get the values for the expression "b >= c" using the list version
t0 = time()
vals = arr[indices]
print(f"Time to get values using list: {time() - t0:.3f} s")
# print(f"vals: {vals[:10]}, len: {len(vals)}")

# Now, get the array of bools for indexing the expression "b >= c"
t0 = time()
bools = (arr["b"] >= arr["c"]).compute()
print(f"Time to get bool indices: {time() - t0:.3f} s")
cratio = bools.schunk.cratio
print(f"storage required by bools idx: {bools.schunk.cbytes / 2**20:.2f} MB ({bools.schunk.cratio:.2f}x)")
# print(f"bools: {bools[:10]}, len: {len(bools)}")

# Get the values for the expression "b >= c" using the bools version
t0 = time()
vals = arr[bools]
print(f"Time to get values using bools: {time() - t0:.3f} s")
# print(f"vals: {vals[:10]}, len: {len(vals)}")
