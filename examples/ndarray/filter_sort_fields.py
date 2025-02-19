#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Filter and sort fields in a structured array
# Note that this only works for 1D arrays

import sys
from time import time

import numpy as np

import blosc2

N = 1_000_000

# arr = blosc2.open("/Users/faltet/Downloads/ds-1d-fields.b2nd")
# Create a numpy structured array with 3 fields and N elements
dt = np.dtype([("a", "i4"), ("b", "f4"), ("c", "f8")])
nsa = np.empty((N,), dtype=dt)
# TODO: Make this work with a 2D array
# nsa = np.empty((N,N), dtype=dt)
nsa["a"][:] = np.arange(N, dtype="i4")
nsa["b"][:] = np.linspace(0, 1, N, dtype="f4")
rng = np.random.default_rng(42)  # to get reproducible results
nsa["c"][:] = rng.random(N)

arr = blosc2.asarray(nsa)

t0 = time()
# Using plain sort in combination with filter
# farr = arr["b >= c"].sort("c").compute()
# You can use indices() to get the indices sorted
farr = arr["b >= c"].indices(order="c").compute()
# You can also use __getitem__ to get numpy arrays as result
# farr = arr["b >= c"].sort("c")[:]
print(f"Time to filter: {time() - t0:.3f} s")
print(f"farr: {farr[:10]}")
if farr.dtype == np.dtype("int64"):
    print(f"sorted (blosc2):\n {arr[farr[:10]]}")

print(f"len(farr): {len(farr)}, len(arr): {len(arr)}")
print(f"type of farr: {farr.dtype}, type of arr: {arr.dtype}")

if isinstance(farr, np.ndarray):
    print(f"nbytes of farr: {farr.nbytes / 2**20:.2f}MB")
    # We cannot proceed anymore
    sys.exit(1)

print(f"cratio of farr: {farr.schunk.cratio:.2f}, cratio of arr: {arr.schunk.cratio:.2f}")
print(
    f"nbytes of farr: {farr.schunk.nbytes / 2**20:.2f}MB, nbytes of arr: {arr.schunk.nbytes / 2**20:.2f}MB"
)
print(
    f"cbytes of farr: {farr.schunk.cbytes / 2**20:.2f}MB, cbytes of arr: {arr.schunk.cbytes / 2**20:.2f}MB"
)
print(f"cparams of farr: {farr.cparams}, cparams of arr: {arr.cparams}")
print(f"chunks of farr: {farr.chunks}, chunks of arr: {arr.chunks}")
