#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Filter and sort fields in a structured array

from time import time

import numpy as np

import blosc2

N = 1_000

# arr = blosc2.open("/Users/faltet/Downloads/ds-1d-fields.b2nd")
# Create a numpy structured array with 3 fields and N elements
dt = np.dtype([("a", "i4"), ("b", "f4"), ("c", "f8")])
nsa = np.empty((N,), dtype=dt)
# Make this work with a 2D array
# nsa = np.empty((N,N), dtype=dt)
nsa["a"][:] = np.arange(N, dtype="i4")
nsa["b"][:] = np.linspace(0, 1, N, dtype="f4")
rng = np.random.default_rng(42)  # to get reproducible results
nsa["c"][:] = rng.random(N)

arr = blosc2.asarray(nsa)

t0 = time()
farr = (
    arr["b >= c"]
    .indices()
    .sort("c")
    .compute(
        cparams={
            "codec": blosc2.Codec.LZ4,
            "clevel": 1,
            # 'use_dict': False,
            # 'filters': [blosc2.Filter.SHUFFLE],
            # 'splitmode': blosc2.SplitMode.ALWAYS_SPLIT,
        }
    )
)
print(f"Time to filter: {time() - t0:.3f} s")
print(f"farr: {farr[:10]}")
print(f"sorted: {arr[:][farr[:10]]}")
# print(f"sorted: {arr[farr[:10]]}")  # TODO

# print(f"len(farr): {len(farr)}, len(arr): {len(arr)}")
print(f"shape of farr: {farr.shape}, shape of arr: {arr.shape}")
print(f"type of farr: {farr.dtype}, type of arr: {arr.dtype}")
print(f"cratio of farr: {farr.schunk.cratio:.2f}, cratio of arr: {arr.schunk.cratio:.2f}")
print(f"nbytes of farr: {farr.schunk.nbytes}, nbytes of arr: {arr.schunk.nbytes}")
print(f"cbytes of farr: {farr.schunk.cbytes}, cbytes of arr: {arr.schunk.cbytes}")
print(f"cparams of farr: {farr.cparams}, cparams of arr: {arr.cparams}")
print(f"chunks of farr: {farr.chunks}, chunks of arr: {arr.chunks}")
