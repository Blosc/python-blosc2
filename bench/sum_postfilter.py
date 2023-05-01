#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from time import time

import numpy as np

import blosc2

# Size and dtype of super-chunks
nchunks = 20_000
chunkshape = 50_000
dtype = np.dtype(np.int32)
chunksize = chunkshape * dtype.itemsize

# Set the compression and decompression parameters
cparams = {"typesize": 4, "nthreads": 1}
dparams = {"nthreads": 1}
storage = {"cparams": cparams, "dparams": dparams}

# Create super-chunks
schunk0 = blosc2.SChunk(chunksize=chunksize, **storage)
schunk = blosc2.SChunk(chunksize=chunksize, **storage)

data = np.arange(chunkshape, dtype=dtype)
t0 = time()
for i in range(nchunks):
    schunk.append_data(data)
    schunk0.append_data(data)
print(f"time append: {time() - t0:.2f}s")
print(f"cratio: {schunk.cratio:.2f}x")


# Associate a postfilter to schunk
@schunk.postfilter(np.dtype(dtype))
def py_postfilter(input, output, offset):
    output[:] = input + 1


t0 = time()
sum = 0
for chunk in schunk0.iterchunks(dtype):
    chunk += 1
    sum += chunk.sum()
print(f"time sum (no postfilter): {time() - t0:.2f}s")
print(sum)

t0 = time()
sum = 0
for chunk in schunk.iterchunks(dtype):
    sum += chunk.sum()
print(f"time sum (postfilter): {time() - t0:.2f}s")
print(sum)
