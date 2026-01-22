#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
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
cparams = blosc2.CParams(typesize=4, nthreads=1)
dparams = blosc2.DParams(nthreads=1)

# Create super-chunks
schunk0 = blosc2.SChunk(chunksize=chunksize, cparams=cparams, dparams=dparams)
schunk = blosc2.SChunk(chunksize=chunksize, cparams=cparams, dparams=dparams)

data = np.arange(chunkshape, dtype=dtype)
t0 = time()
for _i in range(nchunks):
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
