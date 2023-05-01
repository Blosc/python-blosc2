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
nchunks = 10_000
chunkshape = 200_000
dtype = np.dtype(np.int32)

# Set the compression and decompression parameters
dparams = {"nthreads": 1}

# Create array
arr = blosc2.empty(shape=(nchunks * chunkshape,), chunks=(chunkshape,), dtype=dtype, dparams=dparams)
data = np.arange(chunkshape, dtype=dtype)
t0 = time()
for i in range(nchunks):
    arr[i * chunkshape : (i + 1) * chunkshape] = data
t = time() - t0
print(
    f"time append: {t:.2f}s ({arr.schunk.nbytes / (t * 2**30):.3f} GB/s)"
    f" / cratio: {arr.schunk.cratio:.2f}x"
)

t0 = time()
arr_ = arr.copy()
t = time() - t0
print(
    f"time copy (no postfilter): {t:.2f}s ({arr_.schunk.nbytes / (t * 2**30):.3f} GB/s)"
    f" / cratio: {arr_.schunk.cratio:.2f}x"
)


# Associate a postfilter to schunk
@arr.schunk.postfilter(dtype)
def py_postfilter(input, output, offset):
    output[:] = 0


t0 = time()
arr_ = arr.copy()
t = time() - t0
print(
    f"time sum (postfilter): {t:.2f}s ({arr_.schunk.nbytes / (t * 2**30):.3f} GB/s)"
    f" / cratio: {arr_.schunk.cratio:.2f}x"
)
