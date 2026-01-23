#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Example of prefiltering data before compression

import numpy as np

import blosc2

nchunks = 3
input_dtype = np.dtype(np.int32)
output_dtype = np.dtype(np.float32)

# Set the compression and decompression parameters
cparams = blosc2.CParams(typesize=4, nthreads=1)
dparams = blosc2.DParams(nthreads=4)
storage = blosc2.Storage(mode="a")
# Create empty schunk
schunk = blosc2.SChunk(
    chunksize=200 * 1000 * input_dtype.itemsize, cparams=cparams, dparams=dparams, storage=storage
)


# Set prefilter with decorator
@schunk.prefilter(input_dtype, output_dtype)
def prefilter(input, output, offset):
    output[:] = input - np.pi


# Append data
data = np.arange(200 * 1000 * nchunks, dtype=input_dtype)
schunk[: 200 * 1000 * nchunks] = data

# Check prefilter is applied correctly
out2 = np.empty(200 * 1000 * nchunks, dtype=output_dtype)
schunk.get_slice(0, 200 * 1000 * nchunks, out=out2)

res = np.empty(data.shape, dtype=output_dtype)
prefilter(data, res, None)
assert np.allclose(res, out2)
