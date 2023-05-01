#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to implement an user defined filter in pure Python

import numpy as np

import blosc2

nchunks = 2
chunk_len = 20 * 1000
dtype = np.dtype(np.int32)


# Define forward and backward functions
def forward(input, output, meta, schunk):
    nd_input = input.view(dtype)
    nd_output = output.view(dtype)

    nd_output[:] = nd_input + 1


def backward(input, output, meta, schunk):
    nd_input = input.view(dtype)
    nd_output = output.view(dtype)

    nd_output[:] = nd_input - 1


# Register filter
id = 160
blosc2.register_filter(id, forward, backward)

# Set the compression and decompression parameters
cparams = {
    "typesize": dtype.itemsize,
    "nthreads": 1,
    "filters": [blosc2.Filter.NOFILTER, id],
    "filters_meta": [0, 0],
}
dparams = {"nthreads": 1}

# Create SChunk and fill it with data
data = np.arange(0, chunk_len * nchunks, 1, dtype=dtype)
schunk = blosc2.SChunk(chunksize=chunk_len * dtype.itemsize, data=data, cparams=cparams, dparams=dparams)

# Check data can be decompressed correctly
out = np.empty(chunk_len * nchunks, dtype=dtype)
schunk.get_slice(0, chunk_len * nchunks, out=out)
assert np.array_equal(data, out)
