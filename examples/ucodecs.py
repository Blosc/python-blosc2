#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to implement an user defined codec in pure Python

import sys

import numpy as np

import blosc2

nchunks = 2
chunk_len = 20 * 1000
dtype = np.dtype(np.int32)


# Define encoder and decoder functions
def encoder1(input, output, meta, schunk):
    # Check whether the data is an arange
    nd_input = input.view(dtype)
    step = int(nd_input[1] - nd_input[0])
    res = nd_input[1:] - nd_input[:-1]
    if np.min(res) == np.max(res):
        output[0:4] = input[0:4]  # start
        n = step.to_bytes(4, sys.byteorder)
        output[4:8] = [n[i] for i in range(4)]
        return 8
    else:
        # Not compressible, tell Blosc2 to do a memcpy
        return 0


def decoder1(input, output, meta, schunk):
    # For decoding we only have to worry about the arange case
    # (other cases are handled by Blosc2)
    nd_input = input.view(dtype)
    nd_output = output.view(dtype)
    nd_output[:] = [nd_input[0] + i * nd_input[1] for i in range(nd_output.size)]

    return nd_output.size * schunk.typesize


# Register codec
codec_name = "codec"
id = 180
blosc2.register_codec(codec_name, id, encoder1, decoder1)

# Set the compression and decompression parameters
cparams = {
    "typesize": dtype.itemsize,
    "nthreads": 1,
    "filters": [blosc2.Filter.NOFILTER],
    "filters_meta": [0],
}
dparams = {"nthreads": 1}
cparams["codec"] = id

# Create SChunk and fill it with data
data = np.arange(0, chunk_len * nchunks, 1, dtype=dtype)
schunk = blosc2.SChunk(chunksize=chunk_len * dtype.itemsize, data=data, cparams=cparams, dparams=dparams)

# Check data can be decompressed correctly
out = np.empty(chunk_len * nchunks, dtype=dtype)
schunk.get_slice(0, chunk_len * nchunks, out=out)
assert np.array_equal(data, out)
