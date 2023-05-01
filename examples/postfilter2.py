#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

nchunks = 10
input_dtype = np.dtype("M8[D]")
output_dtype = np.int64  # output dtype has to be of the same size as input

# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.LZ4, "typesize": input_dtype.itemsize}
dparams = {"nthreads": 1}
contiguous = True
urlpath = "filename"
storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
# Remove previous SChunk
blosc2.remove_urlpath(urlpath)
# Create and set data
chunkshape = 200 * 1000
data = np.arange(0, chunkshape * nchunks, dtype=input_dtype)
schunk = blosc2.SChunk(chunksize=chunkshape * input_dtype.itemsize, data=data, **storage)

out1 = np.empty(chunkshape * nchunks, dtype=input_dtype)
schunk.get_slice(0, chunkshape * nchunks, out=out1)


# Set postfilter with decorator
@schunk.postfilter(input_dtype, output_dtype)
def postfilter(input, output, offset):
    output[:] = input <= np.datetime64("1997-12-31")


out2 = np.empty(chunkshape * nchunks, dtype=output_dtype)
schunk.get_slice(0, chunkshape * nchunks, out=out2)

res = np.empty(out1.shape, dtype=output_dtype)
postfilter(data, res, None)
# Check postfilter is applied
assert np.array_equal(res, out2)
