#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

nchunks = 5
input_dtype = np.dtype(np.int32)
output_dtype = np.dtype(np.float32)

# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.LZ4, "typesize": 4}
dparams = {"nthreads": 1}
contiguous = True
urlpath = None
storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
# Remove previous SChunk
blosc2.remove_urlpath(urlpath)
# Create and set data
data = np.arange(200 * 1000 * nchunks, dtype=input_dtype)
schunk = blosc2.SChunk(chunksize=200 * 1000 * input_dtype.itemsize, data=data, **storage)

out1 = np.empty(200 * 1000 * nchunks, dtype=input_dtype)
schunk.get_slice(0, 200 * 1000 * nchunks, out=out1)


# Set postfilter with decorator
@schunk.postfilter(input_dtype, output_dtype)
def postfilter(input, output, offset):
    output[:] = input - np.pi


out2 = np.empty(200 * 1000 * nchunks, dtype=output_dtype)
schunk.get_slice(0, 200 * 1000 * nchunks, out=out2)

res = np.empty(out1.shape, dtype=output_dtype)
postfilter(data, res, None)
# Check postfilter is applied
assert np.allclose(res, out2)
