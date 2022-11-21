########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import blosc2
import numpy as np

nchunks = 3
input_dtype = np.dtype(np.int32)
output_dtype = np.dtype(np.float32)

# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.LZ4, "typesize": 4, "nthreads": 1}
dparams = {"nthreads": 4}
contiguous = True
urlpath = None
storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

# Create empty schunk
schunk = blosc2.SChunk(chunksize=200 * 1000 * input_dtype.itemsize, **storage)


# Set prefilter with decorator
@schunk.prefilter(input_dtype, output_dtype)
def prefilter(input, output, offset):
    output[:] = input - np.pi


# Append data
data = np.arange(200 * 1000 * nchunks, dtype=input_dtype)
schunk[:200 * 1000 * nchunks] = data

# Check prefilter is applied correctly
out2 = np.empty(200 * 1000 * nchunks, dtype=output_dtype)
schunk.get_slice(0, 200 * 1000 * nchunks, out=out2)

res = np.empty(data.shape, dtype=output_dtype)
prefilter(data, res, None)
assert np.allclose(res, out2)
