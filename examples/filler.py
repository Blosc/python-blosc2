#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Fill an SChunk with a filler decorator

import numpy as np

import blosc2

nchunks = 3
chunk_len = 200 * 1000
schunk_dtype = np.dtype(np.float64)

# Set the compression parameters. We need nthreads=1 for this example.
cparams = {"typesize": schunk_dtype.itemsize, "nthreads": 1}

# Create empty SChunk
schunk = blosc2.SChunk(chunksize=chunk_len * schunk_dtype.itemsize, cparams=cparams)

# Create operands (can be a SChunk, numpy.ndarray or Python scalar)
op_dtype = np.dtype(np.int32)
data = np.full(chunk_len * nchunks, 1234, dtype=op_dtype)
schunk_op = blosc2.SChunk(chunksize=chunk_len * op_dtype.itemsize, data=data)
op2_dtype = np.dtype(np.float32)
nparray_op = np.arange(0, chunk_len * nchunks, dtype=op2_dtype)
py_scalar = np.e


# Set filler with decorator
@schunk.filler(((schunk_op, op_dtype), (nparray_op, op2_dtype), (py_scalar, np.float32)), schunk_dtype)
def filler(inputs_tuple, output, offset):
    output[:] = inputs_tuple[0] - inputs_tuple[1] * inputs_tuple[2]


# Check that SChunk has been filled correctly
out = np.empty(chunk_len * nchunks, dtype=schunk_dtype)
schunk.get_slice(0, chunk_len * nchunks, out=out)

res = np.empty(data.shape, dtype=schunk_dtype)
filler((data, nparray_op, py_scalar), res, None)
np.testing.assert_allclose(out, res)
