#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import blosc2
import numpy as np

shape = (128, 128)
chunks = (32, 32)
blocks = (8, 8)

urlpath = "ex_persistency.b2nd"
blosc2.remove_urlpath(urlpath)

dtype = np.dtype(np.complex128)
typesize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a b2nd array from a numpy array (on disk)
a = blosc2.from_buffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks,
                       urlpath=urlpath, contiguous=False)

# Read a b2nd array from disk
b = blosc2.open(urlpath)

# Convert a b2nd array to a numpy array
nparray2 = b[...]
np.testing.assert_almost_equal(nparray, nparray2)

# Remove file on disk
blosc2.remove_urlpath(urlpath)
