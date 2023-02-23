#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import numpy as np

import blosc2

shape = (128, 128)

urlpath = "ex_persistency.b2nd"
blosc2.remove_urlpath(urlpath)

dtype = np.dtype(np.complex128)
typesize = dtype.itemsize

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a NDArray from a numpy array (on disk)
a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, urlpath=urlpath, contiguous=False)

# Read the array from disk
b = blosc2.open(urlpath)

# Convert NDArray to a numpy array
nparray2 = b[...]
np.testing.assert_almost_equal(nparray, nparray2)

# Remove file on disk
blosc2.remove_urlpath(urlpath)
