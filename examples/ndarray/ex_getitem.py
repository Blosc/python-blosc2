#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

shape = (10, 10)
slices = (slice(2, 5), slice(4, 8))

dtype = np.int32

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

# Create a NDArray from a numpy array
a = blosc2.asarray(nparray, dtype=dtype)

# Get a slice
buffer = a[slices]
buffer2 = nparray[slices]

np.testing.assert_almost_equal(buffer, buffer2)

a[slices] = np.ones((5, 5), dtype=dtype)

print(a[...])
