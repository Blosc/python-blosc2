#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import blosc2
import numpy as np

np.random.seed(123)

shape = (50, 50)
chunks = (49, 49)

dtype = np.dtype("|S8")
typesize = dtype.itemsize

# Create a buffer
buffer = bytes(np.random.normal(0, 1, np.prod(shape)) * typesize)

# Create a NDArray from a buffer
a = blosc2.from_buffer(buffer, shape, chunks=chunks, dtype=dtype)
print(a.schunk.cparams["filters"])
print(a.schunk.cparams["codec"])
print(a.schunk.cratio)

# Convert a b2nd array to a buffer
buffer2 = a.to_buffer()
assert buffer == buffer2
