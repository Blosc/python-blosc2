#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Creating/dumping an NDArray from/to a buffer

import numpy as np

import blosc2

shape = (50, 50)
chunks = (49, 49)
dtype = np.dtype("|S8")
typesize = dtype.itemsize

# Create a NDArray from a buffer
buffer = bytes(np.random.normal(0, 1, np.prod(shape)) * typesize)
a = blosc2.frombuffer(buffer, shape, chunks=chunks, dtype=dtype)
print("compression ratio:", a.schunk.cratio)

# Convert a NDArray to a buffer
buffer2 = a.tobytes()
assert buffer == buffer2
