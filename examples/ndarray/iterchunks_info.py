#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Using the iterchunks_info for efficient iteration over chunks

from time import time

import blosc2

shape = (1000,) * 3
chunks = (500,) * 3
dtype = "f4"

# Create the NDArray with a mix of different special values (and not special too!)
# a = blosc2.zeros(shape, chunks=chunks, dtype=dtype)
a = blosc2.full(shape, fill_value=9, chunks=chunks, dtype=dtype)
slice_ = (slice(0, 500), slice(0, 500), slice(0, 500))
a[slice_] = 0  # introduce a zeroed chunk (another type of special value)
slice_ = (slice(-500, -1), slice(-500, -1), slice(-500, -1))
a[slice_] = 1  # blosc2 is currently not able to determine special values in this case

# Iterate over chunks
t0 = time()
for info in a.iterchunks_info():
    print(info)
    # Do something fancy with the chunk
print(f"Time for iterating over {a.schunk.nchunks} chunks: {time() - t0:.4f} s")
