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

shape = (1_000,) * 3
chunks = (500,) * 3
dtype = "f4"
a = blosc2.full(shape, fill_value=9, chunks=chunks, dtype=dtype)
# a = blosc2.zeros(shape, chunks=chunks, dtype=dtype)
# print(a.info)

# Iterate over chunks
t0 = time()
for info in a.iterchunks_info():
    print(info)
    # Do something with the chunk
    pass
print(f"Time for iterating over {a.schunk.nchunks} chunks: {time() - t0:.4f} s")
