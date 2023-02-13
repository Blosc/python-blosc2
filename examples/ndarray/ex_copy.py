#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import blosc2
import numpy as np

shape = (10, 10)
chunks = (10, 10)
blocks = (10, 10)

dtype = np.dtype(np.float64)

# Create a buffer
buffer = bytes(np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape))

# Create a b2nd array from a buffer
a = blosc2.from_buffer(buffer, shape, dtype=dtype,
                       chunks=chunks, blocks=blocks)

# Get a copy of a b2nd array
b = blosc2.copy(a)
d = b.copy()

aux = np.asarray(b[...])
aux[1, 2] = 0
aux2 = blosc2.asarray(aux, chunks=chunks, blocks=blocks, dtype=dtype)

print(np.asarray(aux2))

c = np.asarray(b[...])

c[3:5, 2:7] = 0
print(c)

del b

print(c)

# Convert the copy to a buffer
buffer1 = a.to_buffer()
buffer2 = d.to_buffer()

assert buffer1 == buffer2
