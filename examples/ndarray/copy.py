#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Copying NDArrays

import numpy as np

import blosc2

shape = (10, 10)
blocks = (10, 10)
dtype = np.float64

# Create a NDArray from a buffer
buffer = bytes(np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape))
a = blosc2.frombuffer(buffer, shape, dtype=dtype, blocks=blocks)

# Get a copy of a
b = blosc2.copy(a)

# Another copy example
b[1:5, 2:9] = 0
b2 = blosc2.copy(b, blocks=blocks)
print(b2[...])
