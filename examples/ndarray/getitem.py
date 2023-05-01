#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Show how getitem / setitem works for an NDArray

import numpy as np

import blosc2

shape = (10, 10)
slices = (slice(2, 7), slice(4, 8))

# Create a NDArray from a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
a = blosc2.asarray(nparray)

# Get a slice
buffer = a[slices]

# Set a slice
a[slices] = np.ones_like(buffer) - buffer
print(a[...])
