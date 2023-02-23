#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

shape = (1234, 23)
chunks = (253, 23)

dtype = bool

# Create a buffer
nparray = np.random.choice(a=[True, False], size=np.prod(shape)).reshape(shape)

# Create a NDArray from a NumPy array
a = blosc2.asarray(nparray, chunks=chunks)
b = a.copy()

# Convert a NDArray to a NumPy array
nparray2 = b[...]

np.testing.assert_almost_equal(nparray, nparray2)
