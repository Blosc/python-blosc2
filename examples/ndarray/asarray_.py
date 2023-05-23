#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Import structured arrays using the array interface

import numpy as np

import blosc2

shape = (2, 2)
dtype = np.float64

# Create a structured array
arr0 = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
arr1 = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
arr = np.array([arr0, arr1], dtype="f8,f8")
print("NumPy struct array:\n", arr)

# And convert it into a NDArray using the array interface
a = blosc2.asarray(arr)
print("\nNDArray struct array:\n", a[...])
