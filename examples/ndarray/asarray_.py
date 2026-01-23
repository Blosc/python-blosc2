#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
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
