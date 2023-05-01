#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Shows how you can persist an array on disk

import numpy as np

import blosc2

shape = (128, 128)
urlpath = "ex_persistency.b2nd"
dtype = np.complex128

# Create a NDArray from a numpy array (and save it on disk)
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
a = blosc2.asarray(nparray, urlpath=urlpath, mode="w")

# Read the array from disk
b = blosc2.open(urlpath)
# And see its contents
print(b[...])
