#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Shows how to use the bytedelta filter.  Remember that bytedelta is designed
# to work after shuffle.

import math

import numpy as np

import blosc2

shape = (1000, 1000)

# Create a buffer
nparray = np.linspace(0, 1000, math.prod(shape)).reshape(shape)

# Compress with and without bytedelta
cparams = {"filters": [blosc2.Filter.SHUFFLE]}
a = blosc2.asarray(nparray, cparams=cparams)
print(
    f"Compression ratio with shuffle: {a.schunk.cratio:.2f} x",
)

# Now with bytedelta
cparams = {"filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]}
a = blosc2.asarray(nparray, cparams=cparams)
print(
    f"Compression ratio with shuffle + bytedelta: {a.schunk.cratio:.2f} x",
)
