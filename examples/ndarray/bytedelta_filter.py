#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
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
cparams = blosc2.CParams(filters=[blosc2.Filter.SHUFFLE], filters_meta=[0])
a = blosc2.asarray(nparray, cparams=cparams)
print(
    f"Compression ratio with shuffle: {a.schunk.cratio:.2f} x",
)

# Now with bytedelta
cparams = blosc2.CParams(filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA], filters_meta=[0, 0])
a = blosc2.asarray(nparray, cparams=cparams)
print(
    f"Compression ratio with shuffle + bytedelta: {a.schunk.cratio:.2f} x",
)
