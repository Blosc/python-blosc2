#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# A simple example using the pack_tensor and unpack_tensor functions

import numpy as np

import blosc2

a = np.arange(1_000_000)

cparams = blosc2.CParams(
    codec=blosc2.Codec.ZSTD, clevel=9, filters=[blosc2.Filter.BITSHUFFLE], filters_meta=[0]
)
cframe = blosc2.pack_tensor(a, cparams=cparams)
print("Length of packed array in bytes:", len(cframe))

a2 = blosc2.unpack_tensor(cframe)
assert np.all(a == a2)
