#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


# A simple example using the pack_tensor and unpack_tensor functions

import numpy as np

import blosc2

a = np.arange(1_000_000)

cparams = {
    "codec": blosc2.Codec.ZSTD,
    "clevel": 9,
    "filters": [blosc2.Filter.BITSHUFFLE],
    "filters_meta": [0],
}
cframe = blosc2.pack_tensor(a, cparams=cparams)
print("Length of packed array in bytes:", len(cframe))

a2 = blosc2.unpack_tensor(cframe)
assert np.all(a == a2)
