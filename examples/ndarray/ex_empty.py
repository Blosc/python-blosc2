#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import blosc2
import numpy as np

np.random.seed(123)


shape, chunks, blocks, typesize, codec, clevel, use_dict, nthreads, filters = (
    (400, 399, 401),
    (20, 10, 130),
    (6, 6, 26),
    3,
    blosc2.Codec.BLOSCLZ,
    5,
    False,
    2,
    [blosc2.Filter.DELTA, blosc2.Filter.TRUNC_PREC]
)

cparams = {"codec": codec, "clevel": clevel, "use_dict": use_dict,
           "nthreads": nthreads, "filters": filters, "filters_meta": [0] * len(filters)}
a = blosc2.empty(shape, chunks=chunks, blocks=blocks, typesize=typesize,
                 cparams=cparams, dparams={"nthreads": nthreads})

print("HOLA")
