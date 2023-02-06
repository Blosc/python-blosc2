#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pytest

import blosc2
import numpy as np


@pytest.mark.parametrize("shape, chunks, blocks, typesize, cparams",
                         [
                             ((100, 1230), (200, 100), (55, 3), 4,
                              {"codec": blosc2.Codec.ZSTD, "clevel": 4,
                               "use_dict": 0, "nthreads": 1}),
                             ((23, 34), (10, 10), (10, 10), 8,
                              {"codec": blosc2.Codec.BLOSCLZ, "clevel": 8,
                               "use_dict": False, "nthreads": 2}),
                             ((80, 51, 60), (20, 10, 33), (6, 6, 26), 3,
                              {"codec": blosc2.Codec.LZ4, "clevel": 5,
                               "use_dict": 1, "nthreads": 2})
                         ])
def test_zeros(shape, chunks, blocks, typesize, cparams):
    a = blosc2.zeros(shape,
                     chunks=chunks,
                     blocks=blocks,
                     typesize=typesize,
                     cparams=cparams)

    for i in np.nditer(np.array(a[:])):
        assert i == bytes(typesize)
