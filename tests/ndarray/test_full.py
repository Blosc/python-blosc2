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


@pytest.mark.parametrize("shape, chunks, blocks, fill_value, cparams, dparams",
                         [
                             ((100, 1230), (200, 100), (55, 3), b"0123",
                              {"clevel": 4, "use_dict": 0, "nthreads": 1}, {"nthreads": 1}),
                             ((23, 34), (20, 20), (10, 10), b"sun",
                              {"codec": blosc2.Codec.LZ4HC, "clevel": 8, "use_dict": False, "nthreads": 2},
                              {"nthreads": 2}),
                             ((80, 51, 60), (20, 10, 33), (6, 6, 26), b"qwerty",
                              {"codec": blosc2.Codec.ZLIB, "clevel": 5, "use_dict": True, "nthreads": 2},
                              {"nthreads": 1})
                         ])
def test_full(shape, chunks, blocks, fill_value, cparams, dparams):
    a = blosc2.full(shape, chunks, blocks, fill_value, cparams=cparams, dparams=dparams)
    assert a.schunk.dparams == dparams

    for i in np.nditer(np.array(a[:])):
        assert i == fill_value
