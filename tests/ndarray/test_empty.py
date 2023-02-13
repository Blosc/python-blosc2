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


@pytest.mark.parametrize("shape, chunks, blocks, dtype, cparams",
                         [
                             ((100, 1230), (200, 100), (55, 3), np.uint8,
                              {"codec": blosc2.Codec.LZ4, "clevel": 4, "use_dict": 0,
                               "nthreads": 1, "filters": [blosc2.Filter.SHUFFLE]}
                              ),
                             ((234, 125), (90, 90), (20, 10), np.int32,
                              {"codec": blosc2.Codec.LZ4HC, "clevel": 8, "use_dict": False,
                               "nthreads": 2, "filters": [blosc2.Filter.DELTA, blosc2.Filter.BITSHUFFLE]}
                              ),
                             ((400, 399, 401), (20, 10, 130), (6, 6, 26), np.float64,
                              {"codec": blosc2.Codec.BLOSCLZ, "clevel": 5, "use_dict": True,
                               "nthreads": 2, "filters": [blosc2.Filter.DELTA, blosc2.Filter.TRUNC_PREC]}
                              )
                         ])
def test_empty(shape, chunks, blocks, dtype, cparams):
    filters = cparams["filters"]
    cparams["filters_meta"] = [0] * len(filters)
    a = blosc2.empty(shape,
                     chunks=chunks,
                     blocks=blocks,
                     dtype=dtype,
                     cparams=cparams,
                     dparams={"nthreads": 2})

    dtype = np.dtype(dtype)
    assert a.shape == shape
    assert a.chunks == chunks
    assert a.blocks == blocks
    assert a.dtype == dtype
    assert a.schunk.typesize == dtype.itemsize
    assert a.schunk.cparams["codec"] == cparams["codec"]
    assert a.schunk.cparams["clevel"] == cparams["clevel"]
    assert a.schunk.cparams["filters"][:len(filters)] == filters
    assert a.schunk.dparams["nthreads"] == 2
