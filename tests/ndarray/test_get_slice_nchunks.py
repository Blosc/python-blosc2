#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2

argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([456, 200], [258, 100], [73, 25], (slice(0), slice(0)), np.int64),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_getitem(shape, chunks, blocks, slices, dtype):
    a = blosc2.zeros(shape, dtype, chunks=chunks, blocks=blocks)
    schunk = a.schunk
    for i in range(schunk.nchunks):
        chunk = np.full(schunk.chunksize // schunk.typesize, i, dtype=dtype)
        schunk.update_data(i, chunk, True)

    np.array_equal(np.unique(a[slices]), blosc2.get_slice_nchunks(a, slices))
