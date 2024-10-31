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


@pytest.mark.parametrize(
    "shape, chunks, blocks, dtype",
    [
        ([931], [223], [45], np.int32),
        ([134, 121, 78], [12, 13, 18], [4, 4, 9], np.float64),
    ],
)
def test_numpy(shape, chunks, blocks, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    nparray2 = a[...]
    np.testing.assert_almost_equal(nparray, nparray2)
