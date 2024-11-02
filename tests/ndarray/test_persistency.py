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
    "contiguous",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "urlpath", "dtype"),
    [
        ([634], [156], [33], "test00.b2nd", np.float64),
        ([20, 134, 13], [7, 22, 5], [3, 5, 3], "test01.b2nd", np.int32),
        ([12, 13, 14, 15, 16], [4, 6, 4, 7, 5], [2, 4, 2, 3, 3], "test02.b2nd", np.float32),
    ],
)
def test_persistency(shape, chunks, blocks, urlpath, contiguous, dtype):
    blosc2.remove_urlpath(urlpath)

    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    _ = blosc2.asarray(nparray, chunks=chunks, blocks=blocks, urlpath=urlpath, contiguous=contiguous)
    b = blosc2.open(urlpath)

    bc = b[:]

    nparray2 = np.asarray(bc).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)

    blosc2.remove_urlpath(urlpath)
