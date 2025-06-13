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
    ("shape1", "shape2", "dtype", "axis"),
    [
        ([521], [121], "i2", 0),
        ([521, 121], [121, 121], "u4", 0),
        ([521, 121], [521, 121], "i8", 1),
        ([521, 121, 10], [121, 121, 10], "f4", 0),
        ([121, 521, 10], [121, 121, 10], "f8", 1),
        ([121, 121, 101], [121, 121, 10], "i4", 2),
        ([121, 121, 101], [121, 121, 10], "i8", -1),
        # 4-dimensional arrays
        ([21, 121, 101, 10], [2, 121, 101, 10], "f4", 0),
        ([121, 21, 101, 10], [121, 12, 101, 10], "i8", 1),
        ([121, 121, 10, 10], [121, 121, 1, 10], "i8", 2),
        ([121, 121, 101, 2], [121, 121, 101, 10], "i8", -1),
    ],
)
def test_concat2(shape1, shape2, dtype, axis):
    ndarr1 = blosc2.arange(0, int(np.prod(shape1)), 1, dtype=dtype, shape=shape1)
    ndarr2 = blosc2.arange(0, int(np.prod(shape2)), 1, dtype=dtype, shape=shape2)
    cparams = blosc2.CParams(clevel=1)
    result = blosc2.concatenate([ndarr1, ndarr2], axis=axis, cparams=cparams)
    nparray = np.concatenate([ndarr1[:], ndarr2[:]], axis=axis)
    np.testing.assert_almost_equal(result[:], nparray)


@pytest.mark.parametrize(
    ("shape1", "shape2", "shape3", "dtype", "axis"),
    [
        ([521], [121], [21], "i2", 0),
        ([521, 121], [22, 121], [21, 121], "u4", 0),
        ([52, 21], [52, 121], [52, 121], "i8", 1),
        ([521, 121, 10], [121, 121, 10], [21, 121, 10], "f4", 0),
        ([121, 521, 10], [121, 121, 10], [121, 21, 10], "f8", 1),
        ([121, 121, 101], [121, 121, 10], [121, 121, 1], "i4", 2),
        # 4-dimensional arrays
        ([21, 121, 101, 10], [2, 121, 101, 10], [1, 121, 101, 10], "f4", 0),
        ([121, 21, 101, 10], [121, 12, 101, 10], [121, 1, 101, 10], "i8", 1),
        ([121, 121, 10, 10], [121, 121, 1, 10], [121, 121, 3, 10], "i8", 2),
        ([121, 121, 101, 2], [121, 121, 101, 10], [121, 121, 101, 1], "i8", -1),
    ],
)
def test_concat3(shape1, shape2, shape3, dtype, axis):
    ndarr1 = blosc2.arange(0, int(np.prod(shape1)), 1, dtype=dtype, shape=shape1)
    ndarr2 = blosc2.arange(0, int(np.prod(shape2)), 1, dtype=dtype, shape=shape2)
    ndarr3 = blosc2.arange(0, int(np.prod(shape3)), 1, dtype=dtype, shape=shape3)
    cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ)
    result = blosc2.concatenate([ndarr1, ndarr2, ndarr3], axis=axis, cparams=cparams)
    nparray = np.concatenate([ndarr1[:], ndarr2[:], ndarr3[:]], axis=axis)
    np.testing.assert_almost_equal(result[:], nparray)
