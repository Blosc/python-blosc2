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
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_setitem(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks)

    # Python scalar
    nparray = a[...]
    a[slices] = 0
    nparray[slices] = 0
    np.testing.assert_almost_equal(a[...], nparray)

    # Object supporting the Buffer Protocol
    slice_shape = a[slices].shape
    val = np.ones(slice_shape, dtype=dtype)
    a[slices] = val
    nparray[slices] = val
    np.testing.assert_almost_equal(a[...], nparray)

    # blosc2.NDArray
    if np.prod(slice_shape) == 1 or len(slice_shape) != len(blocks):
        chunks = None
        blocks = None

    b = blosc2.full(slice_shape, fill_value=1234567, chunks=chunks, blocks=blocks, dtype=dtype)
    a[slices] = b
    nparray[slices] = b[...]
    np.testing.assert_almost_equal(a[...], nparray)


@pytest.mark.parametrize(
    "shape, slices",
    [
        ([456], slice(0, 1)),
        ([77, 134, 13], (slice(3, 7), slice(50, 100), 7)),
        ([12, 13, 14, 15, 16], (slice(1, 3), ..., slice(3, 6))),
    ],
)
def test_setitem_different_dtype(shape, slices):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=np.int32).reshape(shape)
    a = blosc2.empty(nparray.shape, dtype=np.float64)

    with pytest.raises(ValueError):
        a[slices] = nparray
