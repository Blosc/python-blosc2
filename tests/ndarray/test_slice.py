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
def test_slice(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = a.slice(slices)
    np_slice = a[slices]
    assert b.shape == np_slice.shape
    np.testing.assert_almost_equal(b[...], np_slice)


argnames = "shape, chunks, blocks, slices, dtype, chunks2, blocks2"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32, [1], [1]),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64, [3, 50], None),
    (
        [12, 13, 14, 15, 16],
        [5, 5, 5, 5, 5],
        [2, 2, 2, 2, 2],
        (slice(1, 3), ..., slice(3, 6)),
        np.float32,
        None,
        [2, 3, 3, 5, 2],
    ),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_slice_chunks_blocks(shape, chunks, blocks, chunks2, blocks2, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = a.slice(slices, chunks=chunks2, blocks=blocks2)
    np_slice = a[slices]
    np.testing.assert_almost_equal(b[...], np_slice)
