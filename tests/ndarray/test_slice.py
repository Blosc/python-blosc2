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
    # Consecutive slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 10), slice(0, 100), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 100), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 50)), np.int32),
    # Aligned slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 50), slice(25, 100), slice(50, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 40), slice(25, 75), slice(100, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 35), slice(50, 75), slice(100, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 25), slice(25, 50), slice(50, 100)), np.int32),
    # Non-consecutive slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 10), slice(0, 100), slice(0, 300 - 1)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 100 - 1), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5 - 1), slice(0, 25), slice(0, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 50 - 1)), np.int32),
    # Non-aligned slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 50 - 1), slice(25, 100), slice(50, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 40), slice(25, 75 - 1), slice(100, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 35), slice(50, 75), slice(100, 300 - 1)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20 + 1, 25), slice(25, 50), slice(50, 100)), np.int32),
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


@pytest.mark.parametrize(argnames, argvalues)
def test_slice_codec_and_clevel(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(
        nparray,
        chunks=chunks,
        blocks=blocks,
        cparams={"codec": blosc2.Codec.LZ4, "clevel": 6, "filters": [blosc2.Filter.BITSHUFFLE]},
    )

    b = a.slice(slices)
    assert b.cparams.codec == a.cparams.codec
    assert b.cparams.clevel == a.cparams.clevel
    assert b.cparams.filters == a.cparams.filters


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
