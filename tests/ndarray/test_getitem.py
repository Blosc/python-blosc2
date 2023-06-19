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
def test_getitem(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    np.testing.assert_almost_equal(a[slices], nparray_slice)


@pytest.mark.parametrize(argnames, argvalues)
def test_getitem_numpy(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    a_slice = a[slices]

    np.testing.assert_almost_equal(a_slice, nparray_slice)


@pytest.mark.parametrize(argnames, argvalues)
def test_getitem_simple(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray)
    nparray_slice = nparray[slices]
    a_slice = a[slices]

    np.testing.assert_almost_equal(a_slice, nparray_slice)


def test_getitem_shapes():
    shape = (5, 5)
    slice_ = (slice(4, 6), slice(4, 6))

    npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    b2a = blosc2.asarray(npa)

    # One elem slice
    assert b2a[4, 4].shape == npa[4, 4].shape
    assert b2a[4:, 4].shape == npa[4:, 4].shape
    assert b2a[4, 4:].shape == npa[4, 4:].shape
    assert b2a[4:, 4:].shape == npa[4:, 4:].shape
    assert b2a[slice_].shape == npa[slice_].shape

    # More than one elem slice
    assert b2a[3:, 4].shape == npa[3:, 4].shape
    assert b2a[3, 4:].shape == npa[3, 4:].shape
    assert b2a[3:, 4:].shape == npa[3:, 4:].shape
