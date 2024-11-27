#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import math

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
def test_basic(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.frombuffer(bytes(nparray), nparray.shape, dtype=dtype, chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    np.testing.assert_almost_equal(a[slices], nparray_slice)


@pytest.mark.parametrize(argnames, argvalues)
def test_numpy(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    nparray_slice = nparray[slices]
    a_slice = a[slices]

    np.testing.assert_almost_equal(a_slice, nparray_slice)


@pytest.mark.parametrize(argnames, argvalues)
def test_simple(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray)
    nparray_slice = nparray[slices]
    a_slice = a[slices]

    np.testing.assert_almost_equal(a_slice, nparray_slice)


def test_shapes():
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

    # Negative values for start
    assert b2a[-1, -1].shape == npa[-1, -1].shape
    assert b2a[-1:, -2].shape == npa[-1:, -2].shape
    assert b2a[-2, -3:].shape == npa[-2, -3:].shape
    # Negative values for stop
    assert b2a[1:-1, 1].shape == npa[1:-1, 1].shape
    assert b2a[1, :-2].shape == npa[1, :-2].shape
    assert b2a[1:-2, 2:-3].shape == npa[1:-2, 2:-3].shape


def int_array(shape):
    rng = np.random.Generator(np.random.PCG64(12345))
    return rng.integers(0, shape[0], size=shape)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "idx"),
    [
        ((5,), (2,), (1,), int_array((2,))),
        ((15,), (4,), (2,), int_array((3,))),
        ((501,), (22,), (11,), int_array((221,))),
    ],
)
def test_1d_values(shape, chunks, blocks, idx):
    npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    b2a = blosc2.asarray(npa)

    np.testing.assert_equal(b2a[idx], npa[idx])
    assert b2a[idx].dtype == npa[idx].dtype
    np.testing.assert_equal(b2a[list(idx)], npa[list(idx)])
    assert b2a[list(idx)].dtype == npa[list(idx)].dtype


def bool_array(shape):
    rng = np.random.Generator(np.random.PCG64(12345))
    return rng.choice([True, False], size=shape)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "idx"),
    [
        ((5,), (2,), (1,), bool_array((5,))),
        ((10, 10), (5, 5), (2, 2), bool_array((10, 10))),
        ((8, 8, 8), (4, 4, 4), (2, 2, 2), bool_array((8, 8, 8))),
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1), bool_array((6, 5, 4, 3))),
    ],
)
def test_bool_values(shape, chunks, blocks, idx):
    npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    b2a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    assert b2a[idx].shape == npa[idx].shape
    assert b2a[idx].dtype == npa[idx].dtype
    assert b2a[idx].size == npa[idx].size
    assert b2a[idx].ndim == npa[idx].ndim


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        ((5,), (2,), (1,)),
        ((10, 10), (5, 5), (2, 2)),
        ((8, 8, 8), (4, 4, 4), (2, 2, 2)),
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1)),
    ],
)
def test_iter(shape, chunks, blocks):
    npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    b2a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    for _i, (a, b) in enumerate(zip(b2a, npa, strict=False)):
        np.testing.assert_equal(a, b)
    assert _i == shape[0] - 1


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_ndarray(dtype):
    # Check that we can slice a blosc2 array with a NDArray
    shape = (10,)
    size = math.prod(shape)
    ndarray = blosc2.arange(size - 1, -1, -1, dtype=np.int64, shape=shape)
    a = blosc2.linspace(0, 10, size, shape=shape, dtype=dtype)
    a_slice = a[ndarray]
    na = np.linspace(0, 10, size, dtype=dtype).reshape(shape)
    nparray = np.arange(size - 1, -1, -1, dtype=np.int64).reshape(shape)
    na_slice = na[nparray]
    np.testing.assert_almost_equal(a_slice, na_slice)
