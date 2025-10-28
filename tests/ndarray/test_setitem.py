#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest
import torch

import blosc2

argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 9, 2), ..., slice(3, 6)), np.float32),
    ([12, 13], [5, 5], [2, 2], (slice(11, 2, -1), slice(6, 2, -1)), np.float32),
    ([25, 13, 22], [5, 5, 3], [2, 2, 1], (slice(17, 2, -3), 2, slice(6, 2, -1)), np.float32),
    ([25, 13, 22], [5, 5, 3], [2, 2, 1], (np.s_[-5:-15:-1], np.s_[-3:-11:-2], slice(6, 2, -1)), np.float32),
    ([0, 13, 22], [0, 5, 3], [0, 2, 1], (np.s_[:], np.s_[-5:-15:-1], slice(6, 2, -1)), np.float32),
    ([13, 22], [5, 3], [2, 1], (1, np.s_[-5::-1]), np.float32),
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

    # Object called via SimpleProxy
    slice_shape = a[slices].shape
    dtype_ = {np.float32: torch.float32, np.int32: torch.int32, np.float64: torch.float64}[dtype]
    val = torch.ones(slice_shape, dtype=dtype_)
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
    ("shape", "slices"),
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

    a[slices] = nparray[slices]
    nparray_ = nparray.astype(a.dtype)
    np.testing.assert_almost_equal(a[slices], nparray_[slices])


def test_ndfield():
    # Create a structured NumPy array
    shape = (50, 50)
    na = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
    nb = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
    nsa = np.empty(shape, dtype=[("a", na.dtype), ("b", nb.dtype)])
    nsa["a"] = na
    nsa["b"] = nb
    sa = blosc2.asarray(nsa)

    # Check values
    assert np.allclose(sa["a"][:], na)
    assert np.allclose(sa["b"][:], nb)

    # Change values
    nsa["a"][:] = nsa["b"]
    sa["a"][:] = sa["b"]

    # Check values
    assert np.allclose(sa["a"][:], nsa["a"])
    assert np.allclose(sa["b"][:], nsa["b"])

    # Using NDField accessor
    nsa["b"][:] = 1
    fb = blosc2.NDField(sa, "b")
    fb[:] = blosc2.full(shape, fill_value=1, dtype=np.float64)
    assert np.allclose(sa["a"][:], nsa["a"])
    assert np.allclose(sa["b"][:], nsa["b"])
