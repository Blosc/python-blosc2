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
    ("shape", "new_shape", "chunks", "blocks", "fill_value"),
    [
        ((100, 1230), (200, 1230), (200, 100), (55, 3), b"0123"),
        ((23, 34), (23, 120), (20, 20), (10, 10), 1234),
        ((80, 51, 60), (80, 100, 100), (20, 10, 33), (6, 6, 26), 3.333),
    ],
)
def test_resize(shape, new_shape, chunks, blocks, fill_value):
    a = blosc2.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)

    a.resize(new_shape)
    assert a.shape == new_shape

    slices = tuple(slice(s) for s in shape)
    for i in np.nditer(a[slices]):
        assert i == fill_value


@pytest.mark.parametrize(
    ("shape", "axis", "chunks", "blocks", "fill_value"),
    [
        ((0,), 1, (0,), (0,), 1),
        ((100, 1230), 1, (200, 100), (55, 3), b"0123"),
        ((23, 34), 0, (20, 20), (10, 10), 1234),
        ((80, 51, 60), (-1, -2, 1), (20, 10, 33), (6, 6, 26), 3.333),
    ],
)
def test_expand_dims(shape, axis, chunks, blocks, fill_value):
    a = blosc2.full(shape, fill_value=fill_value, chunks=chunks, blocks=blocks)
    npa = a[:]
    b = blosc2.expand_dims(a, axis=axis)
    npb = np.expand_dims(npa, axis)
    assert npb.shape == b.shape
    np.testing.assert_array_equal(npb, b[:])

    # Repeated expansion
    axis = (axis,) if isinstance(axis, int) else axis
    axis = axis[0] if (len(axis) + b.ndim) > blosc2.MAX_DIM else axis
    b = blosc2.expand_dims(b, axis=axis)
    npb = np.expand_dims(npb, axis)
    assert npb.shape == b.shape
    np.testing.assert_array_equal(npb, b[:])

    # Check that handling of views is correct
    a = blosc2.expand_dims(a, axis=axis)  # could lose ref to original array and thus dealloc data
    npa = np.expand_dims(npa, axis)
    assert a[()].shape == npa[()].shape  # getitem fails if deallocate has happened

    # Now check that garbage collecting works and there will be no memory leaks for views
    import sys

    arr = np.arange(4)
    bloscarr_ = blosc2.asarray(arr)
    assert sys.getrefcount(arr) == sys.getrefcount(bloscarr_) == 2

    view = np.expand_dims(arr, 0)
    bloscview = blosc2.expand_dims(bloscarr_, 0)
    assert sys.getrefcount(arr) == sys.getrefcount(bloscarr_) == 3

    del view
    del bloscview
    assert sys.getrefcount(arr) == sys.getrefcount(bloscarr_) == 2

    # view of a view
    view = np.expand_dims(arr, 0)
    bloscview = blosc2.expand_dims(bloscarr_, 0)
    view2 = np.expand_dims(view, 0)
    bloscview2 = blosc2.expand_dims(bloscview, 0)
    assert sys.getrefcount(arr) == sys.getrefcount(bloscarr_) == 4

    del bloscview
    del bloscarr_
    assert bloscview2[()].shape == bloscview2.shape  # shouldn't fail because still have access to bloscarr_
