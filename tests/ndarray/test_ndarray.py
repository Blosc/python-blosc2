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


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 1),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "typesize": 4}, {}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 10),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_ndarray_cframe(contiguous, urlpath, cparams, dparams, nchunks, copy):
    storage = {"contiguous": contiguous, "urlpath": urlpath}
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 1000 * nchunks, dtype="int32").reshape(200, 1000, nchunks)
    ndarray = blosc2.asarray(data, storage=storage, cparams=cparams, dparams=dparams)

    cframe = ndarray.to_cframe()
    ndarray2 = blosc2.ndarray_from_cframe(cframe, copy)

    data2 = ndarray2[:]
    assert np.array_equal(data, data2)

    cframe = ndarray.to_cframe()
    ndarray3 = blosc2.schunk_from_cframe(cframe, copy)
    del ndarray3
    # Check that we can still access the external cframe buffer
    _ = str(cframe)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "shape, steps",
    [
        ((200,), 1),
        ((200,), 3),
        ((200, 10), 1),
        ((200, 10), 2),
        ((200, 10, 10), 2),
        ((200, 10, 10), 40),
        ((200, 10, 10, 10), 9),
    ],
)
def test_getitem_steps(shape, steps):
    data = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    ndarray = blosc2.asarray(data)

    steps_array = ndarray[::steps]
    steps_data = data[::steps]
    np.testing.assert_equal(steps_array[:], steps_data)


@pytest.mark.parametrize("shape", [(0,), (0, 0), (0, 1), (0, 0, 0), (0, 1, 0)])
@pytest.mark.parametrize("urlpath", [None, "test.b2nd"])
def test_shape_with_zeros(shape, urlpath):
    data = np.zeros(shape, dtype="int32")
    ndarray = blosc2.asarray(data, urlpath=urlpath, mode="w")
    if urlpath is not None:
        ndarray = blosc2.open(urlpath)
    assert isinstance(ndarray, blosc2.NDArray)
    assert ndarray.shape == shape
    assert ndarray.size == 0
    np.testing.assert_allclose(data[()], ndarray[()])
    np.testing.assert_allclose(data[:], ndarray[:])
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "a",
    [
        np.linspace(0, 10),
        np.linspace(0, 10)[0],
        np.linspace(0, 10, 1),
        np.array(3.14),
    ],
)
def test_asarray(a):
    b = blosc2.asarray(a)
    if a.shape == ():
        np.testing.assert_allclose(a[()], b[()])
    else:
        np.testing.assert_allclose(a, b[:])
