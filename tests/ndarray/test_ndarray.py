#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import itertools
import math

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        (blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams(), 1),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "typesize": 4}, blosc2.DParams(), 5),
        (blosc2.CParams(codec=blosc2.Codec.LZ4HC, typesize=4), {}, 10),
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
    ("shape", "steps"),
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


@pytest.mark.parametrize(
    ("shape", "newshape", "chunks", "blocks"),
    [
        ((10,), (2, 5), (1, 5), (1, 2)),
        ((20,), (2, 5, 2), (1, 5, 2), (1, 2, 1)),
        ((60,), (3, 5, 4), (4, 5, 2), (3, 1, 2)),
        ((160,), (8, 5, 4), (4, 5, 2), (3, 2, 1)),
        ((140,), (7, 5, 4), (4, 5, 2), (3, 1, 2)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_reshape(shape, newshape, chunks, blocks, c_order):
    a = np.arange(np.prod(shape))
    b = blosc2.asarray(a)
    c = b.reshape(newshape, chunks=chunks, blocks=blocks, c_order=c_order)
    assert c.shape == newshape
    assert c.dtype == a.dtype
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


@pytest.mark.parametrize(
    ("sss", "shape", "dtype", "chunks", "blocks"),
    [
        ((0, 10, 1), (10,), np.int32, (5,), (2,)),
        ((1, 11, 1), (2, 5), np.int64, (2, 3), (1, 1)),
        ((2, 22, 1), (2, 5, 2), np.float32, (2, 5, 1), (1, 5, 1)),
        ((2, 22, 2), (1, 5, 2), np.float32, (1, 5, 1), (1, 5, 1)),
        ((3, 33, 3), (1, 5, 2), np.float64, (1, 5, 1), (1, 5, 1)),
        ((50, None, None), (10, 5, 1), np.float64, (5, 5, 1), (3, 5, 1)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_arange(sss, shape, dtype, chunks, blocks, c_order):
    start, stop, step = sss
    a = blosc2.arange(
        start, stop, step, dtype=dtype, shape=shape, c_order=c_order, chunks=chunks, blocks=blocks
    )
    assert a.shape == shape
    assert isinstance(a, blosc2.NDArray)
    b = np.arange(start, stop, step, dtype=dtype).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


@pytest.mark.parametrize(
    ("ss", "shape", "dtype", "chunks", "blocks"),
    [
        ((0, 7), (10,), np.float32, (10,), (2,)),
        ((0, 7), (10,), np.float64, (5,), (2,)),
        ((0, 7), (10,), np.complex64, (5,), (2,)),
        ((0, 6), (10,), np.complex128, (5,), (2,)),
        ((-1, 7), (10, 10), np.float32, (10, 2), (2, 2)),
    ],
)
@pytest.mark.parametrize("endpoint", [True, False])
@pytest.mark.parametrize("c_order", [True, False])
def test_linspace(ss, shape, dtype, chunks, blocks, endpoint, c_order):
    start, stop = ss
    num = math.prod(shape)
    a = blosc2.linspace(
        start,
        stop,
        num,
        dtype=dtype,
        shape=shape,
        endpoint=endpoint,
        c_order=c_order,
        chunks=chunks,
        blocks=blocks,
    )
    assert a.shape == shape
    assert a.dtype == dtype
    assert isinstance(a, blosc2.NDArray)
    b = np.linspace(start, stop, num, dtype=dtype, endpoint=endpoint).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass
    with pytest.raises(ValueError):
        a = blosc2.linspace(start, stop, 10, shape=(20,))  # num incompatible with shape
    with pytest.raises(ValueError):
        a = blosc2.linspace(start, stop)  # num or shape should be specified
    a = blosc2.linspace(start, stop, shape=(20,))  # should have length 20
    assert a.shape == (20,)
    a = blosc2.linspace(start, stop, num=20)  # should have length 20
    assert a.shape == (20,)


@pytest.mark.parametrize(("N", "M"), [(10, None), (10, 20), (20, 10)])
@pytest.mark.parametrize("k", [-1, 0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("chunks", [(5, 6), (10, 9)])
def test_eye(k, N, M, dtype, chunks):
    a = np.eye(N, M, k, dtype=dtype)
    b = blosc2.eye(N, M, k, dtype=dtype, chunks=chunks)
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    np.testing.assert_allclose(a, b[:])


@pytest.mark.parametrize(
    ("it", "shape", "dtype", "chunks", "blocks"),
    [
        (range(10), (10,), np.int8, (10,), (2,)),
        (range(1, 11), (10,), np.float64, (5,), (2,)),
        (range(2, 22, 2), (10,), np.int64, (5,), (2,)),
        (range(3, 33, 3), (10,), np.complex128, (5,), (2,)),
        (range(100), (10, 10), np.int32, (10, 2), (2, 2)),
        (range(100), (5, 20), np.int32, (3, 2), (2, 2)),
        (range(24), (2, 3, 4), np.int8, (2, 2, 2), (1, 1, 2)),
        (range(48), (2, 3, 4, 2), np.uint8, (2, 2, 4, 2), (1, 2, 2, 1)),
    ],
)
@pytest.mark.parametrize("c_order", [True, False])
def test_fromiter(it, shape, dtype, chunks, blocks, c_order):
    # Create a duplicate of the iterator
    it, it2 = itertools.tee(it)
    a = blosc2.fromiter(it, dtype=dtype, shape=shape, chunks=chunks, blocks=blocks, c_order=c_order)
    assert a.shape == shape
    assert a.dtype == dtype
    assert isinstance(a, blosc2.NDArray)
    b = np.fromiter(it2, dtype=dtype).reshape(shape)
    if a.ndim == 1 or c_order:
        np.testing.assert_allclose(a[:], b)
    else:
        # This is chunk order, so testing is more laborious, and not really necessary
        pass


@pytest.mark.parametrize("order", ["f0", "f1", "f2", None])
def test_sort(order):
    it = ((x + 1, x - 2, -x) for x in range(10))
    a = blosc2.fromiter(it, dtype="i4, i4, i8", shape=(10,))
    b = blosc2.sort(a, order=order)
    narr = a[:]
    nb = np.sort(narr, order=order)
    assert np.array_equal(b[:], nb)


@pytest.mark.parametrize("order", ["f0", "f1", "f2", None])
def test_indices(order):
    it = ((x + 1, x - 2, -x) for x in range(10))
    a = blosc2.fromiter(it, dtype="i4, i4, i8", shape=(10,))
    b = a.indices(order=order)
    narr = a[:]
    nb = np.argsort(narr, order=order)
    assert np.array_equal(b[:], nb)


def test_save():
    a = blosc2.arange(0, 10, 1, dtype="i4", shape=(10,))
    blosc2.save(a, "test.b2nd")
    c = blosc2.open("test.b2nd")
    assert np.array_equal(a[:], c[:])
    blosc2.remove_urlpath("test.b2nd")
    with pytest.raises(FileNotFoundError):
        blosc2.open("test.b2nd")
