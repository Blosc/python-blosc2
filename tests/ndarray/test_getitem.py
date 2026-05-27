#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import math

import numpy as np
import pytest

import blosc2

argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 56, 3), slice(100, 50, -4), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
    (
        [12, 13, 14, 15, 16],
        [5, 5, 5, 5, 5],
        [2, 2, 2, 2, 2],
        (None, slice(1, 3), None, ..., slice(3, 6)),
        np.float32,
    ),
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
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1), bool_array((6, 5))),
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1), bool_array((6, 0, 4))),
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1), True),
        ((6, 5, 4, 3), (3, 2, 2, 1), (1, 1, 1, 1), False),
    ],
)
def test_bool_values(shape, chunks, blocks, idx):
    npa = np.arange(int(np.prod(shape)), dtype=np.int32).reshape(shape)
    b2a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    assert b2a[idx].shape == npa[idx].shape
    assert b2a[idx].dtype == npa[idx].dtype
    assert b2a[idx].size == npa[idx].size
    assert b2a[idx].ndim == npa[idx].ndim


def test_dense_bool_ndarray_mask_no_recursion():
    nitems = 60_000
    npa = np.arange(nitems, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(20_000,))
    mask = blosc2.asarray(np.ones(nitems, dtype=np.bool_), chunks=(20_000,))

    np.testing.assert_array_equal(a[mask], npa)


def test_lazyexpr_where_full_slice_no_recursion():
    nitems = 60_000
    a = blosc2.linspace(0, 1, nitems, chunks=(20_000,))
    expected = np.linspace(0, 1, nitems)

    np.testing.assert_allclose(a[a < 5][:], expected)


def test_lazyexpr_where_full_slice_persisted_reuses_shared_chunk_cache(tmp_path):
    nitems = 60_000
    expected = np.linspace(0, 1, nitems)
    a = blosc2.asarray(
        expected, chunks=(20_000,), blocks=(2_000,), urlpath=str(tmp_path / "persisted.b2nd"), mode="w"
    )
    old_nthreads = blosc2.nthreads
    blosc2.set_nthreads(max(2, old_nthreads))
    try:
        for _ in range(10):
            np.testing.assert_allclose(a[a < 5][:], expected)
    finally:
        blosc2.set_nthreads(old_nthreads)


def test_sparse_bool_mask_routes_through_take_fastpath(monkeypatch):
    nitems = 120_000
    npa = np.arange(nitems, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(20_000,))
    mask = np.zeros(nitems, dtype=np.bool_)
    mask[[1, 10, 11_111, 55_555, nitems - 1]] = True

    call_count = {"take": 0}
    original_take = blosc2.take

    def wrapped_take(*args, **kwargs):
        call_count["take"] += 1
        return original_take(*args, **kwargs)

    monkeypatch.setattr(blosc2, "take", wrapped_take)

    np.testing.assert_array_equal(a[mask], npa[mask])
    assert call_count["take"] == 1


def test_dense_bool_mask_skips_take_fastpath(monkeypatch):
    nitems = 60_000
    npa = np.arange(nitems, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(20_000,))
    mask = np.ones(nitems, dtype=np.bool_)

    call_count = {"take": 0}
    original_take = blosc2.take

    def wrapped_take(*args, **kwargs):
        call_count["take"] += 1
        return original_take(*args, **kwargs)

    monkeypatch.setattr(blosc2, "take", wrapped_take)

    np.testing.assert_array_equal(a[mask], npa[mask])
    assert call_count["take"] == 0


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


def test_take_1d_uses_sparse_path_matches_numpy(tmp_path):
    npa = np.arange(1000, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(128,), urlpath=tmp_path / "take_sparse.b2nd", mode="w")
    idx = np.array([999, 998, 997, 997, 500, 129, 128, 127, 126, 33, 32, 31, 31, 0], dtype=np.int64)

    np.testing.assert_array_equal(a.take(idx)[()], npa[idx])
    np.testing.assert_array_equal(a[idx], npa[idx])


def test_take_1d_sparse_path_negative_indices():
    npa = np.arange(20, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(8,))
    idx = np.array([-1, -5, 0, 3], dtype=np.int64)

    np.testing.assert_array_equal(a.take(idx)[()], npa[idx])
    np.testing.assert_array_equal(a[idx], npa[idx])


def test_take_1d_sparse_path_structured_non_behaved_partitions():
    npa = np.empty((100,), dtype=[("a", np.int32), ("b", np.int32)])
    npa["a"] = np.arange(1, 101)
    npa["b"] = np.arange(200, 100, -1)
    a = blosc2.asarray(npa, chunks=(44,), blocks=(33,))

    for idx in [
        np.arange(2, 100),
        np.arange(99, 1, -1),
        np.array([5, 1, 5, 99, 0, 44, 43], dtype=np.int64),
    ]:
        np.testing.assert_array_equal(a.take(idx)[()], npa[idx])
        np.testing.assert_array_equal(a[idx], npa[idx])


def test_ndarray_take_1d_matches_numpy():
    npa = np.arange(20, dtype=np.int32)
    a = blosc2.asarray(npa, chunks=(7,))
    idx = np.array([5, 1, -1, 5, 0], dtype=np.int64)

    result = a.take(idx)
    assert isinstance(result, blosc2.NDArray)
    np.testing.assert_array_equal(result[()], np.take(npa, idx))


def test_ndarray_take_axis_with_nd_indices_matches_numpy():
    npa = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    a = blosc2.asarray(npa, chunks=(2, 2, 3))
    idx = np.array([[3, 0], [1, -1]], dtype=np.int64)

    expected = np.take(npa, idx, axis=1)
    result = a.take(idx, axis=1)
    top_level_result = blosc2.take(a, idx, axis=1)
    assert isinstance(result, blosc2.NDArray)
    assert isinstance(top_level_result, blosc2.NDArray)
    np.testing.assert_array_equal(result[()], expected)
    np.testing.assert_array_equal(top_level_result[()], expected)


def test_ndarray_take_axis_none_nd_fallback_matches_numpy():
    npa = np.arange(3 * 4 * 5, dtype=np.int32).reshape(3, 4, 5)
    a = blosc2.asarray(npa, chunks=(2, 2, 3))
    idx = np.array([[0, -1], [17, 5]], dtype=np.int64)

    expected = np.take(npa, idx, axis=None)
    result = a.take(idx)
    top_level_result = blosc2.take(a, idx)
    assert isinstance(result, blosc2.NDArray)
    assert isinstance(top_level_result, blosc2.NDArray)
    np.testing.assert_array_equal(result[()], expected)
    np.testing.assert_array_equal(top_level_result[()], expected)


def test_ndarray_take_rejects_bad_indices_and_axis():
    a = blosc2.asarray(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(TypeError, match="integer"):
        a.take(np.array([1.5]), axis=0)
    with pytest.raises(ValueError, match="axis"):
        a.take([0], axis=2)
    with pytest.raises(IndexError, match="bounds"):
        a.take([3], axis=0)


@pytest.mark.parametrize(
    ("shape", "chunkshape", "axis", "indices"),
    [
        ((10, 10), (5, 5), 0, [0, 5, 9]),
        ((20, 15), (6, 7), 1, [1, 3, 7, 14]),
        ((30, 25), (10, 8), 0, [2, 10, 20]),
    ],
)
def test_take(shape, chunkshape, axis, indices):
    # Create predictable input
    np_arr = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    # Wrap into Blosc2 NDArray
    a = blosc2.asarray(np_arr, chunks=chunkshape)

    # NumPy expected
    expected = np.take(np_arr, indices, axis=axis)

    # Blosc2 result
    result = blosc2.take(a, indices, axis=axis)

    # Compare
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunkshape", "axis"),
    [
        ((8, 6), (4, 3), 1),
        ((12, 7), (6, 7), 0),
        ((5, 9), (5, 3), 1),
    ],
)
def test_take_along_axis(shape, chunkshape, axis):
    # Create predictable input
    np_arr = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    # Wrap into Blosc2 NDArray
    a = blosc2.asarray(np_arr, chunks=chunkshape)

    # Make some indices with same shape except for the given axis
    indices_shape = list(shape)
    indices_shape[axis] = 2  # we'll take 2 indices along that axis
    rng = np.random.default_rng()
    indices = rng.integers(0, shape[axis], size=indices_shape)

    # NumPy expected
    expected = np.take_along_axis(np_arr, indices, axis=axis)

    # Blosc2 result
    result = blosc2.take_along_axis(a, indices, axis=axis)

    # Compare
    np.testing.assert_array_equal(result[()], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "axis", "indices"),
    [
        # 2D
        ((6, 7), (4, 5), (3, 4), 0, [0, 3, 5]),
        ((6, 7), (4, 5), (3, 4), 1, [0, 3, 6]),
        ((20, 15), (6, 7), (3, 4), 0, [0, 10, 19]),
        ((20, 15), (6, 7), (3, 4), 1, [0, 7, 14]),
        # 3D
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 0, [0, 2, 4]),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 1, [0, 3, 5]),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 2, [0, 3, 6]),
        ((9, 10, 11), (4, 5, 6), (2, 3, 3), 0, [0, 4, 8]),
        ((9, 10, 11), (4, 5, 6), (2, 3, 3), 1, [0, 5, 9]),
        ((9, 10, 11), (4, 5, 6), (2, 3, 3), 2, [0, 5, 10]),
        # 4D
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), 0, [0, 2, 3]),
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), 2, [0, 3, 5]),
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), 3, [0, 3, 6]),
    ],
)
def test_ndarray_take_ndim(shape, chunks, blocks, axis, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    expected = np.take(npa, indices, axis=axis)
    result = a.take(indices, axis=axis)
    top_result = blosc2.take(a, indices, axis=axis)

    assert isinstance(result, blosc2.NDArray)
    assert isinstance(top_result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], expected)
    np.testing.assert_array_equal(top_result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "indices"),
    [
        # 2D, 3D, 4D with axis=None
        ((6, 7), (4, 5), (3, 4), [0, 10, 41]),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), [0, 50, 209]),
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), [0, 100, 500, 839]),
    ],
)
def test_ndarray_take_ndim_axis_none(shape, chunks, blocks, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    expected = np.take(npa, indices, axis=None)
    result = a.take(indices)
    top_result = blosc2.take(a, indices)

    assert isinstance(result, blosc2.NDArray)
    assert isinstance(top_result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], expected)
    np.testing.assert_array_equal(top_result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "axis", "indices"),
    [
        # 2D, 3D, 4D with multi-dim index arrays
        ((6, 7), (4, 5), (3, 4), 1, np.array([[0, 3], [6, 2]])),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 0, np.array([[0, 2], [4, 1]])),
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), 2, np.array([[0, 3], [5, 1]])),
    ],
)
def test_ndarray_take_ndim_multidim_indices(shape, chunks, blocks, axis, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    expected = np.take(npa, indices, axis=axis)
    result = a.take(indices, axis=axis)

    assert isinstance(result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "axis", "indices"),
    [
        # Negative indices
        ((6, 7), (4, 5), (3, 4), 0, [-1, -3, 0]),
        ((6, 7), (4, 5), (3, 4), 1, [-1, -7, 3, 0]),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 2, [-1, -7, 3]),
        # Duplicate indices
        ((6, 7), (4, 5), (3, 4), 0, [0, 5, 0, 5, 3]),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 1, [3, 3, 5, 5, 0]),
        # Single index (scalar-like list)
        ((6, 7), (4, 5), (3, 4), 0, [3]),
        ((6, 7), (4, 5), (3, 4), 1, [0]),
        # Empty indices
        ((6, 7), (4, 5), (3, 4), 0, []),
        ((6, 7), (4, 5), (3, 4), 1, []),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 0, []),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 1, []),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 2, []),
    ],
)
def test_ndarray_take_ndim_edge_cases(shape, chunks, blocks, axis, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    expected = np.take(npa, indices, axis=axis)
    result = a.take(indices, axis=axis)

    assert isinstance(result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "axis"),
    [
        # 2D with non-behaved (non-even) partitions
        ((7, 11), (5, 7), (3, 5), 0),
        ((7, 11), (5, 7), (3, 5), 1),
        # 3D with non-behaved partitions
        ((7, 11, 13), (5, 7, 8), (3, 4, 5), 0),
        ((7, 11, 13), (5, 7, 8), (3, 4, 5), 1),
        ((7, 11, 13), (5, 7, 8), (3, 4, 5), 2),
    ],
)
def test_ndarray_take_ndim_non_behaved_partitions(shape, chunks, blocks, axis):
    npa = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    rng = np.random.default_rng(42)
    indices = rng.integers(0, shape[axis], size=min(shape[axis], 8)).tolist()

    expected = np.take(npa, indices, axis=axis)
    result = a.take(indices, axis=axis)

    assert isinstance(result, blosc2.NDArray)
    np.testing.assert_array_equal(result[:], expected)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "axis"),
    [
        # Different dtypes
        ((6, 7), (4, 5), (3, 4), 0),
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), 1),
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), 2),
    ],
)
def test_ndarray_take_ndim_dtypes(shape, chunks, blocks, axis):
    for dtype in [np.int32, np.int64, np.float32, np.float64, np.complex128]:
        npa = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
        a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

        rng = np.random.default_rng(42)
        indices = rng.integers(0, shape[axis], size=min(shape[axis], 5)).tolist()

        expected = np.take(npa, indices, axis=axis)
        result = a.take(indices, axis=axis)

        assert isinstance(result, blosc2.NDArray)
        np.testing.assert_array_equal(result[:], expected)


# --- __getitem__ fancy indexing with integer arrays (uses b2nd_get_sparse_cbuffer) ---


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "indices"),
    [
        # 1-D with 1-D index (was already sparse, regression check)
        ((100,), (23,), (7,), [0, 5, 50, 99]),
        # 1-D with 2-D index (was fancy indexing before, now sparse)
        ((100,), (23,), (7,), [[1, 3], [5, 7]]),
        # 2-D with 1-D index (was fancy indexing before, now sparse)
        ((6, 7), (4, 5), (3, 4), [0, 3, 5]),
        ((20, 15), (6, 7), (3, 4), [0, 10, 19]),
        # 2-D with 2-D index (was fancy indexing before, now sparse)
        ((6, 7), (4, 5), (3, 4), [[0, 3], [5, 2]]),
        # 3-D with 1-D index
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), [0, 2, 4]),
        # 3-D with 2-D index
        ((5, 6, 7), (3, 4, 5), (2, 2, 3), [[0, 2], [4, 1]]),
        # 4-D with 1-D index
        ((4, 5, 6, 7), (3, 3, 4, 5), (2, 2, 2, 3), [0, 2, 3]),
    ],
)
def test_getitem_integer_array_fancy_index(shape, chunks, blocks, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa, chunks=chunks, blocks=blocks)

    expected = npa[indices]
    result = a[indices]

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("shape", "indices"),
    [
        ((6, 7), [-1, 0, 3, -3]),
        ((6, 7), [0, 5, 0, 5, 3]),
        ((6, 7), [3]),
        ((6, 7), []),
        ((5, 6, 7), [-1, 0, 4, -2]),
        ((5, 6, 7), [0, 4, 0, 2]),
        ((5, 6, 7), [2]),
        ((5, 6, 7), []),
    ],
)
def test_getitem_integer_array_edge_cases(shape, indices):
    npa = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    a = blosc2.asarray(npa)

    expected = npa[indices]
    result = a[indices]

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, expected)


def test_getitem_integer_array_out_of_bounds():
    a = blosc2.asarray(np.arange(12, dtype=np.int32).reshape(3, 4))
    with pytest.raises(IndexError, match="bounds"):
        _ = a[[3]]
    with pytest.raises(IndexError, match="bounds"):
        _ = a[[-4]]


def test_getitem_integer_array_still_uses_fancy_for_boolean():
    """Boolean arrays should NOT be routed through the sparse path."""
    a = blosc2.asarray(np.arange(12, dtype=np.int32).reshape(3, 4))
    mask = np.array([True, False, True])
    expected = np.arange(12, dtype=np.int32).reshape(3, 4)[mask]
    result = a[mask]
    np.testing.assert_array_equal(result, expected)
