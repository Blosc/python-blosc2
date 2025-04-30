from itertools import permutations

import numpy as np
import pytest

import blosc2


@pytest.fixture(
    params=[
        np.float64,
        pytest.param(np.int32, marks=pytest.mark.heavy),
        pytest.param(np.int64, marks=pytest.mark.heavy),
        pytest.param(np.float32, marks=pytest.mark.heavy),
    ]
)
def dtype_fixture(request):
    return request.param


@pytest.fixture(
    params=[
        ((10,), (5,), None),
        ((31,), (14,), (9,)),
        ((9,), (4,), (3,)),
    ]
)
def shape_chunks_blocks_1d(request):
    return request.param


@pytest.fixture(
    params=[
        ((4, 4), (3, 3), (2, 2)),
        ((12, 11), (7, 5), (6, 2)),
        ((6, 5), (5, 4), (4, 3)),
        pytest.param(((51, 603), (22, 99), (13, 29)), marks=pytest.mark.heavy),
    ]
)
def shape_chunks_blocks_2d(request):
    return request.param


@pytest.fixture(
    params=[
        ((4, 5, 2), (3, 4, 2), (3, 2, 1)),
        ((12, 10, 10), (11, 9, 7), (9, 7, 3)),
        pytest.param(((37, 63, 55), (12, 30, 41), (10, 5, 11)), marks=pytest.mark.heavy),
    ]
)
def shape_chunks_blocks_3d(request):
    return request.param


@pytest.fixture(
    params=[
        ((3, 3, 5, 7), (2, 3, 2, 4), (1, 2, 1, 4)),
        ((4, 6, 5, 2), (3, 3, 4, 2), (3, 2, 2, 1)),
        pytest.param(((10, 10, 10, 11), (7, 8, 9, 11), (6, 7, 8, 5)), marks=pytest.mark.heavy),
    ]
)
def shape_chunks_blocks_4d(request):
    return request.param


@pytest.mark.parametrize(
    "scalar",
    {
        1,  # int
        5.1,  # float
        1 + 2j,  # complex
        np.int8(2),  # NumPy int8
        np.int16(3),  # NumPy int16
        np.int32(4),  # NumPy int32
        np.int64(5),  # NumPy int64
        np.float32(5.2),  # NumPy float32
        np.float64(5.3),  # NumPy float64
        np.complex64(0 + 3j),  # NumPy complex64
        np.complex128(2 - 4j),  # NumPy complex128
    },
)
def test_scalars(scalar):
    scalar_t = blosc2.permute_dims(scalar)
    np_scalar_t = np.transpose(scalar)
    np.testing.assert_allclose(scalar_t, np_scalar_t)


def test_1d_permute_dims(shape_chunks_blocks_1d, dtype_fixture):
    shape, chunks, blocks = shape_chunks_blocks_1d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype_fixture)
    at = blosc2.permute_dims(a)

    na = a[:]
    nat = np.transpose(na)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    "axes",
    list(permutations([0, 1])),
)
def test_2d_permute_dims(shape_chunks_blocks_2d, dtype_fixture, axes):
    shape, chunks, blocks = shape_chunks_blocks_2d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype_fixture)
    at = blosc2.permute_dims(a, axes=axes)

    na = a[:]
    nat = np.transpose(na, axes=axes)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    "axes",
    list(permutations([0, 1, 2])),
)
def test_3d_permute_dims(shape_chunks_blocks_3d, dtype_fixture, axes):
    shape, chunks, blocks = shape_chunks_blocks_3d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype_fixture)
    at = blosc2.permute_dims(a, axes=axes)

    na = a[:]
    nat = np.transpose(na, axes=axes)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    "axes",
    list(permutations([0, 1, 2, 3])),
)
def test_4d_permute_dims(shape_chunks_blocks_4d, dtype_fixture, axes):
    shape, chunks, blocks = shape_chunks_blocks_4d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype_fixture)
    at = blosc2.permute_dims(a, axes=axes)

    na = a[:]
    nat = np.transpose(na, axes=axes)

    np.testing.assert_allclose(at, nat)


@pytest.mark.heavy
@pytest.mark.parametrize(
    "axes",
    list(permutations([0, 1, 2])),
)
@pytest.mark.parametrize(
    "dtype",
    {np.complex64, np.complex128},
)
def test_complex(shape_chunks_blocks_3d, dtype, axes):
    shape, chunks, blocks = shape_chunks_blocks_3d
    real_part = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype)
    imag_part = blosc2.linspace(1, 0, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype)
    complex_matrix = real_part + 3j * imag_part

    a = blosc2.asarray(complex_matrix)
    at = blosc2.permute_dims(a, axes=axes)

    na = a[:]
    nat = np.transpose(na, axes=axes)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    "axes",
    [
        (0, 0, 1),  # repeated axis
        (0, -1, -1),  # repeated negative
        (0, 1),  # missing one axis
        (0, 1, 2, 3),  # one more axis
        (0, 1, 3),  # out-of-range index
        (0, -4, 1),
    ],
)
def test_invalid_axes_raises(shape_chunks_blocks_3d, axes):
    shape, chunks, blocks = shape_chunks_blocks_3d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks)

    with pytest.raises(ValueError, match="not a valid permutation"):
        blosc2.permute_dims(a, axes=axes)


@pytest.mark.parametrize(
    "shape",
    [(2, 3), (4, 5, 6), (2, 4, 8, 5), (7, 3, 9, 9, 5)],
)
def test_matrix_transpose(shape):
    arr = blosc2.linspace(0, 1, shape=shape)
    result = blosc2.matrix_transpose(arr)

    expected = np.swapaxes(arr[:], -2, -1)

    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "shape",
    [
        (10,),
        (4, 5, 6),
        (2, 3, 4, 5),
    ],
)
def test_T_raises(shape):
    arr = blosc2.linspace(0, 1, shape=shape)
    with pytest.raises(ValueError, match="only works for 2-dimensional"):
        _ = arr.T


def test_disk():
    a = blosc2.linspace(0, 1, shape=(3, 4), urlpath="a_test.b2nd", mode="w")
    c = blosc2.permute_dims(a, urlpath="c_test.b2nd", mode="w")

    na = a[:]
    nc = np.transpose(na)

    np.testing.assert_allclose(c, nc, rtol=1e-6)
    blosc2.remove_urlpath("a_test.b2nd")
    blosc2.remove_urlpath("c_test.b2nd")


def test_transpose(shape_chunks_blocks_2d, dtype_fixture):
    shape, chunks, blocks = shape_chunks_blocks_2d
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype_fixture)
    with pytest.warns(DeprecationWarning):
        at = blosc2.transpose(a)

    na = a[:]
    nat = np.transpose(na)

    np.testing.assert_allclose(at, nat)
