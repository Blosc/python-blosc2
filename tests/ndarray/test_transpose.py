import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    {
        ((3, 3), (2, 2), (1, 1)),
        ((12, 11), (7, 5), (6, 2)),
        ((1, 5), (1, 4), (1, 3)),
        ((201, 1903), None, None),
        ((10,), (5,), None),
    },
)
@pytest.mark.parametrize(
    "dtype",
    {np.int32, np.int64, np.float32, np.float64},
)
def test_transpose(shape, chunks, blocks, dtype):
    a = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype)
    at = blosc2.transpose(a)

    na = a[:]
    nat = np.transpose(na)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    {
        ((3, 3), (2, 2), (1, 1)),
        ((12, 11), (7, 5), (6, 2)),
        ((1, 5), (1, 4), (1, 3)),
        ((201, 1903), None, None),
        ((10,), (5,), None),
    },
)
@pytest.mark.parametrize(
    "dtype",
    {np.complex64, np.complex128},
)
def test_complex(shape, chunks, blocks, dtype):
    real_part = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype)
    imag_part = blosc2.linspace(0, 1, shape=shape, chunks=chunks, blocks=blocks, dtype=dtype)
    complex_matrix = real_part + 1j * imag_part

    a = blosc2.asarray(complex_matrix)
    at = blosc2.transpose(a)

    na = a[:]
    nat = np.transpose(na)

    np.testing.assert_allclose(at, nat)


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
    at = blosc2.transpose(scalar)
    nat = np.transpose(scalar)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3),
        (12, 10, 10),
        (10, 10, 10, 11),
        (5, 4, 3, 2, 1, 0),
    ],
)
def test_dims(shape):
    a = blosc2.linspace(0, 1, shape=shape)

    with pytest.raises(ValueError):
        blosc2.transpose(a)


def test_disk():
    a = blosc2.linspace(0, 1, shape=(3, 4), urlpath="a_test.b2nd", mode="w")
    c = blosc2.transpose(a, urlpath="c_test.b2nd", mode="w")

    na = a[:]
    nc = np.transpose(na)

    np.testing.assert_allclose(c, nc, rtol=1e-6)

    blosc2.remove_urlpath("a_test.b2nd")
    blosc2.remove_urlpath("c_test.b2nd")
