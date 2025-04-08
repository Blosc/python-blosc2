import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    {
        ((12, 10), (7, 5), (3, 3)),
        ((10,), (9,), (7,)),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    {
        ((10,), (4,), (2,)),
        ((10, 5), (3, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 2)),
    },
)
@pytest.mark.parametrize(
    "dtype",
    {np.float32, np.float64},
)
def test_matmul(ashape, achunks, ablocks, bshape, bchunks, bblocks, dtype):
    a = blosc2.linspace(0, 1, dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 1, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)
    c = blosc2.matmul(a, b)

    na = a[:]
    nb = b[:]
    nc = np.matmul(na, nb)

    np.testing.assert_allclose(c, nc, rtol=1e-6)


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    {
        ((12, 10), (7, 5), (3, 3)),
        ((10,), (9,), (7,)),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    {
        ((10,), (4,), (2,)),
        ((10, 5), (3, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 2)),
    },
)
@pytest.mark.parametrize(
    "dtype",
    {np.complex64, np.complex128},
)
def test_complex(ashape, achunks, ablocks, bshape, bchunks, bblocks, dtype):
    real_part = blosc2.linspace(0, 1, shape=ashape, chunks=achunks, blocks=ablocks, dtype=dtype)
    imag_part = blosc2.linspace(0, 1, shape=ashape, chunks=achunks, blocks=ablocks, dtype=dtype)
    complex_matrix_a = real_part + 1j * imag_part
    a = blosc2.asarray(complex_matrix_a)

    real_part = blosc2.linspace(1, 2, shape=bshape, chunks=bchunks, blocks=bblocks, dtype=dtype)
    imag_part = blosc2.linspace(1, 2, shape=bshape, chunks=bchunks, blocks=bblocks, dtype=dtype)
    complex_matrix_b = real_part + 1j * imag_part
    b = blosc2.asarray(complex_matrix_b)

    c = blosc2.matmul(a, b)

    na = a[:]
    nb = b[:]
    nc = np.matmul(na, nb)

    np.testing.assert_allclose(c, nc, rtol=1e-6)


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    {
        ((12, 11), (7, 5), (3, 1)),
        ((0, 0), (0, 0), (0, 0)),
        ((10,), (4,), (2,)),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    {
        ((1, 5), (1, 4), (1, 3)),
        ((4, 6), (2, 4), (1, 3)),
        ((5,), (4,), (2,)),
    },
)
def test_shapes(ashape, achunks, ablocks, bshape, bchunks, bblocks):
    a = blosc2.linspace(0, 10, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 10, shape=bshape, chunks=bchunks, blocks=bblocks)

    with pytest.raises(ValueError):
        blosc2.matmul(a, b)

    with pytest.raises(ValueError):
        blosc2.matmul(b, a)


@pytest.mark.parametrize(
    "scalar",
    {
        5,  # int
        5.3,  # float
        1 + 2j,  # complex
        np.int8(5),  # NumPy int8
        np.int16(5),  # NumPy int16
        np.int32(5),  # NumPy int32
        np.int64(5),  # NumPy int64
        np.float32(5.3),  # NumPy float32
        np.float64(5.3),  # NumPy float64
        np.complex64(1 + 2j),  # NumPy complex64
        np.complex128(1 + 2j),  # NumPy complex128
    },
)
def test_scalars(scalar):
    vector = blosc2.asarray(np.array([1, 2, 3]))

    with pytest.raises(ValueError):
        blosc2.matmul(scalar, vector)

    with pytest.raises(ValueError):
        blosc2.matmul(vector, scalar)

    with pytest.raises(ValueError):
        blosc2.matmul(scalar, scalar)


@pytest.mark.parametrize(
    "ashape",
    [
        (12, 10, 10),
        (3, 3, 3),
    ],
)
@pytest.mark.parametrize(
    "bshape",
    [
        (10, 10, 10, 11),
        (3, 2),
    ],
)
def test_dims(ashape, bshape):
    a = blosc2.linspace(0, 10, shape=ashape)
    b = blosc2.linspace(0, 1, shape=bshape)

    with pytest.raises(ValueError):
        blosc2.matmul(a, b)

    with pytest.raises(ValueError):
        blosc2.matmul(b, a)


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks", "adtype"),
    {
        ((7, 10), (7, 5), (3, 5), np.float32),
        ((10,), (9,), (7,), np.complex64),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks", "bdtype"),
    {
        ((10,), (4,), (2,), np.float64),
        ((10, 6), (9, 4), (2, 3), np.complex128),
        ((10, 12), (2, 4), (1, 2), np.complex128),
    },
)
def test_special_cases(ashape, achunks, ablocks, adtype, bshape, bchunks, bblocks, bdtype):
    a = blosc2.linspace(0, 10, dtype=adtype, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 10, dtype=bdtype, shape=bshape, chunks=bchunks, blocks=bblocks)
    c = blosc2.matmul(a, b)

    na = a[:]
    nb = b[:]
    nc = np.matmul(na, nb)

    np.testing.assert_allclose(c, nc, rtol=1e-6)


def test_disk():
    a = blosc2.linspace(0, 1, shape=(3, 4), urlpath="a_test.b2nd", mode="w")
    b = blosc2.linspace(0, 1, shape=(4, 2), urlpath="b_test.b2nd", mode="w")
    c = blosc2.matmul(a, b, urlpath="c_test.b2nd", mode="w")

    na = a[:]
    nb = b[:]
    nc = np.matmul(na, nb)

    np.testing.assert_allclose(c, nc, rtol=1e-6)

    blosc2.remove_urlpath("a_test.b2nd")
    blosc2.remove_urlpath("b_test.b2nd")
    blosc2.remove_urlpath("c_test.b2nd")
