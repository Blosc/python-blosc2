import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    {
        ((12, 10), (7, 5), (3, 3)),
        ((10,), (9,), (7,)),
        ((0,), (0,), (0,)),
        ((40, 10, 10), (2, 3, 4), (1, 2, 2)),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    {
        ((10,), (4,), (2,)),
        ((10, 5), (3, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 2)),
        ((200, 10, 22), (23, 2, 4), (4, 1, 2)),
        ((0,), (0,), (0,)),
        ((20, 40, 10, 10), (5, 2, 3, 4), (2, 1, 2, 2)),
    },
)
@pytest.mark.parametrize(
    "dtype",
    {np.float32, np.float64},
)
def test_matmul(ashape, achunks, ablocks, bshape, bchunks, bblocks, dtype):
    a = blosc2.linspace(0, 1, dtype=dtype, shape=ashape, chunks=achunks, blocks=ablocks)
    b = blosc2.linspace(0, 1, dtype=dtype, shape=bshape, chunks=bchunks, blocks=bblocks)
    a_np = a[:]
    b_np = b[:]
    try:
        np_res = np.matmul(a_np, b_np)
        np_error = None
    except ValueError as e:
        np_res = None
        np_error = e

    if np_error is not None:
        with pytest.raises(type(np_error)):
            blosc2.matmul(a, b)
    else:
        b2_res = blosc2.matmul(a, b)
        np.testing.assert_allclose(b2_res[()], np_res, rtol=1e-6)


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
    a_np = a[:]
    b_np = b[:]

    try:
        np_res = np.matmul(a_np, b_np)
        np_error = None
    except ValueError as e:
        np_res = None
        np_error = e

    if np_error is not None:
        with pytest.raises(type(np_error)):
            blosc2.matmul(a, b)
    else:
        b2_res = blosc2.matmul(a, b)
        np.testing.assert_allclose(b2_res[:], np_res)


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


@pytest.mark.parametrize(
    ("shape1", "chunk1", "block1", "shape2", "chunk2", "block2", "axes"),
    [
        # 1Dx1D->scalar (uneven chunks)
        ((50,), (17,), (5,), (50,), (13,), (5,), 1),
        # 2Dx2D->matrix multiplication
        (
            (30, 40),
            (17, 21),
            (8, 10),  # chunks not multiples of shape
            (40, 20),
            (19, 20),
            (9, 10),
            ([1], [0]),
        ),
        # 3Dx3D->contraction along last/first
        (
            (10, 20, 30),
            (9, 11, 17),
            (5, 5, 5),  # uneven chunks
            (30, 15, 5),
            (16, 15, 5),
            (8, 15, 5),
            ([2], [0]),
        ),
        # 4Dx3D->contraction along two axes
        ((6, 7, 8, 9), (5, 6, 7, 8), (3, 3, 3, 3), (8, 9, 5), (7, 9, 5), (3, 5, 5), ([2, 3], [0, 1])),
        # 2Dx1D->matrix-vector multiplication
        (
            (12, 7),
            (11, 7),
            (5, 7),  # chunks not multiples
            (7,),
            (5,),
            (5,),
            ([1], [0]),
        ),
        # 3Dx2D->like batched matmul
        (
            (5, 6, 7),
            (4, 5, 6),
            (2, 3, 3),  # uneven chunks
            (7, 4),
            (6, 4),
            (3, 4),
            ([2], [0]),
        ),
        # 1Dx3D->tensor contraction
        ((20,), (9,), (4,), (20, 4, 5), (19, 3, 5), (10, 2, 5), ([0], [0])),
        # 4Dx4D->reduce over 3 axes
        (
            (5, 6, 7, 8),
            (4, 5, 6, 7),
            (2, 3, 3, 4),
            (7, 8, 6, 10),
            (6, 7, 5, 9),
            (3, 4, 3, 5),
            ([1, 2, 3], [0, 1, 2]),
        ),
        # 5Dx5D->reduce over 4 axes
        (
            (3, 4, 5, 6, 7),
            (2, 3, 4, 5, 6),
            (1, 2, 2, 3, 3),
            (5, 6, 7, 4, 8),
            (4, 5, 6, 3, 7),
            (2, 3, 3, 2, 4),
            ([1, 2, 3, 4], [0, 1, 2, 3]),
        ),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        np.int32,
        np.int64,
        np.float32,
        np.float64,
    ],
)
def test_tensordot(shape1, chunk1, block1, shape2, chunk2, block2, axes, dtype):
    # Create operands with requested dtype
    a_b2 = blosc2.arange(0, np.prod(shape1), shape=shape1, chunks=chunk1, blocks=block1, dtype=dtype)
    a_np = a_b2[()]  # decompress
    b_b2 = blosc2.arange(0, np.prod(shape2), shape=shape2, chunks=chunk2, blocks=block2, dtype=dtype)
    b_np = b_b2[()]  # decompress

    # NumPy reference and Blosc2 comparison
    np_raised = None
    try:
        res_np = np.tensordot(a_np, b_np, axes=axes)
    except Exception as e:
        np_raised = type(e)

    if np_raised is not None:
        # Expect Blosc2 to raise the same type
        with pytest.raises(np_raised):
            blosc2.tensordot(a_b2, b_b2, axes=axes)
    else:
        # Both should succeed
        res_np = np.tensordot(a_np, b_np, axes=axes)
        res_b2 = blosc2.tensordot(a_b2, b_b2, axes=axes)
        res_b2_np = res_b2[...]

        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)
