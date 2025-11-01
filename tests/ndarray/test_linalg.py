import inspect
from itertools import permutations

import numpy as np
import pytest
import torch

import blosc2
from blosc2.lazyexpr import linalg_funcs
from blosc2.ndarray import npvecdot


@pytest.mark.parametrize(
    ("ashape", "achunks", "ablocks"),
    {
        ((12, 10), (7, 5), (3, 3)),
        ((10,), (9,), (7,)),
        ((0,), (0,), (0,)),
        ((4, 10, 10), (2, 3, 4), (1, 2, 2)),
    },
)
@pytest.mark.parametrize(
    ("bshape", "bchunks", "bblocks"),
    {
        ((10,), (4,), (2,)),
        ((10, 5), (3, 4), (1, 3)),
        ((10, 12), (2, 4), (1, 2)),
        ((3, 10, 3), (2, 2, 4), (1, 1, 2)),
        ((0,), (0,), (0,)),
        ((6, 3, 10, 10), (5, 2, 3, 4), (2, 1, 2, 2)),
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
def test_matmul_complex(ashape, achunks, ablocks, bshape, bchunks, bblocks, dtype):
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
def test_matmul_scalars(scalar):
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


def test_matmul_disk():
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
    ("shape1", "chunk1", "block1", "shape2", "chunk2", "block2", "chunkres", "axes"),
    [
        # 1Dx1D->scalar (uneven chunks)
        ((50,), (17,), (5,), (50,), (13,), (5,), (), 1),
        # 2Dx2D->matrix multiplication
        (
            (30, 40),
            (17, 21),
            (8, 10),  # chunks not multiples of shape
            (40, 20),
            (19, 20),
            (9, 10),
            (10, 5),
            ([1], [0]),
        ),
        # 2Dx2D->axes arg integer
        ((10, 13), (7, 2), (3, 1), (12, 10), (4, 5), (3, 3), (3, 5), 1),
        # 3Dx3D->contraction along last/first
        (
            (10, 20, 30),
            (9, 11, 17),
            (5, 5, 5),  # uneven chunks
            (30, 15, 5),
            (16, 15, 5),
            (8, 15, 5),
            (7, 6, 3, 1),
            ([2], [0]),
        ),
        # 4Dx3D->contraction along two axes
        (
            (6, 7, 8, 9),
            (5, 6, 7, 8),
            (3, 3, 3, 3),
            (8, 9, 5),
            (7, 9, 5),
            (3, 5, 5),
            (4, 5, 2),
            ([2, 3], [0, 1]),
        ),
        # 2Dx1D->matrix-vector multiplication
        (
            (12, 7),
            (11, 7),
            (5, 7),  # chunks not multiples
            (7,),
            (5,),
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
            (2, 5, 3),
            ([2], [0]),
        ),
        # 1Dx3D->tensor contraction
        ((20,), (9,), (4,), (20, 4, 5), (19, 3, 5), (10, 2, 5), (3, 3), ([0], [0])),
        # 4Dx4D->reduce over 3 axes
        (
            (5, 6, 7, 8),
            (4, 5, 6, 7),
            (2, 3, 3, 4),
            (7, 8, 6, 10),
            (6, 7, 5, 9),
            (3, 4, 3, 5),
            (3, 7),
            ([1, 2, 3], [2, 0, 1]),
        ),
        # 5Dx5D->no reduce
        (
            (1, 2, 1, 5, 3),
            (1, 1, 1, 2, 2),
            (1, 1, 1, 1, 1),
            (2, 3, 2, 1, 5),
            (1, 2, 1, 1, 3),
            (1, 2, 1, 1, 1),
            (1, 2, 1, 2, 2, 2, 1, 2, 1, 3),  # output dims = 10
            ([], []),
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
def test_tensordot(shape1, chunk1, block1, shape2, chunk2, block2, chunkres, axes, dtype):
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
            blosc2.tensordot(a_b2, b_b2, axes=axes, chunks=chunkres)
    else:
        # Both should succeed
        res_np = np.tensordot(a_np, b_np, axes=axes)
        res_b2 = blosc2.tensordot(a_b2, b_b2, axes=axes, chunks=chunkres, fast_path=False)  # test slow path
        res_b2_np = res_b2[...]

        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)

        res_b2 = blosc2.tensordot(a_b2, b_b2, axes=axes, chunks=chunkres, fast_path=True)  # test fast path
        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)


@pytest.mark.parametrize(
    ("shape1", "chunk1", "block1", "shape2", "chunk2", "block2", "chunkres"),
    [
        # 1Dx1D->valid
        ((50,), (17,), (5,), (21,), (13,), (5,), (10, 5)),
        # 2Dx1D->error
        ((50, 22), (17, 21), (5, 3), (50,), (13,), (5,), (12, 13, 10)),
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
def test_outer(shape1, chunk1, block1, shape2, chunk2, block2, chunkres, dtype):
    # test outer
    # Create operands with requested dtype
    a_b2 = blosc2.arange(0, np.prod(shape1), shape=shape1, chunks=chunk1, blocks=block1, dtype=dtype)
    a_np = a_b2[()]  # decompress
    b_b2 = blosc2.arange(0, np.prod(shape2), shape=shape2, chunks=chunk2, blocks=block2, dtype=dtype)
    b_np = b_b2[()]  # decompress
    # NumPy reference and Blosc2 comparison
    res_np = np.outer(a_np, b_np)
    if len(shape1) > 1 or len(shape2) > 1:
        with pytest.raises(ValueError):
            res_b2 = blosc2.outer(a_b2, b_b2, chunks=chunkres, fast_path=False)  # test slow path
    else:
        res_b2 = blosc2.outer(a_b2, b_b2, chunks=chunkres, fast_path=False)  # test slow path
        res_b2_np = res_b2[...]

        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)

        res_b2 = blosc2.outer(a_b2, b_b2, chunks=chunkres, fast_path=True)  # test fast path
        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)


@pytest.mark.parametrize(
    ("shape1", "chunk1", "block1", "shape2", "chunk2", "block2", "chunkres", "axis"),
    [
        # 1Dx1D->scalar
        ((50,), (17,), (5,), (50,), (13,), (5,), (), -1),
        # 2Dx2D
        (
            (30, 40),
            (17, 21),
            (8, 10),
            (30, 40),
            (19, 20),
            (9, 10),
            (10,),
            -1,
        ),
        # 3Dx3D
        (
            (10, 1, 5),
            (9, 1, 1),
            (5, 1, 1),
            (10, 1, 1),
            (4, 1, 1),
            (3, 1, 1),
            (3, 3),
            -2,
        ),
        # 4Dx3D
        (
            (6, 7, 8, 9),
            (5, 6, 7, 8),
            (3, 3, 3, 3),
            (1, 7, 8, 1),
            (1, 7, 3, 1),
            (1, 3, 2, 1),
            (4, 5, 2),
            -2,
        ),
        # 2Dx1D->broadcastable to (12, 7)
        (
            (12, 7),
            (11, 7),
            (5, 7),
            (7,),
            (5,),
            (2,),
            (5,),
            -1,
        ),
        # 3Dx2D->broadcastable to (1, 6, 7)
        (
            (5, 6, 7),
            (4, 5, 6),
            (2, 3, 3),
            (6, 7),
            (6, 4),
            (3, 4),
            (3, 2),
            -2,
        ),
        # 1Dx3D -> broadcastable to (1, 1, 20)
        ((20,), (9,), (4,), (20, 4, 20), (19, 3, 5), (10, 2, 5), (10, 2), -1),
        # 4Dx4D
        (
            (5, 8, 1, 8),
            (4, 5, 1, 7),
            (2, 3, 1, 4),
            (1, 8, 6, 8),
            (1, 7, 5, 5),
            (1, 4, 3, 5),
            (2, 2, 2),
            -3,
        ),
        # 5Dx5D
        (
            (3, 4, 5, 6, 7),
            (2, 3, 4, 5, 6),
            (1, 2, 2, 3, 3),
            (3, 1, 1, 6, 7),
            (2, 1, 1, 3, 5),
            (2, 1, 1, 2, 4),
            (2, 2, 2, 5),
            -2,
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
        np.complex128,
    ],
)
def test_vecdot(shape1, chunk1, block1, shape2, chunk2, block2, chunkres, axis, dtype):
    # Create operands with requested dtype
    a_b2 = blosc2.arange(0, np.prod(shape1), shape=shape1, chunks=chunk1, blocks=block1, dtype=dtype)
    if dtype == np.complex128:
        a_b2 += 1j
        a_b2 = a_b2.compute()
    a_np = a_b2[()]  # decompress
    b_b2 = blosc2.arange(0, np.prod(shape2), shape=shape2, chunks=chunk2, blocks=block2, dtype=dtype)
    b_np = b_b2[()]  # decompress

    # NumPy reference and Blosc2 comparison
    np_raised = None
    try:
        res_np = npvecdot(a_np, b_np, axis=axis)
    except Exception as e:
        np_raised = type(e)

    if np_raised is not None:
        # Expect Blosc2 to raise the same type
        with pytest.raises(np_raised):
            blosc2.vecdot(a_b2, b_b2, axis=axis, chunks=chunkres)
    else:
        # Both should succeed
        res_np = npvecdot(a_np, b_np, axis=axis)
        res_b2 = blosc2.vecdot(a_b2, b_b2, axis=axis, chunks=chunkres, fast_path=False)  # test slow path
        res_b2_np = res_b2[...]

        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)

        res_b2 = blosc2.vecdot(a_b2, b_b2, axis=axis, chunks=chunkres, fast_path=True)  # test fast path
        # Assertions
        assert res_b2_np.shape == res_np.shape
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(res_b2_np, res_np, rtol=1e-5, atol=1e-6)
        else:
            np.testing.assert_array_equal(res_b2_np, res_np)


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
def test_tranpose_scalars(scalar):
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
def test_permutedims_complex(shape_chunks_blocks_3d, dtype, axes):
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
    [(2, 3), (4, 5, 6), (2, 4, 8, 5), (7, 3, 9, 9, 5)],
)
def test_mT(shape):
    arr = blosc2.linspace(0, 1, shape=shape)
    result = arr.mT
    try:
        expected = arr[:].mT
        np.testing.assert_allclose(result, expected)
    except AttributeError:
        pytest.skip("np.ndarray object in Numpy version {np.__version__} does not have .mT attribute.")


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


def test_tranpose_disk():
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
    with pytest.warns(DeprecationWarning, match="^transpose is deprecated"):
        at = blosc2.transpose(a)

    na = a[:]
    nat = np.transpose(na)

    np.testing.assert_allclose(at, nat)


@pytest.mark.parametrize(
    ("shape", "chunkshape", "offset"),
    [
        ((10, 10), (5, 5), 0),
        ((20, 15), (6, 7), 2),
        ((30, 25), (10, 8), -3),
        ((2, 4, 30, 25), (1, 3, 10, 8), -3),
    ],
)
def test_diagonal(shape, chunkshape, offset):
    # Create a Blosc2 NDArray with given shape and chunkshape
    a = blosc2.linspace(0, np.prod(shape), shape=shape, chunks=chunkshape)
    # Create random input data
    np_arr = a[()]

    # Compute diagonal with NumPy
    expected = np_arr.diagonal(offset=offset, axis1=-2, axis2=-1)

    # Compute diagonal with Blosc2
    result = blosc2.diagonal(a, offset=offset)

    # Convert back to NumPy for comparison
    result_np = result[:]

    # Assert equality
    np.testing.assert_array_equal(result_np, expected)


@pytest.mark.parametrize(
    "xp",
    [torch, np],
)
@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float32", "float64", "complex128"],
)
def test_linalgproxy(xp, dtype):
    dtype_ = getattr(xp, dtype) if hasattr(xp, dtype) else np.dtype(dtype)
    for name in linalg_funcs:
        if name == "transpose":
            continue  # deprecated
        func = getattr(blosc2, name)
        N = 10
        shape_a = (N,)
        chunks = (N // 3,)
        if name != "outer":
            shape_a *= 3
            chunks *= 3
        blosc_matrix = blosc2.full(shape=shape_a, fill_value=3, dtype=np.dtype(dtype), chunks=chunks)
        foreign_matrix = xp.ones(shape_a, dtype=dtype_)
        if dtype == "complex128":
            foreign_matrix += 0.5j
            blosc_matrix = blosc2.full(
                shape=shape_a, fill_value=3 + 2j, dtype=np.dtype(dtype), chunks=chunks
            )

        # Check this works
        argspec = inspect.getfullargspec(func)
        num_args = len(argspec.args)
        # handle numpy 1.26
        if name == "permute_dims":
            npfunc = blosc2.linalg.nptranspose
        elif name == "concat" and not hasattr(np, "concat"):
            npfunc = np.concatenate
        elif name == "matrix_transpose":
            npfunc = blosc2.linalg.nptranspose
        elif name == "vecdot":
            npfunc = blosc2.linalg.npvecdot
        else:
            npfunc = getattr(np, name)
        if num_args > 2 or name in ("outer", "matmul"):
            try:
                lexpr = func(blosc_matrix, foreign_matrix)
            except NotImplementedError:
                continue
            foreign_matrix = np.asarray(foreign_matrix)
            res = npfunc(blosc_matrix[()], foreign_matrix)
        else:
            try:
                lexpr = func(foreign_matrix)
            except NotImplementedError:
                continue
            except TypeError:
                continue
            foreign_matrix = np.asarray(foreign_matrix)
            res = npfunc(foreign_matrix, 0) if name == "expand_dims" else npfunc(foreign_matrix)
        np.testing.assert_array_equal(res, lexpr[()])
