#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from dataclasses import asdict

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "fill_value", "dtype", "cparams", "dparams", "urlpath", "contiguous"),
    [
        (
            (100, 1230),
            (200, 100),
            (55, 3),
            b"0123",
            None,
            {"clevel": 4, "use_dict": 0, "nthreads": 1},
            {"nthreads": 1},
            None,
            False,
        ),
        (
            (23, 34),
            (20, 20),
            (10, 10),
            b"sun",
            None,
            blosc2.CParams(codec=blosc2.Codec.LZ4HC, clevel=8, use_dict=False, nthreads=2),
            {"nthreads": 2},
            "full.b2nd",
            True,
        ),
        (
            (80, 51, 60),
            (20, 10, 33),
            (6, 6, 26),
            3.14,
            np.float64,
            {"codec": blosc2.Codec.ZLIB, "clevel": 5, "use_dict": True, "nthreads": 2},
            {"nthreads": 1},
            "full.b2nd",
            False,
        ),
        (
            (13, 13),
            (12, 12),
            (11, 11),
            123456789,
            None,
            blosc2.CParams(codec=blosc2.Codec.LZ4HC, clevel=8, use_dict=False, nthreads=2),
            {"nthreads": 2},
            None,
            True,
        ),
    ],
)
def test_full(shape, chunks, blocks, fill_value, cparams, dparams, dtype, urlpath, contiguous):
    blosc2.remove_urlpath(urlpath)
    storage = {"urlpath": urlpath, "contiguous": contiguous}
    a = blosc2.full(
        shape,
        fill_value,
        chunks=chunks,
        blocks=blocks,
        dtype=dtype,
        cparams=cparams,
        dparams=blosc2.DParams(**dparams),
        **storage,
    )
    assert asdict(a.schunk.dparams) == dparams
    if isinstance(fill_value, bytes):
        dtype = np.dtype(f"S{len(fill_value)}")
    assert a.dtype == np.dtype(dtype) if dtype is not None else np.dtype(np.uint8)

    b = np.full(shape=shape, fill_value=fill_value, dtype=a.dtype)
    tol = 1e-5 if dtype is np.float32 else 1e-14
    if dtype in (np.float32, np.float64):
        np.testing.assert_allclose(a[...], b, rtol=tol, atol=tol)
    else:
        np.array_equal(a[...], b)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("shape", "fill_value", "dtype"),
    [
        ((100, 1230), b"0123", None),
        ((23, 34), b"sun", None),
        ((80, 51, 60), 3.14, "f8"),
        ((13, 13), 123456789, None),
    ],
)
def test_full_simple(shape, fill_value, dtype):
    a = blosc2.full(shape, fill_value)
    if isinstance(fill_value, bytes):
        dtype = np.dtype(f"S{len(fill_value)}")
    assert a.dtype == np.dtype(dtype) if dtype is not None else np.dtype(np.uint8)

    b = np.full(shape=shape, fill_value=fill_value, dtype=a.dtype)
    tol = 1e-5 if dtype is np.float32 else 1e-14
    if dtype in (np.float32, np.float64):
        np.testing.assert_allclose(a[...], b, rtol=tol, atol=tol)
    else:
        np.array_equal(a[...], b)


def test_ones():
    # This is based on blosc2.full, so a full test is not really needed
    shape = (10, 10)
    a = blosc2.ones(shape, dtype=np.float32)
    assert a.shape == shape
    assert a.dtype == np.float32
    assert isinstance(a, blosc2.NDArray)
    b = np.ones(shape, dtype=np.float32)
    np.testing.assert_allclose(a[:], b)


@pytest.mark.parametrize("asarray", [True, False])
@pytest.mark.parametrize("typesize", [255, 256, 257, 261, 256 * 256])
@pytest.mark.parametrize("shape", [(1,), (3,), (10,), (1024,)])
def test_large_typesize(shape, typesize, asarray):
    dtype = np.dtype([("f_001", "<i1", (typesize,))])
    a = np.full(shape, 3, dtype=dtype)
    if asarray:
        b = blosc2.asarray(a)
    else:
        b = blosc2.full(shape, 3, dtype=dtype)
    assert np.array_equal(b[0], a[0])


def test_complex_datatype():
    dtype = np.dtype(
        [
            ("f_001", "<f4", (164,)),
            ("f_002", "<f4", (11,)),
            ("f_003", "<f4", (154,)),
            ("f_004", "<f4", (870,)),
            ("f_005", "<f4", (1062,)),
            ("f_006", "<f4", (22,)),
            ("f_007", "<f4", (44,)),
            ("f_008", "<f4", (512,)),
            ("f_009", "<f4", (64, 77)),
            ("f_010", "<f4", (97, 489)),
            ("f_011", "<f4", (75, 255)),
            ("f_012", "<f4", (8, 293)),
            ("f_013", "<f4", (230, 591)),
            ("f_014", "<f4", (101, 193)),
            ("f_015", "<f4", (12, 48)),
            ("f_016", "<f4", (90, 699)),
            ("f_017", "<f4", (125, 65)),
            ("f_018", "<f4", (132, 81)),
            ("f_019", "<f4", (27, 363)),
            ("f_020", "S1000"),
            ("f_021", "S1000"),
        ]
    )
    a = np.zeros((256,), dtype=dtype)
    cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1, nthreads=3)
    b = blosc2.asarray(a, cparams=cparams, urlpath="b.b2nd", mode="w")
    # Iterate over the fields of the structured array and check that the data is the same
    for field in dtype.fields:
        # TODO: the next is not working
        # np.testing.assert_allclose(b[field][:], a[field], rtol=1e-5, atol=1e-5)
        assert np.array_equal(b[field][:], a[field])
    blosc2.remove_urlpath("b.b2nd")
