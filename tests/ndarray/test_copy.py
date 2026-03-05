#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("shape", "dtype"), [([521], "i8"), ([20, 134, 13], "f4"), ([12, 13, 14, 15, 16], "f8")]
)
def test_simple(shape, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray)
    b = a.copy()
    np.testing.assert_almost_equal(b[...], nparray)


def test_cparams_vlmeta():
    a = blosc2.arange(0, 10, 1, dtype="i4", shape=(10,))
    a.vlmeta["name"] = "a"
    b = blosc2.copy(a)
    assert np.array_equal(a[:], b[:])
    assert a.vlmeta["name"] == b.vlmeta["name"]
    cparams = blosc2.CParams(clevel=9, codec=blosc2.Codec.LZ4)
    c = blosc2.copy(b, cparams=cparams)
    assert c.cparams.clevel == 9
    assert c.cparams.codec == blosc2.Codec.LZ4


@pytest.mark.parametrize(
    ("shape", "chunks1", "blocks1", "chunks2", "blocks2", "dtype"),
    [
        ([521], [212], [33], [121], [18], "|S8"),
        ([521], [212], [33], [121], [18], "|V8"),
        ([521], [212], [33], [121], [18], "f4,i8"),
        ([20, 134, 13], [10, 43, 10], [3, 13, 5], [10, 43, 10], [3, 6, 5], "|S4"),
        ([12, 13, 14, 15, 16], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], "|S8"),
    ],
)
def test_values(shape, chunks1, blocks1, chunks2, blocks2, dtype):
    dtype = np.dtype(dtype)
    typesize = dtype.itemsize
    size = int(np.prod(shape))
    buffer = bytes(size * typesize)
    cparams1 = blosc2.CParams(clevel=2)
    a = blosc2.frombuffer(buffer, shape, dtype=dtype, chunks=chunks1, blocks=blocks1, cparams=cparams1)
    cparams2 = {"clevel": 5, "filters": [blosc2.Filter.BITSHUFFLE], "filters_meta": [0]}
    b = a.copy(chunks=chunks2, blocks=blocks2, cparams=cparams2)
    assert a.shape == b.shape
    assert a.schunk.dparams == b.schunk.dparams
    for key, value in cparams2.items():
        if key in ("filters", "filters_meta"):
            assert getattr(b.schunk.cparams, key)[: len(value)] == value
            continue
        assert getattr(b.schunk.cparams, key) == value
    assert b.chunks == tuple(chunks2)
    assert b.blocks == tuple(blocks2)
    assert a.dtype == b.dtype

    buffer2 = b.tobytes()
    assert buffer == buffer2


@pytest.mark.parametrize(
    ("shape", "chunks1", "blocks1", "chunks2", "blocks2", "dtype"),
    [
        ([521], [212], [33], [121], [18], "i8"),
        ([521], [212], [33], [121], [18], "i8, f4"),
        ([20, 134, 13], [10, 43, 10], [3, 13, 5], [10, 43, 10], [3, 6, 5], "f4"),
        ([12, 13, 14, 15, 16], [6, 6, 6, 6, 6], [2, 2, 2, 2, 2], [7, 7, 7, 7, 7], [3, 3, 5, 3, 3], "f8"),
    ],
)
def test_copy_numpy(shape, chunks1, blocks1, chunks2, blocks2, dtype):
    size = int(np.prod(shape))
    dtype = np.dtype(dtype)
    if dtype.kind == "V":
        nparray = np.ones(size, dtype=dtype).reshape(shape)
    else:
        nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks1, blocks=blocks1)
    cparams = blosc2.CParams(clevel=5, filters=[blosc2.Filter.BITSHUFFLE], filters_meta=[0])
    b = a.copy(chunks=chunks2, blocks=blocks2, cparams=cparams)
    assert b.dtype == nparray.dtype
    if dtype.kind == "V":
        assert b.tobytes() == nparray.tobytes()
    else:
        np.testing.assert_almost_equal(b[...], nparray)
