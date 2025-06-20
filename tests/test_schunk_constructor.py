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
    ("cparams", "dparams", "chunksize"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6}, {}, 40000),
        ({}, {"nthreads": 4}, 20000),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5}, {}, 20000),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 40000),
    ],
)
def test_schunk_numpy(contiguous, urlpath, cparams, dparams, chunksize):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    num_elem = 20 * 1000
    nchunks = num_elem * 4 // chunksize + 1 if num_elem * 4 % chunksize != 0 else num_elem * 4 // chunksize
    data = np.arange(num_elem, dtype="int32")
    bytes_obj = data.tobytes()
    schunk = blosc2.SChunk(chunksize=chunksize, data=data, **kwargs)
    # Test properties
    assert len(schunk) == num_elem
    assert chunksize == schunk.chunksize
    assert chunksize / 4 == schunk.chunkshape
    assert cparams.get("blocksize", 0) == schunk.blocksize
    assert nchunks == schunk.nchunks
    assert num_elem * 4 == schunk.nbytes
    assert schunk.nbytes / schunk.cbytes == schunk.cratio
    assert schunk.typesize == 4

    for i in range(nchunks):
        start = i * chunksize
        np_start = start // 4
        if i == (nchunks - 1):
            end = len(bytes_obj)
        else:
            end = (i + 1) * chunksize
        np_end = end // 4
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj[start:end]

        dest = np.empty(np_end - np_start, dtype=data.dtype)
        schunk.decompress_chunk(i, dest)
        assert np.array_equal(data[np_start:np_end], dest)

        schunk.decompress_chunk(i, memoryview(dest))
        assert np.array_equal(data[np_start:np_end], dest)

        dest = bytearray(data)
        schunk.decompress_chunk(i, dest[start:end])
        assert dest[start:end] == bytes_obj[start:end]

    for i in range(nchunks):
        schunk.get_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "chunksize"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 1}, {}, 500),
        ({"typesize": 1}, {"nthreads": 4}, 500),
        ({"typesize": 1}, {}, 1000),
        ({"typesize": 1}, blosc2.dparams_dflts, 1000),
    ],
)
def test_schunk(contiguous, urlpath, cparams, dparams, chunksize):
    storage = {"contiguous": contiguous, "urlpath": urlpath}

    blosc2.remove_urlpath(urlpath)
    nrep = 1000
    nchunks = 5 * nrep // chunksize + 1 if nrep * 5 % chunksize != 0 else 5 * nrep // chunksize

    buffer = b"1234 " * nrep
    schunk = blosc2.SChunk(chunksize=chunksize, data=buffer, cparams=cparams, dparams=dparams, **storage)

    for i in range(nchunks):
        start = i * chunksize
        if i == (nchunks - 1):
            end = len(buffer)
        else:
            end = (i + 1) * chunksize
        bytes_obj = buffer[start:end]
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj

        dest = bytearray(bytes_obj)
        schunk.decompress_chunk(i, dst=dest)
        assert dest == bytes_obj

    for i in range(nchunks):
        schunk.get_chunk(i)

    # Test properties
    assert chunksize == schunk.chunksize
    assert chunksize == schunk.chunkshape
    assert cparams.get("blocksize", 0) == schunk.blocksize
    assert nchunks == schunk.nchunks
    assert len(buffer) == schunk.nbytes
    assert schunk.nbytes / schunk.cbytes == schunk.cratio
    assert schunk.typesize == 1

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("cparams", "nitems"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, 0),
        ({"typesize": 4}, 200 * 1000),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, 200 * 1000 * 2 + 17),
    ],
)
@pytest.mark.parametrize(
    ("special_value", "expected_value"),
    [
        (blosc2.SpecialValue.ZERO, 0),
        (blosc2.SpecialValue.NAN, np.nan),
        (blosc2.SpecialValue.UNINIT, 0),
        (blosc2.SpecialValue.VALUE, 34),
        (blosc2.SpecialValue.VALUE, np.pi),
        (blosc2.SpecialValue.VALUE, b"0123"),
        (blosc2.SpecialValue.VALUE, True),
    ],
)
def test_schunk_fill_special(contiguous, urlpath, cparams, nitems, special_value, expected_value):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams}
    blosc2.remove_urlpath(urlpath)

    chunk_len = 200 * 1000
    schunk = blosc2.SChunk(chunksize=chunk_len * 4, **kwargs)
    if special_value in [blosc2.SpecialValue.ZERO, blosc2.SpecialValue.NAN, blosc2.SpecialValue.UNINIT]:
        schunk.fill_special(nitems, special_value)
    else:
        schunk.fill_special(nitems, special_value, expected_value)
    assert len(schunk) == nitems

    if special_value != blosc2.SpecialValue.UNINIT:
        dtype = np.int32
        if isinstance(expected_value, float):
            dtype = np.float32
        elif isinstance(expected_value, bytes):
            dtype = np.dtype(f"|S{len(expected_value)}")
        array = np.full(nitems, expected_value, dtype=dtype)
        dest = np.empty(nitems, dtype=dtype)
        schunk.get_slice(out=dest)
        if dtype in [np.float32, np.float64]:
            np.testing.assert_allclose(dest, array)
        else:
            np.testing.assert_equal(dest, array)

    blosc2.remove_urlpath(urlpath)
