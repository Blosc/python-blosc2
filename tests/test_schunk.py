#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize("mode", ["r", "w", "a"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 0),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 10),
    ],
)
def test_schunk_numpy(contiguous, urlpath, mode, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    if mode != "r" or urlpath is None:
        chunk_len = 200 * 1000
        schunk = blosc2.SChunk(chunksize=chunk_len * 4, mode=mode, **storage)
        assert schunk.urlpath == urlpath
        assert schunk.contiguous == contiguous

        for i in range(nchunks):
            buffer = i * np.arange(chunk_len, dtype="int32")
            nchunks_ = schunk.append_data(buffer)
            assert nchunks_ == (i + 1)

        for i in range(nchunks):
            buffer = i * np.arange(chunk_len, dtype="int32")
            bytes_obj = buffer.tobytes()
            res = schunk.decompress_chunk(i)
            assert res == bytes_obj

            dest = np.empty(buffer.shape, buffer.dtype)
            schunk.decompress_chunk(i, dest)
            assert np.array_equal(buffer, dest)

            schunk.decompress_chunk(i, memoryview(dest))
            assert np.array_equal(buffer, dest)

            dest = bytearray(buffer)
            schunk.decompress_chunk(i, dest)
            assert dest == bytes_obj

        for i in range(nchunks):
            schunk.get_chunk(i)

        if nchunks >= 2:
            assert schunk.cratio > 1
            assert schunk.cratio == schunk.nbytes / schunk.cbytes
        assert schunk.nbytes >= nchunks * chunk_len * 4

        blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nbytes, cparams, dparams, nchunks",
    [
        (7, {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 5}, {}, 1),
        (641091, {"typesize": 3}, {"nthreads": 2}, 1),
        (136, {"typesize": 1}, {}, 5),
        (1232, {"typesize": 8}, blosc2.dparams_dflts, 10),
    ],
)
def test_schunk(contiguous, urlpath, nbytes, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    numpy_meta = {b"dtype": str(np.dtype(np.uint8))}
    test_meta = {b"lorem": 1234}
    meta = {"numpy": numpy_meta, "test": test_meta}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=2 * nbytes, meta=meta, **storage)

    assert "numpy" in schunk.meta
    assert "error" not in schunk.meta
    assert schunk.meta["numpy"] == numpy_meta
    assert "test" in schunk.meta
    assert schunk.meta["test"] == test_meta
    test_meta = {b"lorem": 4231}
    schunk.meta["test"] = test_meta
    assert schunk.meta["test"] == test_meta

    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)

    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj

        dest = bytearray(bytes_obj)
        schunk.decompress_chunk(i, dst=dest)
        assert dest == bytes_obj

    for i in range(nchunks):
        schunk.get_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 1),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 10),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_schunk_cframe(contiguous, urlpath, cparams, dparams, nchunks, copy):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 1000 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, data=data, **storage)

    cframe = schunk.to_cframe()
    schunk2 = blosc2.schunk_from_cframe(cframe, copy)
    if not os.getenv("BTUNE_TRADEOFF"):
        for key in cparams:
            if key == "nthreads":
                continue
            assert schunk2.cparams[key] == cparams[key]

    data2 = np.empty(data.shape, dtype=data.dtype)
    schunk2.get_slice(out=data2)
    assert np.array_equal(data, data2)

    cframe = schunk.to_cframe()
    schunk3 = blosc2.schunk_from_cframe(cframe, copy)
    del schunk3
    # Check that we can still access the external cframe buffer
    _ = str(cframe)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    "cparams, dparams, new_cparams, new_dparams",
    [
        (
            {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4},
            {},
            {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4},
            {"nthreads": 4},
        ),
        (
            {"typesize": 4},
            {"nthreads": 4},
            {"codec": blosc2.Codec.ZLIB, "splitmode": blosc2.SplitMode.ALWAYS_SPLIT},
            {"nthreads": 1},
        ),
        (
            {"codec": blosc2.Codec.ZLIB, "splitmode": blosc2.SplitMode.ALWAYS_SPLIT},
            {},
            {
                "splitmode": blosc2.SplitMode.ALWAYS_SPLIT,
                "nthreads": 5,
                "typesize": 4,
                "filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.TRUNC_PREC],
            },
            {"nthreads": 16},
        ),
        (
            {"codec": blosc2.Codec.LZ4HC, "typesize": 4},
            {},
            {"filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.TRUNC_PREC]},
            {"nthreads": 3},
        ),
    ],
)
def test_schunk_cdparams(cparams, dparams, new_cparams, new_dparams):
    storage = {"cparams": cparams, "dparams": dparams}

    chunk_len = 200 * 1000
    schunk = blosc2.SChunk(chunksize=chunk_len * 4, **storage)

    # Check cparams have been set correctly
    for key in cparams:
        assert schunk.cparams[key] == cparams[key]
    for key in dparams:
        assert schunk.dparams[key] == dparams[key]

    schunk.cparams = new_cparams
    schunk.dparams = new_dparams
    for key in schunk.cparams:
        if key in new_cparams:
            if key == "filters":
                assert schunk.cparams[key][: len(new_cparams[key])] == new_cparams[key]
            else:
                assert schunk.cparams[key] == new_cparams[key]
        elif key in cparams:
            if key == "filters":
                assert schunk.cparams[key][: len(cparams[key])] == cparams[key]
            else:
                assert schunk.cparams[key] == cparams[key]
        else:
            if key == "filters":
                assert schunk.cparams[key][: len(blosc2.cparams_dflts[key])] == blosc2.cparams_dflts[key]
            elif key == "filters_meta":
                # Exception for testing bytedelta in the last position
                assert (
                    schunk.cparams[key][: len(blosc2.cparams_dflts[key]) - 1]
                    == blosc2.cparams_dflts[key][:-1]
                )
            else:
                assert schunk.cparams[key] == blosc2.cparams_dflts[key]

    if "nthreads" in new_dparams:
        assert schunk.dparams["nthreads"] == new_dparams["nthreads"]
    elif "nthreads" in dparams:
        assert schunk.dparams["nthreads"] == dparams["nthreads"]
    else:
        assert schunk.dparams["nthreads"] == blosc2.dparams_dflts["nthreads"]
