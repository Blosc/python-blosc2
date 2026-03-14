#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import random

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("nchunks", "nupdates"),
    [
        (0, 0),
        (1, 1),
        (7, 3),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("create_chunk", [True, False])
def test_schunk_update_numpy(contiguous, urlpath, nchunks, nupdates, copy, create_chunk, gil):
    blosc2.set_releasegil(gil)
    kwargs = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **kwargs)
    for i in range(nchunks):
        buffer = i * np.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    for _ in range(nupdates):
        pos = random.randint(0, nchunks - 1)
        buffer = pos * np.arange(200 * 1000, dtype="int32")
        if create_chunk:
            chunk = blosc2.compress2(buffer)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_data(pos, buffer, copy)
        chunk_ = schunk.decompress_chunk(pos)
        bytes_obj = buffer.tobytes()
        assert chunk_ == bytes_obj

        dest = np.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(pos, dest)
        assert np.array_equal(buffer, dest)

    for i in range(nchunks):
        schunk.decompress_chunk(i)
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("gil", [True, False])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("nchunks", "nupdates"),
    [
        (0, 0),
        (1, 1),
        (7, 3),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("create_chunk", [True, False])
def test_update(contiguous, urlpath, nchunks, nupdates, copy, create_chunk, gil):
    blosc2.set_releasegil(gil)
    kwargs = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }

    blosc2.remove_urlpath(urlpath)
    nbytes = 23401

    schunk = blosc2.SChunk(chunksize=nbytes * 2, **kwargs)
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)

    for _ in range(nupdates):
        pos = random.randint(0, nchunks - 1)
        bytes_obj = b"i " * nbytes
        if create_chunk:
            chunk = blosc2.compress2(bytes_obj, typesize=1)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_data(pos, bytes_obj, copy)
        res = schunk.decompress_chunk(pos)
        assert res == bytes_obj

    for i in range(nchunks):
        schunk.decompress_chunk(i)
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_variable_append_chunk.b2frame"),
        (False, "test_variable_append_chunk_s.b2frame"),
    ],
)
def test_append_chunk_variable_sizes(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=-1, contiguous=contiguous, urlpath=urlpath, cparams={"typesize": 1})
    payloads = [b"a" * 13, b"b" * 29, b"c" * 41]

    for i, payload in enumerate(payloads, start=1):
        chunk = blosc2.compress2(payload, typesize=1)
        assert schunk.append_chunk(chunk) == i
        assert schunk.decompress_chunk(i - 1) == payload

    assert schunk.chunksize == 0

    replacement = b"z" * 17
    schunk.update_chunk(1, blosc2.compress2(replacement, typesize=1))
    expected = [payloads[0], replacement, payloads[2]]
    assert [schunk.decompress_chunk(i) for i in range(schunk.nchunks)] == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert reopened.chunksize == 0
        assert [reopened.decompress_chunk(i) for i in range(reopened.nchunks)] == expected

    blosc2.remove_urlpath(urlpath)
