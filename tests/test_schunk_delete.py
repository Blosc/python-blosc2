#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import random

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks, ndeletes",
    [
        (0, 0),
        (1, 1),
        (10, 3),
        (15, 15),
    ],
)
def test_schunk_delete_numpy(contiguous, urlpath, nchunks, ndeletes):
    storage = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)
    for i in range(nchunks):
        buffer = i * np.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    for _ in range(ndeletes):
        pos = random.randint(0, nchunks - 1)
        if pos != (nchunks - 1):
            buff = schunk.decompress_chunk(pos + 1)
        nchunks_ = schunk.delete_chunk(pos)
        assert nchunks_ == (nchunks - 1)
        if pos != (nchunks - 1):
            buff_ = schunk.decompress_chunk(pos)
            assert buff == buff_
        nchunks -= 1

    for i in range(nchunks):
        schunk.decompress_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks, ndeletes",
    [
        (0, 0),
        (1, 1),
        (10, 3),
        (15, 15),
    ],
)
def test_schunk_delete(contiguous, urlpath, nchunks, ndeletes):
    storage = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }
    blosc2.remove_urlpath(urlpath)
    nbytes = 23401

    schunk = blosc2.SChunk(chunksize=nbytes * 2, **storage)
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)

    for _ in range(ndeletes):
        pos = random.randint(0, nchunks - 1)
        if pos != (nchunks - 1):
            buff = schunk.decompress_chunk(pos + 1)
        nchunks_ = schunk.delete_chunk(pos)
        assert nchunks_ == (nchunks - 1)
        if pos != (nchunks - 1):
            buff_ = schunk.decompress_chunk(pos)
            assert buff == buff_
        nchunks -= 1

    for i in range(nchunks):
        schunk.decompress_chunk(i)

    blosc2.remove_urlpath(urlpath)
