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
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {"nthreads": 1}, 0),
        ({"typesize": 4}, {"nthreads": 1}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {"nthreads": 1}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {"nthreads": 1}, 10),
    ],
)
def test_iterchunks(contiguous, urlpath, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)

    for i in range(nchunks):
        buffer = i * np.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    dest = np.empty(200 * 1000, np.int32)
    for i, chunk in enumerate(schunk.iterchunks(np.int32)):
        schunk.decompress_chunk(i, dest)
        assert np.array_equal(chunk, dest)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {"nthreads": 1}, 2),
        ({"typesize": 4}, {"nthreads": 1}, 1),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {"nthreads": 1}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {"nthreads": 1}, 3),
    ],
)
def test_iterchunks_pf(contiguous, urlpath, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    chunkshape = 200 * 1000
    data = np.arange(0, nchunks * chunkshape, dtype=np.int32)
    schunk = blosc2.SChunk(chunksize=chunkshape * 4, data=data, **storage)

    @schunk.postfilter(np.int32, np.int32)
    def postf1(input, output, offset):
        output[:] = input - 1

    data -= 1
    for i, chunk in enumerate(schunk.iterchunks(np.int32)):
        assert np.array_equal(chunk, data[i * chunkshape : (i + 1) * chunkshape])

    blosc2.remove_urlpath(urlpath)
