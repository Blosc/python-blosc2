#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import random

import pytest

import blosc2
import numpy as np


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", ["schunk.b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks, chunk_nitems, dtype",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 2}, {}, 0, 50, np.int16),
        ({"typesize": 4}, {"nthreads": 4}, 1, 200 * 100, float),
        (
         {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 2, "typesize": 1},
         {},
         5,
         201,
         np.int8
         ),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 8}, {}, 10, 30 * 100, np.int64),
    ],
)
@pytest.mark.parametrize("mode", ["w", "r", "a"])
def test_open(contiguous, urlpath, cparams, dparams, nchunks, chunk_nitems, dtype, mode):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    dtype = np.dtype(dtype)
    schunk = blosc2.SChunk(chunksize=chunk_nitems * dtype.itemsize, **storage)
    for i in range(nchunks):
        buffer = i * np.arange(chunk_nitems, dtype=dtype)
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    del schunk
    schunk_open = blosc2.open(urlpath, mode)

    for key in cparams:
        if key == "nthreads":
            continue
        assert schunk_open.cparams[key] == cparams[key]

    buffer = np.zeros(chunk_nitems, dtype=dtype)
    if mode != "r":
        if mode == "w":
            pos = 0
        else:
            pos = random.randint(0, nchunks)
        nchunks_ = schunk_open.insert_data(nchunk=pos, data=buffer, copy=True)
        assert nchunks_ == 1 if mode == "w" else nchunks + 1
    else:
        pos = nchunks
        with pytest.raises(ValueError):
            schunk_open.insert_data(nchunk=pos, data=buffer, copy=True)

    for i in range(pos):
        buffer = i * np.arange(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(i)
        assert res == bytes_obj
    if mode != "r":
        buffer = np.zeros(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(pos)
        assert res == bytes_obj
        if mode == "a":
            for i in range(pos + 1, nchunks + 1):
                buffer = (i - 1) * np.arange(chunk_nitems, dtype=dtype)
                dest = np.empty(buffer.shape, buffer.dtype)
                schunk_open.decompress_chunk(i, dest)
                assert np.array_equal(buffer, dest)

    blosc2.remove_urlpath(urlpath)
