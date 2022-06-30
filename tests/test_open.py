########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################
import random

import numpy
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", ["schunk.b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks, chunk_nitems, dtype",
    [
        ({"compcode": blosc2.Codec.LZ4, "clevel": 6, "typesize": 2}, {}, 0, 50, numpy.int16),
        ({"typesize": 4}, {"nthreads": 4}, 1, 200 * 100, float),
        (
         {"splitmode": blosc2.ALWAYS_SPLIT, "nthreads": 2, "typesize": 1},
         {"schunk": None},
         5,
         201,
         numpy.int8
         ),
        ({"compcode": blosc2.Codec.LZ4HC, "typesize": 8}, {}, 10, 30 * 100, numpy.int64),
    ],
)
@pytest.mark.parametrize("mode", ["w", "r", "a"])
def test_open(contiguous, urlpath, cparams, dparams, nchunks, chunk_nitems, dtype, mode):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    dtype = numpy.dtype(dtype)
    schunk = blosc2.SChunk(chunksize=chunk_nitems * dtype.itemsize, **storage)
    for i in range(nchunks):
        buffer = i * numpy.arange(chunk_nitems, dtype=dtype)
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    del schunk
    schunk_open = blosc2.open(urlpath, mode)

    buffer = numpy.zeros(chunk_nitems, dtype=dtype)
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
        buffer = i * numpy.arange(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(i)
        assert res == bytes_obj
    if mode != "r":
        buffer = numpy.zeros(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(pos)
        assert res == bytes_obj
        if mode == "a":
            for i in range(pos + 1, nchunks + 1):
                buffer = (i - 1) * numpy.arange(chunk_nitems, dtype=dtype)
                dest = numpy.empty(buffer.shape, buffer.dtype)
                schunk_open.decompress_chunk(i, dest)
                assert numpy.array_equal(buffer, dest)

    blosc2.remove_urlpath(urlpath)
