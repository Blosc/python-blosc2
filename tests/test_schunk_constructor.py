########################################################################
#
#       Created: July 13, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import numpy
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, chunksize",
    [
        ({"compcode": blosc2.LZ4, "clevel": 6}, {}, 40000),
        ({}, {"nthreads": 4}, 20 * 100),
        ({"splitmode": blosc2.ALWAYS_SPLIT, "nthreads": 5}, {"schunk": None}, 500),
        ({"compcode": blosc2.LZ4HC, "typesize": 4}, {}, 100),
    ],
)
def test_schunk_numpy(contiguous, urlpath, cparams, dparams, chunksize):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    num_elem = 20 * 1000
    nchunks = num_elem * 4 // chunksize + 1 if num_elem * 4 % chunksize != 0 else num_elem * 4 // chunksize
    buffer = numpy.arange(num_elem, dtype="int32")
    bytes_obj = buffer.tobytes()
    schunk = blosc2.SChunk(chunksize=chunksize, buffer=buffer, **storage)

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

        dest = numpy.empty(np_end - np_start, dtype=buffer.dtype)
        schunk.decompress_chunk(i, dest)
        assert numpy.array_equal(buffer[np_start:np_end], dest)

        schunk.decompress_chunk(i, memoryview(dest))
        assert numpy.array_equal(buffer[np_start:np_end], dest)

        dest = bytearray(buffer)
        schunk.decompress_chunk(i, dest[start:end])
        assert dest[start:end] == bytes_obj[start:end]

    for i in range(nchunks):
        schunk.get_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    " cparams, dparams, chunksize",
    [
        ({"compcode": blosc2.LZ4, "clevel": 6, "typesize": 1}, {}, 100),
        ({"typesize": 1}, {"nthreads": 4}, 25),
        ({"typesize": 1}, {}, 50),
        ({"typesize": 1}, blosc2.dparams_dflts, 1000),
    ],
)
def test_schunk(contiguous, urlpath, cparams, dparams, chunksize):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

    blosc2.remove_urlpath(urlpath)
    nrep = 1000
    nchunks = 5 * nrep // chunksize + 1 if nrep * 5 % chunksize != 0 else 5 * nrep // chunksize

    buffer = b"1234 " * nrep
    schunk = blosc2.SChunk(chunksize=chunksize, buffer=buffer, **storage)

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

    blosc2.remove_urlpath(urlpath)
