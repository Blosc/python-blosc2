########################################################################
#
#       Created: May 26, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################

import numpy
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"compcode": blosc2.LZ4, "clevel": 6, "typesize": 4}, {}, 0),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        ({"splitmode": blosc2.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {"schunk": None}, 5),
        ({"compcode": blosc2.LZ4HC, "typesize": 4}, {}, 10),
    ],
)
def test_schunk_numpy(contiguous, urlpath, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)
    for i in range(nchunks):
        buffer = i * numpy.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    for i in range(nchunks):
        buffer = i * numpy.arange(200 * 1000, dtype="int32")
        bytes_obj = buffer.tobytes()
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj

        dest = numpy.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(i, dest)
        assert numpy.array_equal(buffer, dest)

        schunk.decompress_chunk(i, memoryview(dest))
        assert numpy.array_equal(buffer, dest)

        dest = bytearray(buffer)
        schunk.decompress_chunk(i, dest)
        assert dest == bytes_obj

    for i in range(nchunks):
        schunk.get_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nbytes, cparams, dparams, nchunks",
    [
        (7, {"compcode": blosc2.LZ4, "clevel": 6, "typesize": 5}, {}, 1),
        (641091, {"typesize": 3}, {"nthreads": 2}, 1),
        (136, {"typesize": 1}, {}, 5),
        (1231, {"typesize": 8}, blosc2.dparams_dflts, 10),
    ],
)
def test_schunk(contiguous, urlpath, nbytes, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=2 * nbytes, **storage)
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
