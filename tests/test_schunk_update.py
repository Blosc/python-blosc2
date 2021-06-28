########################################################################
#
#       Created: June 28, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################
import random

import numpy
import pytest

import blosc2
from tests import utilities


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks, nupdates",
    [
        (0, 0),
        (1, 1),
        (7, 3),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("create_chunk", [True, False])
def test_schunk_update_numpy(contiguous, urlpath, nchunks, nupdates, copy, create_chunk):
    storage = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }
    utilities.remove_schunk(contiguous, urlpath)

    schunk = blosc2.SChunk(**storage)
    for i in range(nchunks):
        buffer = i * numpy.arange(200 * 1000)
        nchunks_ = schunk.append_buffer(buffer)
        assert nchunks_ == (i + 1)

    for i in range(nupdates):
        pos = random.randint(0, nchunks - 1)
        buffer = pos * numpy.arange(200 * 1000)
        if create_chunk:
            chunk = blosc2.compress2(buffer)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_buffer(pos, buffer, copy)
        chunk_ = schunk.decompress_chunk(pos)
        bytes_obj = buffer.tobytes()
        assert chunk_ == bytes_obj

        dest = numpy.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(pos, dest)
        assert numpy.array_equal(buffer, dest)

    for i in range(nchunks):
        schunk.decompress_chunk(i)

    utilities.remove_schunk(contiguous, urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks, nupdates",
    [
        (0, 0),
        (1, 1),
        (7, 3),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
@pytest.mark.parametrize("create_chunk", [True, False])
def test_update(contiguous, urlpath, nchunks, nupdates, copy, create_chunk):
    storage = {
        "contiguous": contiguous,
        "urlpath": urlpath,
        "cparams": {"nthreads": 2},
        "dparams": {"nthreads": 2},
    }

    utilities.remove_schunk(contiguous, urlpath)

    schunk = blosc2.SChunk(**storage)
    nbytes = 23401
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_buffer(bytes_obj)
        assert nchunks_ == (i + 1)

    for i in range(nupdates):
        pos = random.randint(0, nchunks - 1)
        bytes_obj = b"i " * nbytes
        if create_chunk:
            chunk = blosc2.compress2(bytes_obj)
            schunk.update_chunk(pos, chunk)
        else:
            schunk.update_buffer(pos, bytes_obj, copy)
        res = schunk.decompress_chunk(pos)
        assert res == bytes_obj

    for i in range(nchunks):
        schunk.decompress_chunk(i)

    utilities.remove_schunk(contiguous, urlpath)
