########################################################################
#
#       Created: June 22, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################
import os
import random

import numpy
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks",
    [
        0,
        1,
        7,
        10,
    ],
)
@pytest.mark.parametrize("ninserts", [0, 1, 17])
def test_schunk_insert_numpy(contiguous, urlpath, nchunks, ninserts):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": {}, "dparams": {}}
    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])

    schunk = blosc2.SChunk(**storage)
    for i in range(nchunks):
        buffer = i * numpy.arange(200 * 1000)
        nchunks_ = schunk.append_buffer(buffer)
        assert nchunks_ == (i + 1)

    for i in range(ninserts):
        pos = random.randint(0, nchunks + i)
        buffer = pos * numpy.arange(200 * 1000)
        chunk = blosc2.compress2(buffer)
        schunk.insert_chunk(pos, chunk)

        chunk_ = schunk.decompress_chunk(pos)
        bytes_obj = buffer.tobytes()
        assert chunk_ == bytes_obj

        dest = numpy.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(pos, dest)
        assert numpy.array_equal(buffer, dest)

        schunk.decompress_chunk(pos, memoryview(dest))
        assert numpy.array_equal(buffer, dest)

        dest = bytearray(buffer)
        schunk.decompress_chunk(pos, dest)
        assert dest == bytes_obj

    for i in range(nchunks + ninserts):
        schunk.decompress_chunk(i)

    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nchunks",
    [
        0,
        1,
        7,
        10,
    ],
)
@pytest.mark.parametrize("ninserts", [0, 1, 17])
def test_insert_operations(contiguous, urlpath, nchunks, ninserts):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": {}, "dparams": {}}

    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])

    schunk = blosc2.SChunk(**storage)
    nbytes = 23401
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_buffer(bytes_obj)
        assert nchunks_ == (i + 1)

    for i in range(ninserts):
        pos = random.randint(0, nchunks + i)
        bytes_obj = b"i " * nbytes
        chunk = blosc2.compress2(bytes_obj)
        schunk.insert_chunk(pos, chunk)

        res = schunk.decompress_chunk(pos)
        assert res == bytes_obj

        dest = bytearray(bytes_obj)
        schunk.decompress_chunk(pos, dst=dest)
        assert dest == bytes_obj

    for i in range(nchunks + ninserts):
        schunk.decompress_chunk(i)

    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])
