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
    "nchunks, ndeletes",
    [
        (0, 0),
        (1, 1),
        (10, 3),
        (15, 15),
    ],
)
def test_schunk_delete_numpy(contiguous, urlpath, nchunks, ndeletes):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": {"nthreads": 2}, "dparams": {"nthreads": 2}}
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

    for i in range(ndeletes):
        pos = random.randint(0, nchunks - 1)
        if pos != (nchunks - 1):
            buff = schunk.decompress_chunk(pos+1)
        nchunks_ = schunk.delete_chunk(pos)
        assert nchunks_ == (nchunks - 1)
        if pos != (nchunks - 1):
            buff_ = schunk.decompress_chunk(pos)
            assert buff == buff_
        nchunks -= 1

    for i in range(nchunks):
        schunk.decompress_chunk(i)


    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])


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

    for i in range(ndeletes):
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

    if urlpath is not None:
        if not contiguous:
            blosc2.remove_dir(storage["urlpath"])
        elif os.path.exists(storage["urlpath"]):
            os.remove(storage["urlpath"])
