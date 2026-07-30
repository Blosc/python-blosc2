#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize("mode", ["w", "a"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks", "start", "stop"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 10, 0, 100),
        ({"typesize": 4}, {"nthreads": 4}, 1, 7, 23),
        (
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4},
            {},
            5,
            21,
            200 * 2 * 100,
        ),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 7, None, None),
        ({"blocksize": 200 * 100, "typesize": 4}, {}, 5, -2456, -234),
        ({"blocksize": 200 * 100, "typesize": 4}, {}, 4, 2456, -234),
        ({"blocksize": 100 * 100, "typesize": 4}, {}, 2, -200 * 100 + 234, 40000),
    ],
)
def test_schunk_get_slice(contiguous, urlpath, mode, cparams, dparams, nchunks, start, stop):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 100 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(chunksize=200 * 100 * 4, data=data, mode=mode, **kwargs)

    start_, stop_ = start, stop
    if start is None:
        start_ = 0
    if stop is None:
        stop_ = data.size

    sl = data[start_:stop]
    res = schunk.get_slice(start, stop)
    assert res == sl.tobytes()

    res = schunk[start:stop]
    assert res == sl.tobytes()

    out = np.empty(sl.shape, dtype="int32")
    schunk.get_slice(start, stop, out)
    assert np.array_equal(data[start_:stop_], out)

    schunk.get_slice(start, stop, memoryview(out))
    assert np.array_equal(data[start_:stop_], out)

    out = bytearray(res)
    schunk.get_slice(start, stop, out)
    assert out == bytearray(data)[start_ * 4 : stop_ * 4]

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("cparams", "nchunks", "elem"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, 10, 0),
        ({"typesize": 4}, 1, 7),
        (
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4},
            5,
            21,
        ),
        ({"blocksize": 200 * 100, "typesize": 4}, 5, -1),
        ({"blocksize": 100 * 100, "typesize": 4}, 2, -200 * 100 + 234),
    ],
)
def test_schunk_getitem_int(cparams, nchunks, elem):
    data = np.arange(200 * 100 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(chunksize=200 * 100 * 4, data=data, cparams=cparams)

    sl = data[elem]
    res = schunk[elem]
    assert res == sl.tobytes()


def test_schunk_get_slice_raises():
    kwargs = {"contiguous": True, "urlpath": "schunk.b2frame", "cparams": {"typesize": 4}, "dparams": {}}
    blosc2.remove_urlpath(kwargs["urlpath"])

    nchunks = 2
    data = np.arange(200 * 100 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(chunksize=200 * 100 * 4, data=data, **kwargs)

    start = 200 * 100
    stop = 200 * 100 * nchunks
    with pytest.raises(IndexError):
        schunk[start:stop:2]

    out = np.empty(stop - start - 1, dtype="int32")
    with pytest.raises(ValueError):
        schunk.get_slice(start, stop, out)

    # The next are not raising errors, but returning empty bytes
    start = -1
    stop = -4
    assert schunk[start:stop] == b""

    start = 200 * 100 * nchunks
    stop = start + 4
    assert schunk[start:stop] == b""

    blosc2.remove_urlpath(kwargs["urlpath"])


# ---------------------------------------------------------------------------
# Typesizes above BLOSC_MAX_TYPESIZE (255)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("typesize", [252, 256, 512])
def test_get_slice_wide_typesize_matches_source(typesize):
    """c-blosc2 records a typesize above 255 as 1 in the chunk header, but
    blosc2_schunk_get_slice_buffer() still divides byte offsets by
    schunk->typesize, so a partially covered chunk addressed the wrong range:
    most slices failed outright and single-element ones returned the wrong
    bytes with no error.  An <U64 NDArray is a 256-byte typesize, so this is
    reachable from ordinary string columns.
    """
    n = 500
    data = np.arange(n * typesize, dtype=np.uint8).reshape(n, typesize)
    data[:, 0] = np.arange(n) % 251
    data[:, 1] = (np.arange(n) // 251) % 251
    schunk = blosc2.SChunk(chunksize=100 * typesize, cparams={"typesize": typesize}, data=data.tobytes())

    for start, stop in [(0, 1), (68, 69), (0, 50), (50, 150), (99, 101), (100, 200), (7, 493), (0, n)]:
        got = np.frombuffer(schunk[start:stop], dtype=np.uint8).reshape(stop - start, typesize)
        assert np.array_equal(got, data[start:stop]), (start, stop)


def test_get_slice_wide_typesize_out_buffer():
    typesize = 256
    data = np.arange(20 * typesize, dtype=np.uint8).reshape(20, typesize)
    data[:, 0] = np.arange(20)
    schunk = blosc2.SChunk(chunksize=8 * typesize, cparams={"typesize": typesize}, data=data.tobytes())
    out = bytearray(5 * typesize)
    schunk.get_slice(6, 11, out=out)
    assert np.array_equal(np.frombuffer(bytes(out), dtype=np.uint8).reshape(5, typesize), data[6:11])

    with pytest.raises(ValueError, match="Not enough space"):
        schunk.get_slice(0, 10, out=bytearray(typesize))


def test_ndarray_schunk_slice_wide_string_dtype():
    # <U64 is 256 bytes -- exactly the width the utf8 expression driver buckets to.
    values = np.array(["hello", "help", "world"] * 200, dtype="<U64")
    arr = blosc2.asarray(values)
    assert arr.schunk.typesize == 256
    assert list(np.frombuffer(arr.schunk[1:4], dtype="<U64")) == list(values[1:4])
