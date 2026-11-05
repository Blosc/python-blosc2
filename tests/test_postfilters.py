#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("func", "input_dtype", "output_dtype", "offset"),
    [
        ("postf1", np.dtype(np.int32), None, 0),
        ("postf1", np.dtype(np.int32), np.dtype(np.float32), 0),
        ("postf2", np.dtype(np.complex128), None, 0),
        ("postf2", np.dtype(np.float64), None, None),
        ("postf3", np.dtype("M8[D]"), np.dtype(np.int64), None),
    ],
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks", "contiguous", "urlpath"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6}, {"nthreads": 1}, 2, True, None),
        ({}, {"nthreads": 1}, 1, True, "test_postfilters.b2frame"),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 4}, {"nthreads": 1}, 5, False, None),
        ({"codec": blosc2.Codec.LZ4HC}, {"nthreads": 1}, 3, False, "test_postfilters.b2frame"),
    ],
)
def test_postfilters(
    contiguous, urlpath, cparams, dparams, nchunks, func, input_dtype, output_dtype, offset
):
    blosc2.remove_urlpath(urlpath)

    output_dtype = input_dtype if output_dtype is None else output_dtype
    chunk_len = 2_000
    data = np.arange(0, chunk_len * nchunks, dtype=input_dtype)
    schunk = blosc2.SChunk(
        chunksize=chunk_len * input_dtype.itemsize,
        data=data,
        contiguous=contiguous,
        urlpath=urlpath,
        cparams=cparams,
        dparams=dparams,
    )
    assert schunk.typesize == input_dtype.itemsize
    if func == "postf1":

        @schunk.postfilter(input_dtype, output_dtype)
        def postf1(input, output, offset):
            for i in range(input.size):
                output[i] = offset + i

    elif func == "postf2":

        @schunk.postfilter(input_dtype, output_dtype)
        def postf2(input, output, offset):
            output[:] = input - np.pi

    else:

        @schunk.postfilter(input_dtype, output_dtype)
        def postf3(input, output, offset):
            output[:] = input <= np.datetime64("1997-12-31")

    schunk.dparams = blosc2.DParams(nthreads=1)
    post_data = np.empty(chunk_len * nchunks, dtype=output_dtype)
    schunk.get_slice(0, chunk_len * nchunks, out=post_data)

    res = np.empty(chunk_len * nchunks, dtype=output_dtype)
    locals()[func](data, res, offset)
    if "f" in input_dtype.str:
        assert np.allclose(post_data, res)
    else:
        assert np.array_equal(post_data, res)

    schunk.remove_postfilter(func)
    res = np.empty(chunk_len * nchunks, dtype=input_dtype)
    schunk.get_slice(out=res)
    if "f" in input_dtype.str:
        assert np.allclose(data, res)
    else:
        assert np.array_equal(data, res)

    blosc2.remove_urlpath(urlpath)


# The postfilters below only record their calls and never touch their arrays, so a call with
# arrays that do not fit the buffers cannot overrun them, with or without the typesize guard.
# Each test runs with the GIL kept and released around the C calls.


@pytest.fixture(params=[False, True], ids=["gil-held", "gil-released"])
def releasegil(request):
    old = blosc2.set_releasegil(request.param)
    yield
    blosc2.set_releasegil(old)


@pytest.mark.parametrize("typesize", [1, 8])
def test_postfilter_wrong_typesize(releasegil, typesize):
    # int32 arrays of size // typesize items would overrun a typesize 1 buffer, and cover only
    # half of a typesize 8 one: the postfilter refuses such chunks
    data = np.arange(1000, dtype=np.int32)
    schunk = blosc2.SChunk(
        chunksize=data.nbytes,
        data=data,
        cparams={"typesize": typesize, "nthreads": 1},
        dparams={"nthreads": 1},
    )
    calls = []

    @schunk.postfilter(np.int32)
    def postf_wrong_typesize(input, output, offset):
        calls.append(input.size)

    with pytest.raises(
        ValueError, match=f"postfilter: data typesize {typesize} does not match dtype itemsize 4"
    ):
        schunk.decompress_chunk(0)
    assert calls == []

    schunk.remove_postfilter("postf_wrong_typesize")
    assert np.array_equal(np.frombuffer(schunk.decompress_chunk(0), dtype=np.int32), data)


def test_postfilter_vlmeta(releasegil):
    # The postfilter is for chunk data, not for the msgpack content (typesize 1) of a vlmetalayer
    data = np.arange(2000, dtype=np.int32)
    schunk = blosc2.SChunk(
        chunksize=4000,
        data=data,
        contiguous=False,
        cparams={"typesize": 4, "nthreads": 1},
        dparams={"nthreads": 1},
    )
    schunk.vlmeta["info"] = {"answer": 42}
    calls = []

    @schunk.postfilter(np.int32)
    def postf_vlmeta(input, output, offset):
        calls.append(input.size)

    # C-Blosc2 before Blosc/c-blosc2#857 decompresses vlmetalayers with the postfilter, which then
    # refuses them; with it the content is read without the postfilter.  Either way the callback
    # never sees the content.
    refused = "postfilter: data typesize 1 does not match dtype itemsize 4"
    try:
        info = schunk.vlmeta["info"]
    except ValueError as e:
        info = str(e)
    assert info in ({"answer": 42}, refused)
    assert calls == []

    # Serializing an in-memory sparse super-chunk copies its vlmetalayers the same way
    try:
        copy = blosc2.schunk_from_cframe(schunk.to_cframe(), copy=True)
        info = copy.vlmeta["info"]
    except ValueError as e:
        info = str(e)
    assert info in ({"answer": 42}, refused)

    schunk.remove_postfilter("postf_vlmeta")
    assert schunk.vlmeta["info"] == {"answer": 42}
