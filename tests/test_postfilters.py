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


@pytest.mark.parametrize(
    "func, input_dtype, output_dtype, offset",
    [
        ("postf1", np.dtype(np.int32), None, 0),
        ("postf1", np.dtype(np.int32), np.dtype(np.float32), 0),
        ("postf2", np.dtype(np.complex128), None, 0),
        ("postf2", np.dtype(np.float64), None, None),
        ("postf3", np.dtype("M8[D]"), np.dtype(np.int64), None),
    ],
)
@pytest.mark.parametrize(
    "cparams, dparams, nchunks, contiguous, urlpath",
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

    schunk.dparams = {"nthreads": 1}
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
