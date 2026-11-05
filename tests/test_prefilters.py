#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import asdict, replace

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("func", "op_dtype", "op2_dtype", "schunk_dtype", "offset"),
    [
        ("fill_f1", np.dtype(np.int32), None, np.dtype(np.int64), 0),
        ("fill_f1", np.dtype(np.int32), None, np.dtype(np.float32), 0),
        ("fill_f1", np.dtype(np.complex128), None, np.dtype(np.complex128), 0),
        ("fill_f2", np.dtype(np.float64), np.dtype(np.int32), np.dtype(np.float64), None),
        ("fill_f3", np.dtype("M8[D]"), None, np.dtype(np.bool_), None),
        ("fill_f4", np.dtype(np.float32), np.dtype(np.int32), np.dtype(np.float64), None),
    ],
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks", "contiguous", "urlpath", "nelem"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "nthreads": 1}, {"nthreads": 4}, 2, True, None, None),
        ({"nthreads": 1}, {"nthreads": 2}, 1, True, "test_fillers.b2frame", 1 * 20_000),
        (
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 1},
            {"nthreads": 4},
            5,
            False,
            None,
            5 * 20_000,
        ),
        (
            {"codec": blosc2.Codec.LZ4HC, "nthreads": 1},
            {"nthreads": 1},
            3,
            False,
            "test_fillers.b2frame",
            None,
        ),
    ],
)
def test_fillers(
    contiguous, urlpath, cparams, dparams, nchunks, nelem, func, op_dtype, op2_dtype, schunk_dtype, offset
):
    blosc2.remove_urlpath(urlpath)

    chunk_len = 20_000
    cparams["typesize"] = schunk_dtype.itemsize

    schunk = blosc2.SChunk(
        chunksize=chunk_len * schunk_dtype.itemsize,
        contiguous=contiguous,
        urlpath=urlpath,
        cparams=cparams,
        dparams=dparams,
    )

    data = np.arange(0, chunk_len * nchunks, dtype=op_dtype)
    schunk_op = blosc2.SChunk(
        chunksize=chunk_len * op_dtype.itemsize, data=data, cparams={"typesize": op_dtype.itemsize}
    )
    res = np.empty(chunk_len * nchunks, dtype=schunk_dtype)
    if func == "fill_f1":

        @schunk.filler(((schunk_op, op_dtype),), schunk_dtype, nelem)
        def fill_f1(inputs_tuple, output, offset):
            for i in range(output.size):
                output[i] = offset + i

        fill_f1((data,), res, offset)

    elif func == "fill_f2":
        data2 = np.full(chunk_len * nchunks, 3, dtype=op2_dtype)
        schunk_op2 = blosc2.SChunk(
            chunksize=chunk_len * op2_dtype.itemsize, data=data2, cparams={"typesize": op2_dtype.itemsize}
        )

        @schunk.filler(((schunk_op, op_dtype), (schunk_op2, op2_dtype)), schunk_dtype, nelem)
        def fill_f2(inputs_tuple, output, offset):
            output[:] = inputs_tuple[0] * inputs_tuple[1]

        fill_f2((data, data2), res, offset)

    elif func == "fill_f3":

        @schunk.filler(((schunk_op, op_dtype),), schunk_dtype, nelem)
        def fill_f3(inputs_tuple, output, offset):
            output[:] = inputs_tuple[0] <= np.datetime64("1997-12-31")

        fill_f3((data,), res, offset)
    else:
        data2 = np.full(chunk_len * nchunks, 3, dtype=op2_dtype)

        @schunk.filler(((schunk_op, op_dtype), (data2, op2_dtype), (np.pi, np.float32)), schunk_dtype, nelem)
        def fill_f4(inputs_tuple, output, offset):
            output[:] = inputs_tuple[0] - inputs_tuple[1] * inputs_tuple[2]

        fill_f4((data, data2, np.pi), res, offset)

    new_cparams = replace(schunk.cparams, nthreads=2)
    schunk.cparams = new_cparams

    pre_data = np.empty(chunk_len * nchunks, dtype=schunk_dtype)
    schunk.get_slice(0, chunk_len * nchunks, out=pre_data)

    if "f" in schunk_dtype.str:
        assert np.allclose(pre_data, res)
    else:
        assert np.array_equal(pre_data, res)

    # Update a chunk
    chunk = np.full(chunk_len, 4, dtype=schunk_dtype)
    schunk[0:chunk_len] = chunk
    sl = np.empty(chunk_len, dtype=schunk_dtype)
    schunk.get_slice(0, chunk_len, sl)
    if "f" in schunk_dtype.str:
        assert np.allclose(chunk, sl)
    else:
        assert np.array_equal(chunk, sl)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("func", "data_dtype", "schunk_dtype", "offset"),
    [
        ("pref1", np.dtype(np.int32), None, 0),
        ("pref1", np.dtype(np.int32), np.dtype(np.float32), 0),
        ("pref2", np.dtype(np.complex128), None, 0),
        ("pref2", np.dtype(np.float64), None, None),
        ("pref3", np.dtype("M8[D]"), np.dtype(np.int64), None),
    ],
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks", "contiguous", "urlpath"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "nthreads": 1}, {}, 2, True, None),
        ({"nthreads": 1}, {"nthreads": 2}, 1, True, "test_prefilters.b2frame"),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 1}, {"nthreads": 4}, 5, False, None),
        ({"codec": blosc2.Codec.LZ4HC, "nthreads": 1}, {"nthreads": 4}, 3, False, "test_prefilters.b2frame"),
    ],
)
def test_prefilters(contiguous, urlpath, cparams, dparams, nchunks, func, data_dtype, schunk_dtype, offset):
    blosc2.remove_urlpath(urlpath)

    schunk_dtype = data_dtype if schunk_dtype is None else schunk_dtype
    chunk_len = 2_000
    data = np.arange(0, chunk_len * nchunks, dtype=data_dtype)
    cparams["typesize"] = schunk_dtype.itemsize
    schunk = blosc2.SChunk(
        chunksize=chunk_len * schunk_dtype.itemsize,
        contiguous=contiguous,
        urlpath=urlpath,
        cparams=cparams,
        dparams=dparams,
    )
    if func == "pref1":

        @schunk.prefilter(data_dtype, schunk_dtype)
        def pref1(input, output, offset):
            for i in range(input.size):
                output[i] = offset + i

    elif func == "pref2":

        @schunk.prefilter(data_dtype, schunk_dtype)
        def pref2(input, output, offset):
            output[:] = input - np.pi

    else:

        @schunk.prefilter(data_dtype, schunk_dtype)
        def pref3(input, output, offset):
            output[:] = input <= np.datetime64("1997-12-31")

    new_cparams = asdict(schunk.cparams)
    new_cparams["nthreads"] = 1
    schunk.cparams = blosc2.CParams(**new_cparams)

    schunk[: nchunks * chunk_len] = data
    post_data = np.empty(chunk_len * nchunks, dtype=schunk_dtype)
    schunk.get_slice(0, chunk_len * nchunks, out=post_data)

    res = np.empty(chunk_len * nchunks, dtype=schunk_dtype)
    locals()[func](data, res, offset)
    if "f" in data_dtype.str:
        assert np.allclose(post_data, res)
    else:
        assert np.array_equal(post_data, res)

    schunk.remove_prefilter(func)
    new_data = np.full(chunk_len, 5, dtype=schunk_dtype)
    schunk[:chunk_len] = new_data
    res = np.empty(chunk_len, dtype=schunk_dtype)
    schunk.get_slice(0, chunk_len, res)
    if "f" in data_dtype.str:
        assert np.allclose(new_data, res)
    else:
        assert np.array_equal(new_data, res)

    blosc2.remove_urlpath(urlpath)


# The prefilters below only record their calls and never touch their arrays, so a call with
# arrays that do not fit the buffers cannot overrun them, with or without the typesize guard.
# Each test runs with the GIL kept and released around the C calls.


@pytest.fixture(params=[False, True], ids=["gil-held", "gil-released"])
def releasegil(request):
    old = blosc2.set_releasegil(request.param)
    yield
    blosc2.set_releasegil(old)


@pytest.mark.parametrize("typesize", [1, 8])
def test_prefilter_wrong_typesize(releasegil, typesize):
    # int32 arrays of size // typesize items would overrun a typesize 1 buffer, and cover only
    # half of a typesize 8 one: the prefilter refuses such chunks
    data = np.arange(1000, dtype=np.int32)
    schunk = blosc2.SChunk(chunksize=data.nbytes, cparams={"typesize": typesize, "nthreads": 1})
    calls = []

    @schunk.prefilter(np.int32)
    def pref_wrong_typesize(input, output, offset):
        calls.append(input.size)

    with pytest.raises(
        ValueError, match=f"prefilter: data typesize {typesize} does not match dtype itemsize 4"
    ):
        schunk.append_data(data)
    assert calls == []
    assert schunk.nchunks == 0

    schunk.remove_prefilter("pref_wrong_typesize")
    schunk.append_data(data)
    assert np.array_equal(np.frombuffer(schunk.decompress_chunk(0), dtype=np.int32), data)


@pytest.mark.parametrize("typesize", [1, 8])
def test_filler_wrong_typesize(releasegil, typesize):
    # Same for a filler, which then must not stay set: it keeps only the id of its inputs
    data = np.arange(1000, dtype=np.int32)
    schunk = blosc2.SChunk(chunksize=data.nbytes, cparams={"typesize": typesize, "nthreads": 1})
    calls = []

    with pytest.raises(
        ValueError, match=f"filler: data typesize {typesize} does not match dtype itemsize 4"
    ):

        @schunk.filler(((data, np.int32),), np.int32)
        def fill_wrong_typesize(inputs_tuple, output, offset):
            calls.append(output.size)

    assert calls == []
    assert schunk.nchunks == 0
    assert "fill_wrong_typesize" not in blosc2.prefilter_funcs
    schunk.append_data(data)
    assert schunk.nchunks == 1
