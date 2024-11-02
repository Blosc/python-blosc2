#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pathlib

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("mode", "mmap_mode"), [("r", None), ("w", None), ("a", None), ("r", "r"), ("w", "w+")]
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4}, {}, 5),
    ],
)
def test_schunk_pathlib(mode, mmap_mode, cparams, dparams, nchunks):
    urlpath = pathlib.Path("b2frame")
    kwargs = {"urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    if mode != "r":
        chunk_len = 200 * 1000
        schunk = blosc2.SChunk(chunksize=chunk_len * 4, mode=mode, mmap_mode=mmap_mode, **kwargs)
        assert schunk.urlpath == str(urlpath)

        for i in range(nchunks):
            buffer = i * np.arange(chunk_len, dtype="int32")
            nchunks_ = schunk.append_data(buffer)
            assert nchunks_ == (i + 1)

        for i in range(nchunks):
            buffer = i * np.arange(chunk_len, dtype="int32")
            dest = np.empty(buffer.shape, buffer.dtype)
            schunk.decompress_chunk(i, dest)
            assert np.array_equal(buffer, dest)

    blosc2.remove_urlpath(urlpath)


argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
]


@pytest.mark.parametrize(("mode", "mmap_mode"), [("w", None), (None, "w+")])
@pytest.mark.parametrize(argnames, argvalues)
def test_ndarray_pathlib(tmp_path, mode, mmap_mode, shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(
        nparray, chunks=chunks, blocks=blocks, urlpath=tmp_path / "test.b2nd", mode=mode, mmap_mode=mmap_mode
    )
    b = a.slice(slices)
    np_slice = a[slices]
    assert b.shape == np_slice.shape
    np.testing.assert_almost_equal(b[...], np_slice)

    b = blosc2.open(
        tmp_path / "test.b2nd",
        mode="a" if mmap_mode is None else None,
        mmap_mode="r+" if mode is None else None,
    )
    np.testing.assert_almost_equal(b[...], nparray)

    a = blosc2.zeros(shape, dtype, urlpath=tmp_path / "test2.b2nd", mode=mode, mmap_mode=mmap_mode)
    b = np.zeros(shape, dtype)
    np.testing.assert_almost_equal(b[...], a[...])

    a = blosc2.full(shape, 3, urlpath=tmp_path / "test3.b2nd", mode=mode, mmap_mode=mmap_mode)
    b = np.full(shape, 3)
    np.testing.assert_almost_equal(b[...], a[...])

    a = blosc2.frombuffer(
        bytes(nparray), shape, dtype, urlpath=tmp_path / "test4.b2nd", mode=mode, mmap_mode=mmap_mode
    )
    np.testing.assert_almost_equal(nparray[...], a[...])
