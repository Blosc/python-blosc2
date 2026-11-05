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
    "contiguous",
    [
        True,
        False,
    ],
)
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "urlpath", "dtype"),
    [
        ([634], [156], [33], "test00.b2nd", np.float64),
        ([20, 134, 13], [7, 22, 5], [3, 5, 3], "test01.b2nd", np.int32),
        ([12, 13, 14, 15, 16], [4, 6, 4, 7, 5], [2, 4, 2, 3, 3], "test02.b2nd", np.float32),
    ],
)
def test_persistency(shape, chunks, blocks, urlpath, contiguous, dtype):
    blosc2.remove_urlpath(urlpath)

    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    _ = blosc2.asarray(nparray, chunks=chunks, blocks=blocks, urlpath=urlpath, contiguous=contiguous)
    b = blosc2.open(urlpath, mode="r")

    bc = b[:]

    nparray2 = np.asarray(bc).view(dtype)
    np.testing.assert_almost_equal(nparray, nparray2)

    blosc2.remove_urlpath(urlpath)


def _layout(cparams):
    return cparams.typesize, cparams.blocksize, cparams.codec, cparams.clevel, cparams.filters


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize(
    "cparams",
    [
        {"nthreads": 2},
        {"clevel": 1},
        {"nthreads": 2, "typesize": 4},
        {"codec": blosc2.Codec.LZ4, "filters": [blosc2.Filter.BITSHUFFLE]},
        blosc2.CParams(nthreads=2),
    ],
)
def test_open_cparams_keep_layout(tmp_path, contiguous, cparams):
    # open(cparams=...) changes the given parameters only; typesize and blocksize stay those of the array,
    # so that writes produce chunks that fit the array's blocks (they used to fail, or to corrupt the file)
    urlpath = tmp_path / "a.b2nd"
    x = np.arange(48 * 40 * 24, dtype=np.float32).reshape(48, 40, 24)
    blosc2.asarray(
        x,
        urlpath=urlpath,
        contiguous=contiguous,
        chunks=(24, 20, 24),
        blocks=(8, 10, 12),
        cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 5, "filters": [blosc2.Filter.SHUFFLE]},
    )
    stored = blosc2.open(urlpath).cparams
    a = blosc2.open(urlpath, mode="a", cparams=cparams)
    # A CParams instance gives every field but typesize and blocksize
    given = cparams
    if isinstance(cparams, blosc2.CParams):
        given = {key: getattr(cparams, key) for key in ("nthreads", "codec", "clevel", "filters")}
    expected = {"typesize": stored.typesize, "blocksize": stored.blocksize, "codec": stored.codec}
    expected |= {"clevel": stored.clevel, "filters": stored.filters} | given
    if blosc2.IS_WASM:
        expected["nthreads"] = 1
    for key, value in expected.items():
        got = getattr(a.cparams, key)
        assert (got[: len(value)] if key == "filters" else got) == value

    y = x[::-1].copy()
    a[:] = y
    a[1:5, 3:7, 2:9] = -1
    y[1:5, 3:7, 2:9] = -1
    a.resize((60, 40, 24))
    a[48:] = 7
    a.schunk.update_data(0, np.full(24 * 20 * 24, 5, np.float32), copy=True)
    del a
    y = np.concatenate([y, np.full((12, 40, 24), 7, np.float32)])
    y[:24, :20, :24] = 5
    np.testing.assert_array_equal(blosc2.open(urlpath)[:], y)


@pytest.mark.parametrize(
    "cparams",
    [{"typesize": 8}, {"blocksize": 1024}, blosc2.CParams(typesize=2), blosc2.CParams(blocksize=4096)],
)
def test_open_cparams_change_layout(tmp_path, cparams):
    urlpath = tmp_path / "a.b2nd"
    blosc2.asarray(np.arange(1000, dtype=np.float32), urlpath=urlpath, chunks=(500,), blocks=(100,))
    with pytest.raises(ValueError, match="Cannot change the"):
        blosc2.open(urlpath, mode="a", cparams=cparams)
    # 0 (automatic) leaves the blocksize as it is
    assert blosc2.open(urlpath, mode="a", cparams={"blocksize": 0}).cparams.blocksize == 400


def test_open_cparams_read_mode(tmp_path):
    urlpath = tmp_path / "a.b2nd"
    blosc2.asarray(
        np.arange(1000, dtype=np.int16), urlpath=urlpath, cparams={"codec": blosc2.Codec.LZ4, "clevel": 3}
    )
    stored = blosc2.open(urlpath).cparams
    a = blosc2.open(urlpath, mode="r", cparams={"nthreads": 1})
    assert a.cparams.nthreads == 1
    assert _layout(a.cparams) == _layout(stored)


def test_open_cparams_filters_meta(tmp_path):
    # Without filters_meta, new filters get zero metas, as in CParams(filters=...): the stored metas
    # belong to the stored filters
    urlpath = tmp_path / "a.b2nd"
    x = np.arange(1000, dtype=np.float32)
    filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
    blosc2.asarray(x, urlpath=urlpath, cparams={"filters": filters, "filters_meta": [23, 0]})
    assert blosc2.open(urlpath, mode="a", cparams={"nthreads": 1}).cparams.filters_meta[:2] == [23, 0]
    a = blosc2.open(urlpath, mode="a", cparams={"filters": [blosc2.Filter.BITSHUFFLE]})
    assert a.cparams.filters[0] == blosc2.Filter.BITSHUFFLE
    assert a.cparams.filters_meta[0] == 0
    a[:] = x[::-1].copy()
    np.testing.assert_array_equal(blosc2.open(urlpath)[:], x[::-1])
