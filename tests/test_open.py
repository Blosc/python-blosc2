#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import dataclasses
import os
import random

import numpy as np
import pytest

import blosc2
from conftest import expected_nthreads


@pytest.mark.parametrize("urlpath", ["schunk.b2frame"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks", "chunk_nitems", "dtype"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 2}, {}, 0, 50, np.int16),
        ({"typesize": 4}, {"nthreads": 4}, 1, 200 * 100, float),
        ({"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 2, "typesize": 1}, {}, 5, 201, np.int8),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 8}, {}, 10, 30 * 100, np.int64),
    ],
)
@pytest.mark.parametrize(
    ("contiguous", "mode", "mmap_mode"),
    [
        (False, "w", None),
        (False, "r", None),
        (False, "a", None),
        (True, "w", None),
        (True, "r", None),
        (True, "a", None),
        (True, "r", "r"),
        (True, "a", "r+"),
        (True, "a", "c"),
    ],
)
def test_open(contiguous, urlpath, cparams, dparams, nchunks, chunk_nitems, dtype, mode, mmap_mode):
    if os.name == "nt" and mmap_mode == "c":
        pytest.skip("Cannot test mmap_mode 'c' on Windows")
    if blosc2.IS_WASM and mmap_mode is not None:
        pytest.skip("mmap_mode is not supported reliably on wasm32")

    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    dtype = np.dtype(dtype)
    schunk = blosc2.SChunk(
        chunksize=chunk_nitems * dtype.itemsize, mmap_mode="w+" if mmap_mode is not None else None, **kwargs
    )
    for i in range(nchunks):
        buffer = i * np.arange(chunk_nitems, dtype=dtype)
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    if mmap_mode == "c":
        with open(urlpath, "rb") as f:
            file_contents_beginning = f.read()

    del schunk
    cparams2 = cparams
    cparams2["nthreads"] = 1
    schunk_open = blosc2.open(urlpath, mode, mmap_mode=mmap_mode, cparams=cparams2)
    assert schunk_open.cparams.nthreads == cparams2["nthreads"]

    for key in cparams:
        if key == "nthreads":
            continue
        assert getattr(schunk_open.cparams, key) == cparams[key]

    buffer = np.zeros(chunk_nitems, dtype=dtype)
    if mode != "r":
        if mode == "w":
            pos = 0
        else:
            pos = random.randint(0, nchunks)
        nchunks_ = schunk_open.insert_data(nchunk=pos, data=buffer, copy=True)
        assert nchunks_ == 1 if mode == "w" else nchunks + 1
    else:
        pos = nchunks
        with pytest.raises(ValueError):
            schunk_open.insert_data(nchunk=pos, data=buffer, copy=True)

    for i in range(pos):
        buffer = i * np.arange(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(i)
        assert res == bytes_obj
    if mode != "r":
        buffer = np.zeros(chunk_nitems, dtype=dtype)
        bytes_obj = buffer.tobytes()
        res = schunk_open.decompress_chunk(pos)
        assert res == bytes_obj
        if mode == "a":
            for i in range(pos + 1, nchunks + 1):
                buffer = (i - 1) * np.arange(chunk_nitems, dtype=dtype)
                dest = np.empty(buffer.shape, buffer.dtype)
                schunk_open.decompress_chunk(i, dest)
                assert np.array_equal(buffer, dest)

    if mmap_mode == "c":
        with open(urlpath, "rb") as f:
            file_contents_end = f.read()
        assert file_contents_beginning == file_contents_end

    blosc2.remove_urlpath(urlpath)


def test_open_fake():
    with pytest.raises(FileNotFoundError):
        _ = blosc2.open("none.b2nd", mode="r")


def test_load_ndarray_returns_in_memory_copy(tmp_path):
    urlpath = tmp_path / "array.b2nd"
    arr = blosc2.asarray(np.arange(12).reshape(3, 4), urlpath=urlpath, mode="w")

    loaded = blosc2.load(urlpath)

    assert isinstance(loaded, blosc2.NDArray)
    assert loaded.urlpath is None
    assert np.array_equal(loaded[:], arr[:])


@pytest.mark.xfail(
    blosc2.IS_WASM,
    reason="get_slice on an in-memory SChunk (loaded from a frame) fails on the "
    "Pyodide 0.29.x Emscripten toolchain; passes on Pyodide 314 and natively. "
    "See https://github.com/Blosc/python-blosc2/issues/664",
    strict=False,
)
def test_load_schunk_returns_in_memory_copy(tmp_path):
    urlpath = tmp_path / "schunk.b2frame"
    data = np.arange(20, dtype=np.int32)
    blosc2.SChunk(data=data, urlpath=urlpath, mode="w", cparams={"typesize": data.dtype.itemsize})

    loaded = blosc2.load(urlpath)

    assert isinstance(loaded, blosc2.SChunk)
    assert loaded.urlpath is None
    assert loaded[:] == data.tobytes()


@pytest.mark.parametrize("offset", [0, 42])
@pytest.mark.parametrize("urlpath", ["schunk.b2frame"])
@pytest.mark.parametrize(("mode", "mmap_mode"), [("r", None), (None, "r")])
def test_open_offset(offset, urlpath, mode, mmap_mode):
    if blosc2.IS_WASM and mmap_mode is not None:
        pytest.skip("mmap_mode is not supported reliably on wasm32")

    urlpath_temp = urlpath + ".temp"

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(urlpath_temp)

    # Create a temporary file with data.
    data = np.arange(100)
    blosc2.SChunk(data=data, urlpath=urlpath_temp, mmap_mode="w+" if mmap_mode is not None else None)
    # Create the final file with the temporary data after "offset" bytes.
    with open(urlpath, "wb") as schunk_file:
        schunk_temp_data = None
        with open(urlpath_temp, "rb") as schunk_temp_file:
            schunk_temp_data = schunk_temp_file.read()
        schunk_file.seek(offset)
        schunk_file.write(schunk_temp_data)
    blosc2.remove_urlpath(urlpath_temp)

    schunk_data = blosc2.open(urlpath, mode, mmap_mode=mmap_mode, offset=offset)[:]
    assert np.array_equal(schunk_data, data.tobytes())

    with pytest.raises(RuntimeError):
        blosc2.open(urlpath, mode, mmap_mode=mmap_mode, offset=offset + 1)

    if offset > 0:
        with pytest.raises(RuntimeError):
            blosc2.open(urlpath, mode, mmap_mode=mmap_mode)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("suffix", [".h5", ".hdf5", ".b2z", ".zarr"])
@pytest.mark.parametrize("array", [False, True])
def test_open_embedded_frame_ignores_container_suffix(tmp_path, suffix, array):
    data = np.arange(20, dtype=np.int32)
    original = blosc2.asarray(data) if array else blosc2.SChunk(data=data)
    path = tmp_path / ("embedded" + suffix)
    prefix = b"surrounding container header"
    path.write_bytes(prefix + original.to_cframe())

    for target in (path, str(path), path.as_uri()):
        with blosc2.open(target, offset=len(prefix), dparams={"nthreads": 1}) as opened:
            assert type(opened) is type(original)
            if array:
                np.testing.assert_array_equal(opened[:], data)
            else:
                assert opened[:] == data.tobytes()
    assert path.read_bytes() == prefix + original.to_cframe()


def test_open_defaults_to_readonly(tmp_path):
    """open() defaults to read-only mode when mode is omitted."""
    urlpath = str(tmp_path / "test.b2nd")
    blosc2.asarray(np.arange(10), urlpath=urlpath, mode="w")
    # Opening without explicit mode should work (read-only by default)
    obj = blosc2.open(urlpath)
    assert obj.schunk.mode == "r"
    assert obj.schunk.vlmeta.mode == "r"


def test_open_explicit_mode_no_warn(tmp_path):
    """No warnings are emitted when mode is explicitly given."""
    urlpath = str(tmp_path / "test.b2nd")
    blosc2.asarray(np.arange(10), urlpath=urlpath, mode="w")
    obj = blosc2.open(urlpath, mode="r")
    assert obj.schunk.vlmeta.mode == "r"
    obj = blosc2.open(urlpath, mode="a")
    assert obj.schunk.vlmeta.mode == "a"


def test_open_mmap_defaults_to_readonly(tmp_path):
    """When mode is omitted, mmap_mode open also defaults to read-only."""
    if blosc2.IS_WASM:
        pytest.skip("mmap_mode is not supported reliably on wasm32")

    urlpath = str(tmp_path / "test.b2nd")
    blosc2.asarray(np.arange(10), urlpath=urlpath, mode="w")
    obj = blosc2.open(urlpath, mmap_mode="r")
    assert obj.schunk.mode == "r"
    assert obj.schunk.vlmeta.mode == "r"


def test_open_ndarray_context_manager(tmp_path):
    urlpath = tmp_path / "array.b2nd"
    expected = np.arange(12).reshape(3, 4)
    blosc2.asarray(expected, urlpath=urlpath, mode="w")

    with blosc2.open(urlpath) as arr:
        assert isinstance(arr, blosc2.NDArray)
        np.testing.assert_array_equal(arr[:], expected)


def test_open_schunk_context_manager(tmp_path):
    urlpath = tmp_path / "schunk.b2frame"
    data = np.arange(20, dtype=np.int32)
    blosc2.SChunk(data=data, urlpath=urlpath, mode="w", cparams={"typesize": data.dtype.itemsize})

    with blosc2.open(urlpath, mode="r") as schunk:
        assert isinstance(schunk, blosc2.SChunk)
        assert schunk[:] == data.tobytes()


# open(..., cparams=...) of an existing container updates the stored compression parameters


def _float_array(tmp_path, **kwargs):
    urlpath = tmp_path / "a.b2nd"
    blosc2.asarray(
        np.arange(1000, dtype=np.float32), urlpath=urlpath, chunks=(500,), blocks=(100,), **kwargs
    )
    return urlpath


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
    # Typesize and blocksize stay those of the array, so that writes produce chunks that fit its blocks
    # (they used to fail, or to corrupt the file)
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
    fields = ("nthreads", "codec", "clevel", "filters")
    # A CParams instance gives every field but typesize and blocksize
    given = cparams if isinstance(cparams, dict) else {key: getattr(cparams, key) for key in fields}
    expected = {key: getattr(stored, key) for key in ("typesize", "blocksize", *fields)} | given
    expected["nthreads"] = expected_nthreads(expected["nthreads"])
    for key, value in expected.items():
        got = getattr(a.cparams, key)
        if key == "filters":  # the opened array's filters are padded to 6
            got = got[: len(value)]
        assert got == value, key

    a[:] = x[::-1].copy()
    a[1:5, 3:7, 2:9] = -1
    a.resize((60, 40, 24))
    a[48:] = 7
    a.schunk.update_data(0, np.full(24 * 20 * 24, 5, np.float32), copy=True)
    del a
    y = np.concatenate([x[::-1], np.full((12, 40, 24), 7, np.float32)])
    y[1:5, 3:7, 2:9] = -1
    y[:24, :20, :24] = 5
    np.testing.assert_array_equal(blosc2.open(urlpath)[:], y)


def test_open_cparams_mode_w_ndarray(tmp_path):
    # Mode "w" does not empty an existing NDArray, so its layout must stay too
    urlpath = _float_array(tmp_path)
    a = blosc2.open(urlpath, mode="w", cparams={"nthreads": 2})
    assert (a.cparams.typesize, a.cparams.blocksize) == (4, 100 * 4)
    a[:] = np.arange(1000, dtype=np.float32)[::-1].copy()
    del a
    np.testing.assert_array_equal(blosc2.open(urlpath)[:], np.arange(1000, dtype=np.float32)[::-1])


@pytest.mark.parametrize(
    "cparams",
    [{"typesize": 8}, {"blocksize": 1024}, blosc2.CParams(typesize=2), blosc2.CParams(blocksize=4096)],
)
def test_open_cparams_change_layout(tmp_path, cparams):
    urlpath = _float_array(tmp_path)
    with pytest.raises(ValueError, match="Cannot change the"):
        blosc2.open(urlpath, mode="a", cparams=cparams)


def test_open_cparams_blocksize_zero(tmp_path):
    # 0 (automatic) leaves the blocksize as it is
    urlpath = _float_array(tmp_path)
    assert blosc2.open(urlpath, mode="a", cparams={"blocksize": 0}).cparams.blocksize == 100 * 4


def test_open_cparams_read_mode(tmp_path):
    urlpath = tmp_path / "a.b2nd"
    blosc2.asarray(
        np.arange(1000, dtype=np.int16), urlpath=urlpath, cparams={"codec": blosc2.Codec.LZ4, "clevel": 3}
    )
    stored = blosc2.open(urlpath).cparams
    a = blosc2.open(urlpath, mode="r", cparams={"nthreads": 1})
    assert a.cparams == dataclasses.replace(stored, nthreads=1)


def test_open_cparams_filters_meta(tmp_path):
    # Without filters_meta, new filters get zero metas, as in CParams(filters=...): the stored metas
    # belong to the stored filters
    filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
    urlpath = _float_array(tmp_path, cparams={"filters": filters, "filters_meta": [23, 0]})
    assert blosc2.open(urlpath, mode="a", cparams={"nthreads": 1}).cparams.filters_meta[:2] == [23, 0]
    a = blosc2.open(urlpath, mode="a", cparams={"filters": [blosc2.Filter.BITSHUFFLE]})
    assert a.cparams.filters[0] == blosc2.Filter.BITSHUFFLE
    assert a.cparams.filters_meta[0] == 0
    y = np.arange(1000, dtype=np.float32)[::-1].copy()
    a[:] = y
    np.testing.assert_array_equal(blosc2.open(urlpath)[:], y)


@pytest.mark.parametrize(
    "cparams",
    [
        {"filters": [blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]},
        blosc2.CParams(filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]),
    ],
)
def test_open_cparams_bytedelta_meta(tmp_path, cparams):
    # BYTEDELTA's meta is the item size: that of the array, not CParams' default typesize (8)
    urlpath = _float_array(tmp_path)
    assert blosc2.open(urlpath, mode="a", cparams=cparams).cparams.filters_meta[:2] == [0, 4]


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("cparams", [{"nthreads": 2}, {"clevel": 1}, blosc2.CParams(nthreads=2)])
def test_open_cparams_keep_typesize(tmp_path, contiguous, cparams):
    # Items of an opened super-chunk keep their size: typesize used to become the default (8)
    urlpath = tmp_path / "schunk.b2frame"
    schunk = blosc2.SChunk(
        chunksize=4000,
        urlpath=urlpath,
        contiguous=contiguous,
        cparams={"typesize": 4, "codec": blosc2.Codec.LZ4HC, "clevel": 3},
    )
    schunk.append_data(np.arange(1000, dtype=np.int32))
    del schunk
    schunk = blosc2.open(urlpath, mode="a", cparams=cparams)
    assert schunk.typesize == schunk.cparams.typesize == 4
    if isinstance(cparams, dict):
        codec, clevel = blosc2.Codec.LZ4HC, cparams.get("clevel", 3)
    else:
        codec, clevel = cparams.codec, cparams.clevel
    assert (schunk.cparams.codec, schunk.cparams.clevel) == (codec, clevel)
    schunk.append_data(np.arange(1000, 2000, dtype=np.int32))
    schunk.update_data(0, np.full(1000, 5, dtype=np.int32), copy=True)
    schunk[10:20] = np.zeros(10, dtype=np.int32)
    expected = np.arange(2000, dtype=np.int32)
    expected[:1000] = 5
    expected[10:20] = 0
    np.testing.assert_array_equal(np.frombuffer(schunk[:], dtype=np.int32), expected)
    np.testing.assert_array_equal(np.frombuffer(blosc2.open(urlpath)[:], dtype=np.int32), expected)
