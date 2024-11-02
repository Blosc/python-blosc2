#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os
from dataclasses import asdict, fields, replace

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("urlpath", "contiguous", "mode", "mmap_mode"),
    [
        (None, False, "r", None),
        (None, False, "w", None),
        (None, False, "a", None),
        (None, True, "r", None),
        (None, True, "w", None),
        (None, True, "a", None),
        ("b2frame", False, "r", None),
        ("b2frame", False, "w", None),
        ("b2frame", False, "a", None),
        ("b2frame", True, "r", None),
        ("b2frame", True, "w", None),
        ("b2frame", True, "a", None),
        ("b2frame", True, "r", "r"),
        ("b2frame", True, "w", "w+"),
        ("b2frame", True, "a", "w+"),  # r+ cannot be used here because the file does not exist
    ],
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        (blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4), blosc2.DParams(), 0),
        ({"typesize": 4}, blosc2.DParams(nthreads=4), 1),
        (blosc2.CParams(splitmode=blosc2.SplitMode.ALWAYS_SPLIT, nthreads=5, typesize=4), {}, 5),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 10),
    ],
)
def test_schunk_numpy(contiguous, urlpath, mode, mmap_mode, cparams, dparams, nchunks):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath, mode=mode, mmap_mode=mmap_mode)
    blosc2.remove_urlpath(urlpath)

    chunk_len = 200 * 1000
    if mode != "r":
        schunk = blosc2.SChunk(chunksize=chunk_len * 4, storage=storage, cparams=cparams, dparams=dparams)

    else:
        with pytest.raises(
            ValueError, match="not specify a urlpath" if urlpath is None else "does not exist"
        ):
            blosc2.SChunk(chunksize=chunk_len * 4, storage=storage, cparams=cparams, dparams=dparams)

        # Create a schunk which we can read later
        storage2 = replace(
            storage,
            mode="w" if mmap_mode is None else None,
            mmap_mode="w+" if mmap_mode is not None else None,
        )
        schunk = blosc2.SChunk(chunksize=chunk_len * 4, storage=storage2, cparams=cparams, dparams=dparams)

    assert schunk.urlpath == urlpath
    assert schunk.contiguous == contiguous

    for i in range(nchunks):
        buffer = i * np.arange(chunk_len, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    if mode == "r":
        if urlpath is not None:
            schunk = blosc2.SChunk(chunksize=chunk_len * 4, **asdict(storage))
        else:
            return
    assert schunk.nchunks == nchunks

    for i in range(nchunks):
        buffer = i * np.arange(chunk_len, dtype="int32")
        bytes_obj = buffer.tobytes()
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj

        dest = np.empty(buffer.shape, buffer.dtype)
        schunk.decompress_chunk(i, dest)
        assert np.array_equal(buffer, dest)

        schunk.decompress_chunk(i, memoryview(dest))
        assert np.array_equal(buffer, dest)

        dest = bytearray(buffer)
        schunk.decompress_chunk(i, dest)
        assert dest == bytes_obj

    for i in range(nchunks):
        schunk.get_chunk(i)

    if nchunks >= 2:
        assert schunk.cratio > 1
        assert schunk.cratio == schunk.nbytes / schunk.cbytes
    assert schunk.nbytes >= nchunks * chunk_len * 4

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("mode_write", "mode_read", "mmap_mode_write", "mmap_mode_read"),
    [("w", "r", None, None), (None, None, "w+", "r")],
)
def test_schunk_ndarray(tmp_path, mode_write, mode_read, mmap_mode_write, mmap_mode_read):
    urlpath = tmp_path / "test.b2nd"

    data = np.arange(2 * 10, dtype="int32")
    blosc2.asarray(data, urlpath=urlpath, mode=mode_write, mmap_mode=mmap_mode_write)
    with pytest.raises(ValueError, match="Cannot open an NDArray as a SChunk"):
        blosc2.SChunk(mode=mode_read, mmap_mode=mmap_mode_read, urlpath=urlpath)


@pytest.mark.parametrize(
    ("urlpath", "contiguous", "mode", "mmap_mode"),
    [
        (None, False, "w", None),
        (None, True, "w", None),
        ("b2frame", False, "w", None),
        ("b2frame", True, "w", None),
        ("b2frame", True, None, "w+"),
    ],
)
@pytest.mark.parametrize(
    ("nbytes", "cparams", "dparams", "nchunks"),
    [
        (7, blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=5), {}, 1),
        (641091, {"typesize": 3}, blosc2.DParams(nthreads=2), 1),
        (136, blosc2.CParams(typesize=1), blosc2.DParams(), 5),
        (1232, {"typesize": 8}, blosc2.dparams_dflts, 10),
    ],
)
def test_schunk(contiguous, urlpath, mode, mmap_mode, nbytes, cparams, dparams, nchunks):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    numpy_meta = {b"dtype": str(np.dtype(np.uint8))}
    test_meta = {b"lorem": 1234}
    meta = {"numpy": numpy_meta, "test": test_meta}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=2 * nbytes, meta=meta, mode=mode, mmap_mode=mmap_mode, **kwargs)

    assert "numpy" in schunk.meta
    assert "error" not in schunk.meta
    assert schunk.meta["numpy"] == numpy_meta
    assert "test" in schunk.meta
    assert schunk.meta["test"] == test_meta
    test_meta = {b"lorem": 4231}
    schunk.meta["test"] = test_meta
    assert schunk.meta["test"] == test_meta

    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)

    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        res = schunk.decompress_chunk(i)
        assert res == bytes_obj

        dest = bytearray(bytes_obj)
        schunk.decompress_chunk(i, dst=dest)
        assert dest == bytes_obj

    for i in range(nchunks):
        schunk.get_chunk(i)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("urlpath", "contiguous", "mode", "mmap_mode"),
    [
        (None, False, "w", None),
        (None, True, "w", None),
        ("b2frame", False, "w", None),
        ("b2frame", True, "w", None),
        ("b2frame", True, None, "w+"),
    ],
)
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, blosc2.DParams(), 1),
        ({"typesize": 4}, {"nthreads": 4}, 1),
        (blosc2.CParams(splitmode=blosc2.SplitMode.ALWAYS_SPLIT, nthreads=5, typesize=4), {}, 5),
        (blosc2.CParams(codec=blosc2.Codec.LZ4HC, typesize=4), blosc2.DParams(), 10),
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_schunk_cframe(contiguous, urlpath, mode, mmap_mode, cparams, dparams, nchunks, copy):
    storage = blosc2.Storage(contiguous=contiguous, urlpath=urlpath, mode=mode, mmap_mode=mmap_mode)
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 1000 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(
        chunksize=200 * 1000 * 4, data=data, **asdict(storage), cparams=cparams, dparams=dparams
    )

    cframe = schunk.to_cframe()
    schunk2 = blosc2.schunk_from_cframe(cframe, copy)
    cparams_dict = cparams if isinstance(cparams, dict) else asdict(cparams)
    if not os.getenv("BTUNE_TRADEOFF"):
        for key in cparams_dict:
            if key == "nthreads":
                continue
            if key == "blocksize" and cparams_dict[key] == 0:
                continue
            assert getattr(schunk2.cparams, key) == cparams_dict[key]

    data2 = np.empty(data.shape, dtype=data.dtype)
    schunk2.get_slice(out=data2)
    assert np.array_equal(data, data2)

    cframe = schunk.to_cframe()
    schunk3 = blosc2.schunk_from_cframe(cframe, copy)
    del schunk3
    # Check that we can still access the external cframe buffer
    _ = str(cframe)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("cparams", "dparams", "new_cparams", "new_dparams"),
    [
        (
            blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4),
            {},
            blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=6, typesize=4),
            blosc2.DParams(nthreads=4),
        ),
        (
            {"typesize": 4},
            blosc2.DParams(nthreads=4),
            blosc2.CParams(codec=blosc2.Codec.ZLIB, splitmode=blosc2.SplitMode.ALWAYS_SPLIT),
            blosc2.DParams(nthreads=1),
        ),
        (
            {"codec": blosc2.Codec.ZLIB, "splitmode": blosc2.SplitMode.ALWAYS_SPLIT},
            {},
            blosc2.CParams(
                splitmode=blosc2.SplitMode.ALWAYS_SPLIT,
                nthreads=5,
                typesize=4,
                filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.TRUNC_PREC],
            ),
            blosc2.DParams(nthreads=16),
        ),
        (
            blosc2.CParams(codec=blosc2.Codec.LZ4HC, typesize=4),
            blosc2.DParams(),
            blosc2.CParams(filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.TRUNC_PREC]),
            blosc2.DParams(nthreads=3),
        ),
    ],
)
def test_schunk_cdparams(cparams, dparams, new_cparams, new_dparams):
    kwargs = {"cparams": cparams, "dparams": dparams}

    chunk_len = 200 * 1000
    schunk = blosc2.SChunk(chunksize=chunk_len * 4, **kwargs)

    # Check cparams have been set correctly
    cparams_dict = cparams if isinstance(cparams, dict) else asdict(cparams)
    dparams_dict = dparams if isinstance(dparams, dict) else asdict(dparams)
    for key in cparams_dict:
        assert getattr(schunk.cparams, key) == cparams_dict[key]
    for key in dparams_dict:
        assert getattr(schunk.dparams, key) == dparams_dict[key]

    schunk.cparams = new_cparams
    schunk.dparams = new_dparams
    for field in fields(schunk.cparams):
        if field.name in ["filters", "filters_meta"]:
            assert getattr(schunk.cparams, field.name)[: len(getattr(new_cparams, field.name))] == getattr(
                new_cparams, field.name
            )
        else:
            assert getattr(schunk.cparams, field.name) == getattr(new_cparams, field.name)

    assert schunk.dparams.nthreads == new_dparams.nthreads
