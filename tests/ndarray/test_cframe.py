#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Serialization of arrays to cframes (to_cframe) and back (ndarray_from_cframe / schunk_from_cframe), for
# every storage kind and for chunks that are special (zeros, NaNs, uninitialized), updated or resized.

import numpy as np
import pytest

import blosc2

STORAGES = ["memory", "memory_contiguous", "file", "directory"]


def storage_kwargs(storage, tmp_path):
    if storage == "memory":
        return {}
    if storage == "memory_contiguous":
        return {"contiguous": True}
    if storage == "file":
        return {"urlpath": str(tmp_path / "src.b2nd"), "contiguous": True, "mode": "w"}
    return {"urlpath": str(tmp_path / "src_dir.b2nd"), "contiguous": False, "mode": "w"}


def data2d():
    return (np.sin(np.arange(200 * 300, dtype=np.float32) / 50) * 100).reshape(200, 300)


def make_plain(**kw):
    return blosc2.asarray(data2d(), chunks=(50, 300), blocks=(10, 300), **kw)


def make_updated(**kw):
    a = blosc2.asarray(data2d(), chunks=(50, 300), blocks=(10, 300), **kw)
    a[10:20] = 0
    a[60:70, :100] = np.arange(1000, dtype=np.float32).reshape(10, 100)
    a.resize((260, 300))  # zero chunks at the end
    a[255:260] = 7
    return a


def make_shrunk(**kw):
    a = blosc2.asarray(data2d(), chunks=(30, 300), blocks=(10, 300), **kw)
    a.resize((95, 300))
    return a


def make_meta(**kw):
    a = blosc2.asarray(data2d(), meta={"m1": [1, 2, 3], "m2": "hello"}, **kw)
    a.vlmeta["v1"] = {"a": 1, "b": [1.5, 2]}
    a.vlmeta["v2"] = b"x" * 1000
    return a


def make_mixed_special(**kw):
    a = blosc2.zeros((1000,), dtype=np.int16, chunks=(100,), blocks=(20,), **kw)
    a[150:450] = np.arange(300, dtype=np.int16)
    a[900:] = 3
    return a


def make_zeros(**kw):
    return blosc2.zeros((500, 500), dtype=np.float64, chunks=(100, 500), **kw)


def make_nans(**kw):
    return blosc2.full((500, 500), np.nan, dtype=np.float32, chunks=(100, 500), **kw)


def make_empty(**kw):
    return blosc2.zeros((0, 10), dtype=np.float32, **kw)


def make_scalar(**kw):
    return blosc2.asarray(np.array(3.5), **kw)


def make_many_chunks(**kw):
    # more than 2048 chunks: the frame offsets take several blocks
    return blosc2.asarray(np.arange(300_000, dtype=np.int32), chunks=(100,), blocks=(100,), **kw)


MAKERS = [
    make_plain,
    make_updated,
    make_shrunk,
    make_meta,
    make_mixed_special,
    make_zeros,
    make_nans,
    make_empty,
    make_scalar,
    make_many_chunks,
]


def check_same(a, b):
    assert b.shape == a.shape
    assert b.dtype == a.dtype
    assert b.chunks == a.chunks
    assert b.blocks == a.blocks
    if a.ndim == 0:
        np.testing.assert_array_equal(b[()], a[()])
    else:
        np.testing.assert_array_equal(b[...], a[...])
    assert b.schunk.meta.getall() == a.schunk.meta.getall()
    assert b.vlmeta.getall() == a.vlmeta.getall()


@pytest.mark.parametrize("storage", STORAGES)
@pytest.mark.parametrize("make", MAKERS)
def test_roundtrip(make, storage, tmp_path):
    a = make(**storage_kwargs(storage, tmp_path))
    cframe = a.to_cframe()
    assert isinstance(cframe, bytes)
    for copy in (True, False):
        check_same(a, blosc2.ndarray_from_cframe(cframe, copy=copy))
    check_same(a, blosc2.from_cframe(cframe))
    # Saving goes through the same frame building
    path = str(tmp_path / "saved.b2nd")
    a.save(path, mode="w")
    check_same(a, blosc2.open(path))
    copied = a.copy(urlpath=str(tmp_path / "copied.b2nd"), mode="w")
    check_same(a, copied)


@pytest.mark.parametrize("storage", ["memory", "file", "directory"])
@pytest.mark.parametrize("make", MAKERS)
def test_reserialize(make, storage, tmp_path):
    # Arrays that are not a frame in memory get a new frame built; it must not depend on where the
    # chunks come from (special chunks are plain chunks in a sparse super-chunk, but offsets in a frame)
    a = make(**storage_kwargs(storage, tmp_path))
    if a.schunk.nchunks > 2048:
        pytest.skip("the offsets take several blocks, compressed by several threads in a varying order")
    cframe = a.to_cframe()
    assert blosc2.ndarray_from_cframe(cframe, copy=True).to_cframe() == cframe
    assert a.to_cframe() == cframe
    if storage == "memory":
        path = str(tmp_path / "saved.b2nd")
        a.save(path, mode="w")
        assert blosc2.open(path).to_cframe() == cframe


@pytest.mark.parametrize("storage", STORAGES)
def test_schunk_roundtrip(storage, tmp_path):
    kw = storage_kwargs(storage, tmp_path)
    kw.pop("mode", None)
    if "urlpath" in kw:
        blosc2.remove_urlpath(kw["urlpath"])
    schunk = blosc2.SChunk(chunksize=4000, cparams={"typesize": 4}, **kw)
    schunk.append_data(np.arange(1000, dtype=np.int32))
    schunk.append_data(np.zeros(1000, dtype=np.int32))
    schunk.append_data(np.arange(500, dtype=np.int32))  # a shorter last chunk
    schunk.vlmeta["k"] = "v"
    cframe = schunk.to_cframe()
    for copy in (True, False):
        back = blosc2.schunk_from_cframe(cframe, copy=copy)
        assert back.nchunks == 3
        assert back[:] == schunk[:]
        assert back.vlmeta["k"] == "v"


def test_schunk_empty():
    schunk = blosc2.SChunk(chunksize=4000, cparams={"typesize": 4})
    cframe = schunk.to_cframe()
    back = blosc2.schunk_from_cframe(cframe, copy=True)
    assert back.nchunks == 0
    assert back.to_cframe() == cframe
