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

fsspec = pytest.importorskip("fsspec")


@pytest.fixture(autouse=True)
def clean_memory_fs():
    fsspec.filesystem("memory").store.clear()


def test_open_memory_url():
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://x.b2nd", "wb") as f:
        f.write(a.to_cframe())

    b = blosc2.open("memory://x.b2nd")
    assert isinstance(b, blosc2.NDArray)
    assert np.array_equal(b[:], a[:])


def test_save_array_to_url():
    a = np.arange(100, dtype="f8").reshape(10, 10)
    nbytes = blosc2.save_array(a, "memory://y.b2nd")
    assert nbytes > 0
    assert np.array_equal(blosc2.load_array("memory://y.b2nd"), a)


def test_save_tensor_to_url():
    a = np.arange(50, dtype="f4")
    blosc2.save_tensor(a, "memory://z.b2nd")
    assert np.array_equal(blosc2.load_tensor("memory://z.b2nd"), a)


def test_schunk_roundtrip():
    schunk = blosc2.SChunk(chunksize=1000)
    schunk.append_data(np.arange(1000, dtype="u1"))
    with fsspec.open("memory://s.b2f", "wb") as f:
        f.write(schunk.to_cframe())

    sc = blosc2.open("memory://s.b2f")
    assert isinstance(sc, blosc2.SChunk)
    assert sc.nbytes == schunk.nbytes


def test_chained_url(tmp_path):
    # A container inside a local zip, reached through fsspec's chained syntax
    import zipfile

    a = blosc2.arange(20, dtype="i2")
    zippath = tmp_path / "archive.zip"
    with zipfile.ZipFile(zippath, "w") as zf:
        zf.writestr("inner.b2nd", a.to_cframe())

    b = blosc2.open(f"zip://inner.b2nd::file://{zippath}")
    assert np.array_equal(b[:], a[:])


@pytest.mark.parametrize("mode", ["a", "w"])
def test_mode_not_supported(mode):
    with pytest.raises(NotImplementedError):
        blosc2.open("memory://x.b2nd", mode=mode, cache_storage="/tmp/nope")


def test_offset_needs_cache():
    with pytest.raises(NotImplementedError, match="cache_storage"):
        blosc2.open("memory://x.b2nd", offset=32)


def test_mmap_needs_cache():
    with pytest.raises(NotImplementedError, match="cache_storage"):
        blosc2.open("memory://x.b2nd", mmap_mode="r")


def test_dir_container_needs_cache():
    with pytest.raises(NotImplementedError, match="cache_storage"):
        blosc2.open("memory://store.b2d")


def test_cached_open(tmp_path):
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://c.b2nd", "wb") as f:
        f.write(a.to_cframe())

    b = blosc2.open("memory://c.b2nd", cache_storage=tmp_path)
    assert np.array_equal(b[:], a[:])
    assert any(tmp_path.iterdir())


def test_cached_open_is_local(tmp_path):
    # The cached container is a real local file, so mmap works on it
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://m.b2nd", "wb") as f:
        f.write(a.to_cframe())

    b = blosc2.open("memory://m.b2nd", cache_storage=tmp_path, mmap_mode="r")
    assert np.array_equal(b[:], a[:])


def test_cache_hit_avoids_refetch(tmp_path, monkeypatch):
    a = blosc2.arange(10, dtype="i4")
    with fsspec.open("memory://h.b2nd", "wb") as f:
        f.write(a.to_cframe())

    fetches = []
    memfs = type(fsspec.filesystem("memory"))
    orig = memfs._open
    monkeypatch.setattr(
        memfs, "_open", lambda self, path, *a, **kw: (fetches.append(path), orig(self, path, *a, **kw))[1]
    )

    blosc2.open("memory://h.b2nd", cache_storage=tmp_path)
    assert len(fetches) == 1
    blosc2.open("memory://h.b2nd", cache_storage=tmp_path)
    assert len(fetches) == 1


def test_cache_refetches_when_remote_changes(tmp_path):
    with fsspec.open("memory://s.b2nd", "wb") as f:
        f.write(blosc2.arange(10, dtype="i4").to_cframe())
    assert blosc2.open("memory://s.b2nd", cache_storage=tmp_path).shape == (10,)

    with fsspec.open("memory://s.b2nd", "wb") as f:
        f.write(blosc2.arange(20, dtype="i4").to_cframe())
    assert blosc2.open("memory://s.b2nd", cache_storage=tmp_path).shape == (20,)


def test_cached_dict_store(tmp_path):
    # A .b2d store is a directory, so it only works through the cache
    localstore = str(tmp_path / "local.b2d")
    with blosc2.DictStore(localstore, mode="w") as dstore:
        dstore["/a"] = blosc2.arange(10, dtype="i4")
        dstore["/b"] = blosc2.arange(5, dtype="f8")
    fsspec.filesystem("memory").put(localstore, "memory://store.b2d", recursive=True)

    with blosc2.open("memory://store.b2d", cache_storage=tmp_path / "cache") as dstore:
        assert sorted(dstore.keys()) == ["/a", "/b"]
        assert np.array_equal(dstore["/a"][:], np.arange(10, dtype="i4"))


def test_cached_dir_refetches_when_remote_changes(tmp_path):
    memfs = fsspec.filesystem("memory")
    cache = tmp_path / "cache"
    localstore = str(tmp_path / "d.b2d")
    with blosc2.DictStore(localstore, mode="w") as dstore:
        dstore["/a"] = blosc2.arange(10, dtype="i4")
    memfs.put(localstore, "memory://d.b2d", recursive=True)
    with blosc2.open("memory://d.b2d", cache_storage=cache) as dstore:
        assert list(dstore.keys()) == ["/a"]

    with blosc2.DictStore(localstore, mode="a") as dstore:
        dstore["/b"] = blosc2.arange(5, dtype="i4")
    memfs.rm("/d.b2d", recursive=True)
    memfs.put(localstore, "memory://d.b2d", recursive=True)
    with blosc2.open("memory://d.b2d", cache_storage=cache) as dstore:
        assert sorted(dstore.keys()) == ["/a", "/b"]


def test_cached_sparse_frame(tmp_path):
    localpath = str(tmp_path / "sparse.b2nd")
    a = blosc2.arange(1000, dtype="i4", chunks=(100,), urlpath=localpath, mode="w", contiguous=False)
    fsspec.filesystem("memory").put(localpath, "memory://sparse.b2nd", recursive=True)

    b = blosc2.open("memory://sparse.b2nd", cache_storage=tmp_path / "cache")
    assert np.array_equal(b[:], a[:])


def _put(name, arr):
    fsspec.filesystem("memory").pipe_file("/" + name, arr.to_cframe())
    return "memory://" + name


@pytest.mark.parametrize("chunks", [(100,), (37,)])
def test_lazy_roundtrip(chunks):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=chunks, blocks=(11,))
    p = blosc2.open(_put("lazy.b2nd", a), lazy=True)
    assert (p.shape, p.chunks, p.blocks, p.dtype) == (a.shape, a.chunks, a.blocks, a.dtype)
    assert np.array_equal(p[:], a[:])


def test_lazy_multidim():
    a = blosc2.arange(0, 10000, dtype="f4", shape=(100, 100), chunks=(10, 100))
    p = blosc2.open(_put("lazy2d.b2nd", a), lazy=True)
    assert np.array_equal(p[3:7, 20:30], a[3:7, 20:30])


@pytest.mark.parametrize(
    "arr",
    [
        blosc2.zeros((1000,), dtype="f8", chunks=(100,)),
        blosc2.full((1000,), np.nan, dtype="f8", chunks=(100,)),
    ],
    ids=["zeros", "nan"],
)
def test_lazy_special_chunks(arr):
    # Run-length chunks live in the offset itself, with no bytes in the file
    p = blosc2.open(_put("special.b2nd", arr), lazy=True)
    assert np.allclose(p[:], arr[:], equal_nan=True)


def test_lazy_fetches_only_touched_chunks(monkeypatch):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    url = _put("touched.b2nd", a)

    fetched = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (fetched.append(nchunk), orig(self, nchunk))[1],
    )

    p = blosc2.open(url, lazy=True)
    assert np.array_equal(p[150:250], a[150:250])
    assert fetched == [1, 2]
    # The proxy caches what it fetched, so asking again costs nothing
    assert np.array_equal(p[150:250], a[150:250])
    assert fetched == [1, 2]


def test_lazy_afetch():
    import asyncio

    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    p = blosc2.open(_put("afetch.b2nd", a), lazy=True)
    cache = asyncio.run(p.afetch(slice(150, 250)))
    assert np.array_equal(cache[150:250], a[150:250])


def test_lazy_persistent_proxy_cache(tmp_path, monkeypatch):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    url = _put("persist.b2nd", a)
    cache = str(tmp_path / "proxy.b2nd")

    fetched = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (fetched.append(nchunk), orig(self, nchunk))[1],
    )

    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[0:100], a[0:100])
    assert fetched == [0]
    del p

    # A later run picks the cache up and only fetches what is missing from it
    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[0:100], a[0:100])
    assert fetched == [0]
    assert np.array_equal(p[500:600], a[500:600])
    assert fetched == [0, 5]


def test_lazy_needs_an_ndarray():
    schunk = blosc2.SChunk(chunksize=1000)
    schunk.append_data(np.arange(1000, dtype="u1"))
    fsspec.filesystem("memory").pipe_file("/plain.b2f", schunk.to_cframe())
    with pytest.raises(NotImplementedError, match="b2nd metalayer"):
        blosc2.open("memory://plain.b2f", lazy=True)


def test_lazy_rejects_directories(tmp_path):
    localpath = str(tmp_path / "sparse.b2nd")
    blosc2.arange(0, 1000, dtype="i4", chunks=(100,), urlpath=localpath, mode="w", contiguous=False)
    fsspec.filesystem("memory").put(localpath, "memory://sparse.b2nd", recursive=True)
    with pytest.raises(NotImplementedError, match="cache_storage"):
        blosc2.open("memory://sparse.b2nd", lazy=True)


def test_lazy_not_a_frame():
    fsspec.filesystem("memory").pipe_file("/junk.b2nd", b"not a frame at all" * 4)
    with pytest.raises(ValueError, match="contiguous frame"):
        blosc2.open("memory://junk.b2nd", lazy=True)


def test_lazy_excludes_cache_storage(tmp_path):
    with pytest.raises(ValueError, match="only one"):
        blosc2.open("memory://x.b2nd", lazy=True, cache_storage=tmp_path)


def test_lazy_offset_not_supported():
    with pytest.raises(NotImplementedError, match="offset"):
        blosc2.open("memory://x.b2nd", lazy=True, offset=32)


def test_unknown_protocol():
    # fsspec owns this error; we only check that we do not swallow it into a
    # misleading FileNotFoundError
    with pytest.raises(ValueError):
        blosc2.open("nosuchproto://bucket/key.b2nd")


def test_http_does_not_reach_fsspec():
    # http(s) is reserved for Caterva2, which is entered through blosc2.URLPath;
    # a bare URL keeps failing as a missing local path rather than being fetched
    with pytest.raises(FileNotFoundError):
        blosc2.open("http://localhost:1/foo.b2nd")


def test_local_path_untouched(tmp_path):
    urlpath = str(tmp_path / "local.b2nd")
    a = blosc2.arange(10, dtype="i4", urlpath=urlpath, mode="w")
    assert np.array_equal(blosc2.open(urlpath)[:], a[:])
