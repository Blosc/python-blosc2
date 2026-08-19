#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os
import pathlib
import threading

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


def test_save_ndarray_to_url():
    a = blosc2.arange(0, 100, dtype="i4", shape=(10, 10), chunks=(5, 10))
    a.save("memory://sv.b2nd")
    b = blosc2.open("memory://sv.b2nd")
    assert np.array_equal(b[:], a[:])
    assert b.chunks == a.chunks


def test_module_save_to_url():
    a = blosc2.arange(0, 50, dtype="f8")
    blosc2.save(a, "memory://sv2.b2nd")
    assert np.array_equal(blosc2.open("memory://sv2.b2nd")[:], a[:])


def test_save_to_url_honours_cparams():
    a = blosc2.arange(0, 100, dtype="i4", shape=(10, 10), chunks=(5, 10))
    a.save("memory://sv3.b2nd", cparams=blosc2.CParams(codec=blosc2.Codec.LZ4))
    b = blosc2.open("memory://sv3.b2nd")
    assert b.schunk.cparams.codec == blosc2.Codec.LZ4
    assert np.array_equal(b[:], a[:])


def test_save_sparse_to_url():
    a = blosc2.arange(10, dtype="i4")
    with pytest.raises(NotImplementedError, match="sparse frame"):
        a.save("memory://sv4.b2nd", contiguous=False)


@pytest.mark.parametrize(
    "make",
    [
        lambda: blosc2.zeros((10,), urlpath="memory://c.b2nd", mode="w"),
        lambda: blosc2.asarray(np.arange(10), urlpath="memory://c.b2nd", mode="w"),
        lambda: blosc2.arange(10).copy(urlpath="memory://c.b2nd", mode="w"),
        lambda: blosc2.SChunk(chunksize=100, urlpath="memory://c.b2f", mode="w"),
    ],
    ids=["zeros", "asarray", "copy", "schunk"],
)
def test_container_cannot_be_backed_by_url(make):
    # These write incrementally through the C layer, which an object store
    # cannot serve; the error has to say so rather than fail deep in C
    with pytest.raises(ValueError, match="save"):
        make()


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


def test_dir_container_with_query_needs_cache():
    with pytest.raises(NotImplementedError, match="cache_storage"):
        blosc2.open("memory://store.b2d?version=1")


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
        # blocks != chunks: a rebuilt run-length chunk must carry the container's
        # blocksize, not whatever blosc2 picks when left to choose
        blosc2.zeros((1000,), dtype="f8", chunks=(100,), blocks=(10,)),
        blosc2.zeros((4_000_000,), dtype="f8", chunks=(1_000_000,)),
        blosc2.uninit((1000,), dtype="i4", chunks=(100,), blocks=(10,)),
    ],
    ids=["zeros", "nan", "small-blocks", "auto-blocks", "uninit"],
)
def test_lazy_special_chunks(arr):
    # Run-length chunks live in the offset itself, with no bytes in the file
    p = blosc2.open(_put("special.b2nd", arr), lazy=True)
    assert p[:].shape == arr.shape
    if arr.dtype.kind == "f":
        assert np.allclose(p[:], arr[:], equal_nan=True)


@pytest.mark.parametrize("clevel", [1, 5, 8, 9])
def test_lazy_any_clevel(clevel):
    # The frame header's flags are a msgpack *string* of raw bytes, and clevel
    # rides in the high nibble of one of them: from 8 up it is not valid UTF-8
    a = blosc2.arange(0, 10000, dtype="i4", chunks=(1000,), cparams={"clevel": clevel})
    p = blosc2.open(_put(f"clevel{clevel}.b2nd", a), lazy=True)
    assert np.array_equal(p[:], a[:])


def test_lazy_structured_dtype():
    data = np.zeros(1000, dtype=[("a", "<i4"), ("b", "<f8")])
    data["a"] = np.arange(1000)
    a = blosc2.asarray(data, chunks=(100,), blocks=(10,))
    p = blosc2.open(_put("struct.b2nd", a), lazy=True)
    assert p.dtype == data.dtype
    assert np.array_equal(p[150:250], data[150:250])


def test_lazy_one_request_per_chunk(monkeypatch):
    from fsspec.implementations.memory import MemoryFileSystem

    a = blosc2.arange(0, 10000, dtype="i4", shape=(100, 100), chunks=(10, 100))
    p = blosc2.open(_put("req.b2nd", a), lazy=True)

    calls = []
    orig = MemoryFileSystem.cat_file
    monkeypatch.setattr(
        MemoryFileSystem,
        "cat_file",
        lambda self, path, start=None, end=None, **kw: (
            calls.append((start, end)),
            orig(self, path, start, end, **kw),
        )[1],
    )

    assert np.array_equal(p[15:25], a[15:25])
    assert len(calls) == 2  # one range read per chunk, not one per chunk header too


def test_lazy_reads_updated_chunks(tmp_path):
    # An updated chunk is appended at the end of the frame, so offsets stop
    # being ascending and a chunk's extent has to come from the next one *in
    # file order*, not the next by index
    localpath = str(tmp_path / "upd.b2nd")
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,), urlpath=localpath, mode="w")
    a[250:350] = 7

    fsspec.filesystem("memory").pipe_file("/upd.b2nd", pathlib.Path(localpath).read_bytes())
    p = blosc2.open("memory://upd.b2nd", lazy=True)
    assert np.array_equal(p[:], a[:])


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


def test_lazy_afetch(monkeypatch):
    import asyncio

    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    p = blosc2.open(_put("afetch.b2nd", a), lazy=True)

    # aget_chunk must go through the blocking get_chunk in a worker thread.
    # Awaiting an async filesystem's own coroutine instead raises "got Future
    # attached to a different loop" on s3fs, which memory:// cannot reproduce
    # because it is not an async backend at all
    fetched = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (fetched.append(nchunk), orig(self, nchunk))[1],
    )

    cache = asyncio.run(p.afetch(slice(150, 250)))
    assert np.array_equal(cache[150:250], a[150:250])
    assert fetched == [1, 2]


def test_lazy_fetch_is_serial_when_asked(monkeypatch):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    p = blosc2.open(_put("serial.b2nd", a), lazy=True, max_concurrency=1)

    threads = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (threads.append(threading.get_ident()), orig(self, nchunk))[1],
    )

    assert np.array_equal(p[:], a[:])
    assert len(threads) == 10
    assert set(threads) == {threading.get_ident()}


@pytest.mark.parametrize("kwargs", [{}, {"max_concurrency": 4}], ids=["default", "explicit"])
def test_lazy_overlaps_fetches(monkeypatch, kwargs):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    p = blosc2.open(_put("concurrent.b2nd", a), lazy=True, **kwargs)

    # Each fetch waits for another one to be in flight, so this deadlocks into a
    # BrokenBarrierError if the fetches are actually serial
    barrier = threading.Barrier(2, timeout=10)
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (barrier.wait(), orig(self, nchunk))[1],
    )

    assert np.array_equal(p[:], a[:])


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


def test_lazy_cache_converges_for_run_length_chunks(tmp_path, monkeypatch):
    # A fetched chunk that is a run of a single value is stored in the cache as a
    # special chunk, just like the empty ones it was created with, so whether it
    # is there cannot be read off the cache itself
    a = blosc2.full((1000,), 3.0, dtype="f8", chunks=(100,))
    url = _put("runlength.b2nd", a)
    cache = str(tmp_path / "runlength-cache.b2nd")

    fetched = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (fetched.append(nchunk), orig(self, nchunk))[1],
    )

    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[:], a[:])
    assert len(fetched) == 10
    assert np.array_equal(p[:], a[:])
    assert len(fetched) == 10
    del p

    # And the same across runs, which is what the persistent cache promises
    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[:], a[:])
    assert len(fetched) == 10


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


def test_lazy_with_cache_storage(tmp_path, monkeypatch):
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))
    url = _put("lazycache.b2nd", a)

    fetched = []
    orig = blosc2.FsspecNDSource.get_chunk
    monkeypatch.setattr(
        blosc2.FsspecNDSource,
        "get_chunk",
        lambda self, nchunk: (fetched.append(nchunk), orig(self, nchunk))[1],
    )

    p = blosc2.open(url, lazy=True, cache_storage=tmp_path)
    assert np.array_equal(p[0:100], a[0:100])
    assert fetched == [0]
    del p

    # A later run starts from the chunks the previous one pulled
    p = blosc2.open(url, lazy=True, cache_storage=tmp_path)
    assert np.array_equal(p[0:100], a[0:100])
    assert fetched == [0]
    assert np.array_equal(p[500:600], a[500:600])
    assert fetched == [0, 5]


def test_lazy_cache_rebuilt_when_remote_changes(tmp_path):
    # Uncompressed, so both frames are byte-for-byte the same size: the stamp
    # cannot fall back to comparing sizes and get this right by luck
    cparams = blosc2.CParams(clevel=0)
    a = blosc2.asarray(np.arange(1000, dtype="i4"), chunks=(100,), cparams=cparams)
    b = blosc2.asarray(np.arange(7000, 8000, dtype="i4"), chunks=(100,), cparams=cparams)
    assert len(a.to_cframe()) == len(b.to_cframe())

    url = _put("lazystale.b2nd", a)
    p = blosc2.open(url, lazy=True, cache_storage=tmp_path)
    assert np.array_equal(p[0:100], a[0:100])
    del p

    # Replacing the frame invalidates both the cached chunks and the offsets
    # they were fetched by, so the cache must be thrown away rather than reused
    _put("lazystale.b2nd", b)
    p = blosc2.open(url, lazy=True, cache_storage=tmp_path)
    assert np.array_equal(p[0:100], b[0:100])


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


def test_zip_store_needs_cache(tmp_path):
    # A .b2z store is a zip archive, not a cframe, so there is nothing for the
    # in-memory read to rebuild
    localpath = str(tmp_path / "t.b2z")
    with blosc2.TreeStore(localpath, mode="w") as tstore:
        tstore["/a"] = blosc2.arange(10, dtype="i4")
    fsspec.filesystem("memory").pipe_file("/t.b2z", pathlib.Path(localpath).read_bytes())

    with pytest.raises(RuntimeError):
        blosc2.open("memory://t.b2z")
    with blosc2.open("memory://t.b2z", cache_storage=tmp_path / "cache") as tstore:
        assert np.array_equal(tstore["/a"][:], np.arange(10, dtype="i4"))


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("file:///tmp/a.b2nd", "/tmp/a.b2nd"),
        ("file://localhost/tmp/a.b2nd", "/tmp/a.b2nd"),
    ],
)
def test_normalize_file_url(url, expected):
    # as_posix() because the separator is the platform's, the layout is not
    assert expected in pathlib.PurePath(blosc2.core.normalize_urlpath(url)).as_posix()


def test_normalize_windows_drive_url():
    # file://C:/x names the host C:, which only Windows can reach, as a drive
    url = "file://C:/data/a.b2nd"
    if os.name == "nt":
        assert pathlib.PurePath(blosc2.core.normalize_urlpath(url)).as_posix() == "C:/data/a.b2nd"
    else:
        with pytest.raises(ValueError, match="C:"):
            blosc2.core.normalize_urlpath(url)


def test_normalize_file_url_with_a_host():
    # A host authority is a UNC path, which only Windows can reach; concatenating
    # it without its two slashes would silently make it a relative path instead
    url = "file://server/share/a.b2nd"
    if os.name == "nt":
        assert pathlib.PurePath(blosc2.core.normalize_urlpath(url)).as_posix() == ("//server/share/a.b2nd")
    else:
        with pytest.raises(ValueError, match="server"):
            blosc2.core.normalize_urlpath(url)


def test_file_url_uses_the_local_path(tmp_path):
    # file:// is kept off the fsspec branch so it can use mmap and every
    # container format, which only works if the scheme is stripped first
    a = blosc2.arange(10, dtype="i4")
    url = (tmp_path / "f.b2nd").as_uri()

    a.save(url)
    assert (tmp_path / "f.b2nd").is_file()
    assert np.array_equal(blosc2.open(url)[:], a[:])
    assert np.array_equal(blosc2.open(url, mmap_mode="r")[:], a[:])


def test_file_url_backs_a_container(tmp_path):
    url = (tmp_path / "c.b2nd").as_uri()
    a = blosc2.arange(10, dtype="i4", urlpath=url, mode="w")
    a[0:5] = 7
    assert np.array_equal(blosc2.open(url)[:], a[:])


def test_cached_dir_refetches_on_same_size_change(tmp_path):
    # Sizes and names alone cannot see this, and memory:// has no mtime to fall
    # back on, so the manifest has to use each backend's own identity token
    memfs = fsspec.filesystem("memory")
    memfs.pipe_file("/samesize.b2d/a.bin", b"A" * 100)
    localdir = blosc2.core.localize_fsspec_url("memory://samesize.b2d", tmp_path)
    assert pathlib.Path(localdir, "a.bin").read_bytes() == b"A" * 100

    memfs.pipe_file("/samesize.b2d/a.bin", b"B" * 100)
    localdir = blosc2.core.localize_fsspec_url("memory://samesize.b2d", tmp_path)
    assert pathlib.Path(localdir, "a.bin").read_bytes() == b"B" * 100


def test_local_path_untouched(tmp_path):
    urlpath = str(tmp_path / "local.b2nd")
    a = blosc2.arange(10, dtype="i4", urlpath=urlpath, mode="w")
    assert np.array_equal(blosc2.open(urlpath)[:], a[:])


def test_cached_container_keeps_its_extension(tmp_path):
    # An .b2e store is told apart from a bare SChunk by its name, so a cached copy
    # under fsspec's plain hash would come back as the wrong type
    localpath = str(tmp_path / "e.b2e")
    estore = blosc2.EmbedStore(urlpath=localpath, mode="w")
    estore["/a"] = blosc2.arange(10, dtype="i4")
    del estore
    fsspec.filesystem("memory").pipe_file("/e.b2e", pathlib.Path(localpath).read_bytes())

    opened = blosc2.open("memory://e.b2e", cache_storage=tmp_path / "cache")
    assert isinstance(opened, blosc2.EmbedStore)
    assert np.array_equal(opened["/a"][:], np.arange(10, dtype="i4"))


def test_lazy_empty_array(tmp_path):
    # A frame with no chunks has no offsets chunk either, so the index read has
    # nothing to decompress and used to fail with a decompression error
    a = blosc2.asarray(np.zeros((0,), dtype="i4"))
    fsspec.filesystem("memory").pipe_file("/empty.b2nd", a.to_cframe())

    b = blosc2.open("memory://empty.b2nd", lazy=True)
    assert b.shape == (0,)
    assert np.array_equal(b[:], np.zeros((0,), dtype="i4"))


def test_lazy_cache_rebuilt_when_corrupt(tmp_path):
    # An interrupted run can leave a half-written cache behind; the whole point of
    # cache_storage is surviving across runs, so it has to be discarded, not fatal
    a = blosc2.arange(100, dtype="i4", chunks=(10,))
    fsspec.filesystem("memory").pipe_file("/c.b2nd", a.to_cframe())

    with blosc2.open("memory://c.b2nd", lazy=True, cache_storage=tmp_path) as b:
        assert np.array_equal(b[:10], a[:10])
    cache = next(p for p in tmp_path.iterdir() if p.suffix == ".b2nd")
    cache.write_bytes(cache.read_bytes()[:50])

    with blosc2.open("memory://c.b2nd", lazy=True, cache_storage=tmp_path) as b:
        assert np.array_equal(b[:], a[:])


def test_max_concurrency_needs_lazy(tmp_path):
    fsspec.filesystem("memory").pipe_file("/m.b2nd", blosc2.arange(10, dtype="i4").to_cframe())
    with pytest.raises(NotImplementedError, match="max_concurrency"):
        blosc2.open("memory://m.b2nd", cache_storage=tmp_path, max_concurrency=4)


def test_storage_mapping_is_normalized(tmp_path):
    # A mapping never reaches Storage.__post_init__, which is where both the
    # file:// normalization and the fsspec rejection live
    url = (tmp_path / "s.b2nd").as_uri()
    a = blosc2.zeros((10,), dtype="i4", storage={"urlpath": url, "mode": "w"})
    assert (tmp_path / "s.b2nd").is_file()
    assert np.array_equal(blosc2.open(url)[:], a[:])

    with pytest.raises(ValueError, match="fsspec URL"):
        blosc2.zeros((10,), dtype="i4", storage={"urlpath": "memory://s.b2nd", "mode": "w"})


def test_save_to_url_rejects_reading_mode():
    a = blosc2.arange(10, dtype="i4")
    with pytest.raises(ValueError, match="reading mode"):
        a.save("memory://ro.b2nd", mode="r")
    with pytest.raises(ValueError, match="reading mode"):
        blosc2.pack_tensor(np.arange(10), urlpath="memory://ro.b2nd", mode="r")


def test_lazy_open_never_opens_a_handle(monkeypatch):
    # A buffered handle reads a whole block per seek (50 MiB on s3fs by default),
    # so the index has to come out of exact range reads instead
    a = blosc2.arange(1000, dtype="i4", chunks=(100,))
    fsspec.filesystem("memory").pipe_file("/ranges.b2nd", a.to_cframe())

    memfs = type(fsspec.filesystem("memory"))
    monkeypatch.setattr(memfs, "_open", lambda *args, **kwargs: pytest.fail("opened a handle"))

    b = blosc2.open("memory://ranges.b2nd", lazy=True)
    assert np.array_equal(b[100:200], a[100:200])


def test_handbuilt_proxy_rejects_a_stale_cache(tmp_path):
    # The FsspecNDSource docstring recommends wrapping it in a Proxy by hand, which
    # bypasses the refetch blosc2.open() does; same geometry is not the same bytes
    memfs = fsspec.filesystem("memory")
    cache = str(tmp_path / "hand.b2nd")
    memfs.pipe_file("/hand.b2nd", blosc2.arange(100, dtype="i4", chunks=(10,)).to_cframe())
    p = blosc2.Proxy(blosc2.FsspecNDSource("memory://hand.b2nd"), urlpath=cache, mode="a")
    assert np.array_equal(p[:10], np.arange(10, dtype="i4"))
    del p

    other = blosc2.arange(100, 200, dtype="i4", chunks=(10,))
    memfs.pipe_file("/hand.b2nd", other.to_cframe())
    with pytest.raises(ValueError, match="different remote bytes"):
        blosc2.Proxy(blosc2.FsspecNDSource("memory://hand.b2nd"), urlpath=cache, mode="a")

    p = blosc2.Proxy(blosc2.FsspecNDSource("memory://hand.b2nd"), urlpath=cache, mode="w")
    assert np.array_equal(p[:], other[:])


# Block-granular fetching -------------------------------------------------


@pytest.fixture
def any_chunk_wants_blocks(monkeypatch):
    """Take the size threshold out of the way, so small test arrays exercise blocks."""
    monkeypatch.setattr(blosc2.proxy, "BLOCK_MIN_CBYTES", 0)


def _traffic(monkeypatch):
    """Record every range read and whole-chunk read a source makes.

    Everything the source reads goes through `read_range`, opening the frame and
    fetching a whole chunk included, so `reads` counts requests and a
    whole-chunk fetch appears in both lists.  Install it after the open where
    the frame index would otherwise be counted in.
    """
    reads, chunks = [], []
    for name, log in (("read_range", reads), ("get_chunk", chunks)):
        orig = getattr(blosc2.FsspecNDSource, name)
        monkeypatch.setattr(
            blosc2.FsspecNDSource,
            name,
            lambda self, *args, _o=orig, _log=log: (
                data := _o(self, *args),
                _log.append(len(data)),
            )[0],
        )
    return reads, chunks


def _incompressible(shape, chunks, blocks, seed=0):
    data = np.random.default_rng(seed).random(shape)
    return data, blosc2.asarray(data, chunks=chunks, blocks=blocks)


def test_lazy_open_costs_one_read(monkeypatch):
    # The format asks how long the header is in an answer as dear as the header,
    # so the head is guessed at generously instead -- and that one read is the
    # whole of an open, since where the chunks are is nothing an open decides
    data, a = _incompressible((600, 600), (300, 600), (30, 600))
    url = _put("openreads.b2nd", a)
    reads, chunks = _traffic(monkeypatch)

    src = blosc2.FsspecNDSource(url)
    assert len(reads) == 1
    assert not chunks
    assert src.shape == (600, 600)

    # ... and the offsets are read by the first thing that asks where a chunk is
    assert len(src._offsets) == 2
    assert len(reads) == 2
    assert len(src._offsets) == 2  # read once and kept, not once per question
    assert len(reads) == 2


def test_lazy_open_of_a_small_frame_never_reads_twice(monkeypatch):
    # A frame that fits in the first read is wholly in hand: the offsets chunk
    # is in those bytes too, so there is nothing left to ask for, then or later
    a = blosc2.arange(0, 100, dtype="i4", chunks=(10,))
    assert a.schunk.cbytes < blosc2.proxy._FRAME_PREFETCH
    url = _put("smallframe.b2nd", a)
    reads, _ = _traffic(monkeypatch)

    src = blosc2.FsspecNDSource(url)
    assert len(reads) == 1
    assert len(src._offsets) == 10
    assert len(reads) == 1
    assert np.array_equal(blosc2.Proxy(src)[:], a[:])


def test_lazy_open_reads_a_header_that_did_not_fit(monkeypatch):
    # A metalayer big enough to push the header past the guess: the exact read
    # the format asks for happens after all, and nothing is misread
    monkeypatch.setattr(blosc2.proxy, "_FRAME_PREFETCH", 256)
    data = np.arange(1000, dtype="i4")
    a = blosc2.asarray(data, chunks=(100,), meta={"big": {"pad": "x" * 4096}})
    url = _put("bigheader.b2nd", a)
    reads, _ = _traffic(monkeypatch)

    src = blosc2.FsspecNDSource(url)
    assert len(reads) == 2  # the guess, then the header
    assert reads[1] > 4096
    assert np.array_equal(blosc2.Proxy(src)[:], data)
    assert len(reads) > 2  # the offsets, and the chunks themselves


def test_a_cache_that_holds_the_slice_asks_for_no_offsets(monkeypatch, tmp_path):
    # What the deferral is for: a later run over a cache that already covers the
    # slice fetches nothing, and so has no use for where the chunks are either
    data, a = _incompressible((600, 600), (300, 600), (30, 600))
    url = _put("cachedslice.b2nd", a)
    cache = str(tmp_path / "cachedslice-cache.b2nd")
    item = (slice(0, 30), slice(0, 600))
    blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="w").fetch(item)

    reads, _ = _traffic(monkeypatch)
    src = blosc2.FsspecNDSource(url)
    proxy = blosc2.Proxy(src, urlpath=cache, mode="a")
    assert len(reads) == 1  # the header, which is what says the frame is readable

    proxy.fetch(item)
    assert len(reads) == 1
    assert np.array_equal(proxy[0:30, 0:600], data[0:30, 0:600])
    assert len(reads) == 1

    # A slice it does not hold pays for the offsets then, and reads right
    assert np.array_equal(proxy[300:330, 0:600], data[300:330, 0:600])
    assert len(reads) > 1


def test_lazy_open_reads_an_index_that_did_not_fit(monkeypatch):
    # The same for the offsets chunk, which is bounded by the frame's own length
    # but capped in case a large trailer sits behind it
    monkeypatch.setattr(blosc2.proxy, "_INDEX_PREFETCH", 16)
    data, a = _incompressible((600, 600), (30, 600), (30, 600))
    url = _put("bigindex.b2nd", a)
    reads, _ = _traffic(monkeypatch)

    src = blosc2.FsspecNDSource(url)
    assert len(reads) == 1  # the head, and the offsets not asked for yet
    assert len(src._offsets) == 20
    assert len(reads) == 3  # ... then the capped tail, and the offsets in full
    assert reads[1] == 16
    assert np.array_equal(blosc2.Proxy(src)[:], data)


def test_lazy_fetches_only_touched_blocks(monkeypatch):
    # Chunks big enough to be worth taking apart, at the real threshold
    data, a = _incompressible((600, 600), (300, 600), (30, 600))
    cbytes = a.schunk.cbytes // a.schunk.nchunks
    assert cbytes > blosc2.proxy.BLOCK_MIN_CBYTES

    p = blosc2.open(_put("blocks.b2nd", a), lazy=True)
    reads, chunks = _traffic(monkeypatch)  # after the open, which reads the header
    assert np.array_equal(p[10:12, 30:40], data[10:12, 30:40])
    # One read for where the chunks are, one for the block offsets inside the one
    # wanted, one for the single block the slice lands in
    assert len(reads) == 3
    assert not chunks
    assert sum(reads) < cbytes / 5


def test_lazy_small_chunks_are_fetched_whole(monkeypatch):
    # Below the threshold a chunk is one cheap request, so blocks would only add
    # a round trip; nothing must go looking for block offsets
    a = blosc2.arange(0, 1000, dtype="i4", chunks=(100,))

    p = blosc2.open(_put("smallblocks.b2nd", a), lazy=True)
    reads, chunks = _traffic(monkeypatch)
    assert np.array_equal(p[150:250], a[150:250])
    assert len(chunks) == 2
    assert len(reads) == len(chunks)  # one request each, none of them for block offsets


def test_lazy_whole_array_skips_the_block_path(monkeypatch, any_chunk_wants_blocks):
    # Wanting every block of a chunk is what fetching the chunk already does
    data, a = _incompressible((200, 200), (100, 200), (10, 200))

    p = blosc2.open(_put("wholeblocks.b2nd", a), lazy=True)
    reads, chunks = _traffic(monkeypatch)
    assert np.array_equal(p[:], data)
    assert len(chunks) == 2
    # One request per chunk and none for block offsets, after the one that says
    # where the chunks are
    assert len(reads) == len(chunks) + 1


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "item"),
    [
        ((1000,), (500,), (50,), slice(120, 140)),
        ((200, 200), (100, 200), (10, 20), (slice(5, 7), slice(30, 90))),
        ((20, 60, 60), (10, 30, 60), (5, 10, 20), (3, slice(10, 20), slice(10, 20))),
        ((200, 200), (100, 200), (10, 20), (5, 5)),
        ((200, 200), (100, 200), (10, 20), (slice(None), 7)),
    ],
    ids=["1d", "2d", "3d", "point", "column"],
)
def test_lazy_block_reads_are_correct(monkeypatch, any_chunk_wants_blocks, shape, chunks, blocks, item):
    data, a = _incompressible(shape, chunks, blocks)
    p = blosc2.open(_put("geom.b2nd", a), lazy=True)
    assert np.array_equal(p[item], data[item])
    # ... and the rest of the array still arrives correctly afterwards
    assert np.array_equal(p[...], data)


def test_lazy_blocks_accumulate_in_a_chunk(monkeypatch, any_chunk_wants_blocks):
    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    reads, _ = _traffic(monkeypatch)
    p = blosc2.open(_put("accum.b2nd", a), lazy=True)

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    after_first = len(reads)
    # A different block of the same chunk: what is already cached stays cached
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(reads) > after_first
    after_second = len(reads)
    # Both are now cached in the same chunk, and both are still right
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(reads) == after_second


def test_lazy_block_cache_survives_reopen(tmp_path, monkeypatch, any_chunk_wants_blocks):
    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    url = _put("blockcache.b2nd", a)
    cache = str(tmp_path / "blocks-cache.b2nd")
    reads, _ = _traffic(monkeypatch)

    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    # A partly filled chunk survives, so the blocks in it do not travel again
    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    reopened = len(reads)  # every open reads the header afresh
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(reads) == reopened
    # ... and the ones missing from it still do
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(reads) > reopened
    assert np.array_equal(p[...], data)


def test_lazy_blocks_merge_adjacent_reads(monkeypatch, any_chunk_wants_blocks):
    # Neighbouring blocks are near-adjacent in the file, so a slice spanning many
    # of them must not cost one request each
    data, a = _incompressible((200, 200), (200, 200), (2, 200))
    reads, _ = _traffic(monkeypatch)
    p = blosc2.open(_put("merge.b2nd", a), lazy=True)

    assert np.array_equal(p[0:40], data[0:40])
    assert len(reads) < 1 + 20  # the offsets, plus fewer requests than blocks


def test_lazy_blocks_fall_back_for_memcpyed_chunks(monkeypatch, any_chunk_wants_blocks):
    # A memcpyed chunk stores its blocks raw and has no offsets section to read
    data = np.random.default_rng(0).integers(0, 256, (300, 300), dtype="u1")
    a = blosc2.asarray(data, chunks=(150, 300), blocks=(15, 300), cparams={"clevel": 0})

    p = blosc2.open(_put("memcpyed.b2nd", a), lazy=True)
    reads, chunks = _traffic(monkeypatch)
    assert np.array_equal(p[10:12, 30:40], data[10:12, 30:40])
    assert len(chunks) == 1
    # Where the chunks are, then the block offsets, which say there is nothing to
    # skip, and the chunk follows
    assert len(reads) == 3


def test_lazy_blocks_with_run_length_chunks(monkeypatch, any_chunk_wants_blocks):
    # A chunk that is a run of a single value has no bytes in the file at all
    data = np.zeros((200, 200))
    data[100:] = np.random.default_rng(0).random((100, 200))
    a = blosc2.asarray(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.open(_put("runlen.b2nd", a), lazy=True)

    assert np.array_equal(p[0:2, 0:10], data[0:2, 0:10])
    assert np.array_equal(p[100:102, 0:10], data[100:102, 0:10])
    assert np.array_equal(p[...], data)


def test_lazy_blocks_of_a_structured_array(monkeypatch, any_chunk_wants_blocks):
    data = np.empty(10_000, dtype=[("a", "i4"), ("b", "f8")])
    rng = np.random.default_rng(0)
    data["a"], data["b"] = rng.integers(0, 1000, 10_000), rng.random(10_000)
    a = blosc2.asarray(data, chunks=(5_000,), blocks=(500,))
    p = blosc2.open(_put("structured.b2nd", a), lazy=True)

    assert np.array_equal(p[10:20], data[10:20])
    assert np.array_equal(p[...], data)


def test_lazy_blocks_serve_a_single_block_chunk(monkeypatch, any_chunk_wants_blocks):
    # A chunk of one block is its own block: nothing to take apart
    data, a = _incompressible((200, 200), (100, 200), (100, 200))
    reads, chunks = _traffic(monkeypatch)
    p = blosc2.open(_put("oneblock.b2nd", a), lazy=True)

    assert np.array_equal(p[0:2], data[0:2])
    assert len(chunks) == 1


def test_lazy_blocks_reuse_what_they_just_wrote(monkeypatch, any_chunk_wants_blocks):
    # Growing a chunk block by block must not read it back out of the cache each
    # time; the blocks already in it are still in hand
    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    p = blosc2.open(_put("hot.b2nd", a), lazy=True)
    read_back = []
    orig = p.schunk.get_chunk
    monkeypatch.setattr(p.schunk, "get_chunk", lambda n: (out := orig(n), read_back.append(n))[0])

    for col in range(0, 200, 20):
        assert np.array_equal(p[0:5, col : col + 5], data[0:5, col : col + 5])
    assert not read_back


def test_lazy_blocks_survive_eviction(monkeypatch, any_chunk_wants_blocks):
    # More chunks in flight than the hot cache holds: the evicted ones fall back
    # to reading the chunk back, which must reconstruct exactly the same blocks
    monkeypatch.setattr(blosc2.proxy, "BLOCK_HOT_CHUNKS", 2)
    data, a = _incompressible((400, 200), (100, 200), (10, 20))
    p = blosc2.open(_put("evict.b2nd", a), lazy=True)

    for chunk_row in range(0, 400, 100):  # touch every chunk once, evicting as it goes
        assert np.array_equal(p[chunk_row : chunk_row + 5, 0:5], data[chunk_row : chunk_row + 5, 0:5])
    # ... then come back to the first, whose blocks are no longer held
    assert np.array_equal(p[0:5, 100:105], data[0:5, 100:105])
    assert np.array_equal(p[0:5, 0:5], data[0:5, 0:5])
    assert np.array_equal(p[...], data)


def test_lazy_blocks_after_a_whole_chunk_arrives(any_chunk_wants_blocks):
    # afetch fetches whole chunks, which replaces a partly filled one; what was
    # held about its blocks must not be spliced back over it afterwards
    import asyncio

    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    p = blosc2.open(_put("mixed.b2nd", a), lazy=True)

    assert np.array_equal(p[0:5, 0:5], data[0:5, 0:5])
    asyncio.run(p.afetch((slice(0, 5), slice(None))))
    assert np.array_equal(p[0:5], data[0:5])
    assert np.array_equal(p[...], data)


def test_lazy_blocks_after_an_eviction(monkeypatch, any_chunk_wants_blocks):
    # Evicting a chunk that holds only some of its blocks must clear all of them,
    # including the copies the proxy keeps in hand for the next splice
    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    reads, _ = _traffic(monkeypatch)
    p = blosc2.open(_put("evicted.b2nd", a), lazy=True)

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    fetched = len(reads)
    p.schunk.update_special(0, blosc2.SpecialValue.UNINIT)

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(reads) > fetched
    assert np.array_equal(p[...], data)


@pytest.mark.parametrize(
    ("cparams", "why"),
    [
        ({"use_dict": True, "codec": blosc2.Codec.ZSTD}, "dictionary"),
        ({"clevel": 0}, "memcpyed"),
    ],
    ids=["use_dict", "clevel0"],
)
def test_lazy_blocks_skip_chunks_they_cannot_splice(monkeypatch, any_chunk_wants_blocks, cparams, why):
    # A chunk compressed against a codec dictionary keeps it between the block
    # offsets and the blocks, so a spliced chunk would promise a dictionary it
    # does not carry.  The header says so; such chunks are fetched whole.
    rng = np.random.default_rng(0)
    data = np.tile(rng.random(500), 400).reshape(400, 500)  # repetitive, so a dict pays
    a = blosc2.asarray(data, chunks=(200, 500), blocks=(20, 500), cparams=cparams)
    reads, chunks = _traffic(monkeypatch)
    p = blosc2.open(_put(f"nosplice-{why}.b2nd", a), lazy=True)

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(chunks) == 1  # the whole chunk, after the one header read that found out
    assert np.array_equal(p[...], data)


def test_lazy_eviction_survives_a_reopen(tmp_path, monkeypatch, any_chunk_wants_blocks):
    # Evict a chunk and close without reading again: the record of what the cache
    # holds lives in the cache, so the next run must not trust the evicted bit
    data, a = _incompressible((200, 200), (100, 200), (10, 20))
    url = _put("evict-reopen.b2nd", a)
    cache = str(tmp_path / "evicted-cache")
    reads, _ = _traffic(monkeypatch)

    p = blosc2.open(url, lazy=True, cache_storage=cache)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    fetched = len(reads)
    p.schunk.update_special(0, blosc2.SpecialValue.UNINIT)
    del p

    q = blosc2.open(url, lazy=True, cache_storage=cache)
    assert np.array_equal(q[0:5, 0:10], data[0:5, 0:10])
    assert len(reads) > fetched


def test_lazy_blocks_with_a_repeated_value_chunk(monkeypatch, any_chunk_wants_blocks):
    # blosc2.full() writes a chunk that is its header plus the value it repeats,
    # at a real offset -- unlike a run of zeros, which the frame keeps in the
    # offsets themselves.  There are no block offsets to read there, and reading
    # them anyway walks into whatever follows the chunk.
    a = blosc2.full((400, 500), fill_value=3.5, chunks=(200, 500), blocks=(20, 500))
    reads, chunks = _traffic(monkeypatch)
    p = blosc2.open(_put("repeated.b2nd", a), lazy=True)

    assert p.src.chunk_layout(0) is None
    assert np.array_equal(p[0:5, 0:10], np.full((5, 10), 3.5))
    assert len(chunks) == 1
    assert np.array_equal(p[...], np.full((400, 500), 3.5))
