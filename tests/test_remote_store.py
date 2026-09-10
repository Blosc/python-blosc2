"""Public remote discovery, shared readers and dependent handle lifetime."""

import gc
import json
import weakref

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


def test_shared_memory_lru_and_revisit(hierarchy):
    url, data = hierarchy
    with blosc2.RemoteStore(url) as store:
        assert store.max_cache_bytes == 256 * 1024**2
        a, b = store["group/a"], store["group/b"]
        first = np.s_[:10, :10]
        second = np.s_[10:20, :10]
        for array, offset in ((a, 0), (b, 1), (a, 0)):
            np.testing.assert_array_equal(array[first], data[first] + offset)
        assert store.cache_bytes == a.cache_bytes + b.cache_bytes
        a.close()
        with store["group"] as group, group["a"] as alias:
            before = store.traffic.nbytes
            np.testing.assert_array_equal(alias[first], data[first])
            assert store.traffic.nbytes == before
            # Calibrate the limit from actual compressed payloads, then force
            # eviction of B's chunk after touching A's chunk.
            coordinator = store._owner.cache_coordinator
            coordinator.max_cache_bytes = store.cache_bytes
            np.testing.assert_array_equal(alias[second], data[second])
            assert store.cache_bytes <= coordinator.max_cache_bytes
            assert not b.cache_contains(first)
            assert alias.cache_contains(second)
            assert store.cache_bytes == alias.cache_bytes + b.cache_bytes
        b.close()


def test_shared_memory_partial_and_failed_publication(hierarchy, monkeypatch):
    url, data = hierarchy
    with blosc2.RemoteStore(url, max_cache_bytes=1) as store, store["group/a"] as a:
        proxy = a._proxy
        # Fail after publishing a native chunk. The operation's finally block
        # must still enforce the store budget before propagating the failure.
        original = proxy._store_chunk

        def fail(*args, **kwargs):
            original(*args, **kwargs)
            raise OSError("publication failed")

        with monkeypatch.context() as patch:
            patch.setattr(proxy, "_store_chunk", fail)
            with pytest.raises(OSError, match="publication failed"):
                a.get_chunk(0)
        assert store.cache_bytes == 0
        np.testing.assert_array_equal(a[:2, :2], data[:2, :2])
        assert store.cache_bytes == 0

    with blosc2.RemoteStore(url) as store, store["group/a"] as a:
        for item in (np.s_[:2, :2], np.s_[:5, :5], np.s_[:10, :10]):
            np.testing.assert_array_equal(a[item], data[item])
            proxy = a._proxy
            expected = sum(proxy._cache_sizes.values()) + sum(
                len(payload) for blocks in proxy._hot_payloads.values() for payload in blocks.values()
            )
            assert store.cache_bytes == a.cache_bytes == expected
        a.trim_cache(0)
        assert store.cache_bytes == 0


def test_shared_memory_failed_eviction_keeps_charge(hierarchy, monkeypatch):
    url, data = hierarchy
    with blosc2.RemoteStore(url) as store, store["group/a"] as a:
        np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        before = store.cache_bytes
        schunk = a._proxy.schunk
        original = type(schunk).update_special

        def fail(self, *args, **kwargs):
            if self is schunk:
                raise OSError("eviction failed")
            return original(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(schunk), "update_special", fail)
            with pytest.raises(OSError, match="eviction failed"):
                a.trim_cache(0)
        assert store.cache_bytes == before
        a.trim_cache(0)
        assert store.cache_bytes == 0


def test_shared_memory_oversized_and_concurrent(hierarchy):
    from concurrent.futures import ThreadPoolExecutor

    url, data = hierarchy
    with blosc2.RemoteStore(url, max_cache_bytes=1) as store:
        a, b = store["group/a"], store["group/b"]
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(lambda array: array[:], (a, b)))
        np.testing.assert_array_equal(results[0], data)
        np.testing.assert_array_equal(results[1], data + 1)
        assert store.cache_bytes == a.cache_bytes == b.cache_bytes == 0
        a.close()
        b.close()


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5])
def test_store_rejects_invalid_memory_limit(limit):
    with pytest.raises((ValueError, TypeError), match="positive integer"):
        blosc2.RemoteStore("memory://missing.b2z", max_cache_bytes=limit)


def test_store_none_rejects_limit():
    with pytest.raises(ValueError, match="not applicable"):
        blosc2.RemoteStore(
            "memory://missing.b2z", cache_policy=blosc2.CachePolicy.NONE, max_cache_bytes=None
        )


def test_disk_reopen_all_leaves_and_refresh(hierarchy, tmp_path, monkeypatch):
    url, data = hierarchy
    parent = tmp_path / "cache"
    with blosc2.RemoteStore(url, cache_dir=parent) as store:
        assert store.cache_policy is blosc2.CachePolicy.DISK
        with store["group/a"] as a, store["group/b"] as b:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            np.testing.assert_array_equal(b[:10, :10], data[:10, :10] + 1)
        retained = store.cache_bytes
        assert retained > 0
        assert store.metadata_bytes > 0
        with pytest.raises(RuntimeError, match="already owned"):
            blosc2.RemoteStore(url, cache_dir=parent)
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            raise AssertionError("reopen fetched remote bytes")

        patch.setattr(type(fsspec.filesystem("memory")), "cat_file", forbidden)
        reopened = blosc2.RemoteStore(url, cache_dir=parent)
        reopened.close()
    with blosc2.RemoteStore(url, cache_dir=parent) as store:
        # Partial B2Z blocks may have in-memory duplicates in the first session.
        assert 0 < store.cache_bytes <= retained
        assert len(store._owner.caches) == 2
        with store["group/a"] as a:
            before = store.traffic.nbytes
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            assert store.traffic.nbytes == before
        group = store["group"]
        a = group["a"]
        store.refresh()
        assert store.cache_bytes == 0
        with pytest.raises(RuntimeError, match="stale"):
            a[:1]
        with pytest.raises(RuntimeError, match="stale"):
            group.keys()
        a.close()
        group.close()
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:], data)
    with blosc2.RemoteStore(url, cache_dir=parent, max_cache_bytes=1) as store:
        assert store.cache_bytes == 0


def test_disk_lock_outlives_root(hierarchy, tmp_path):
    url, data = hierarchy
    root = blosc2.RemoteStore(url, cache_dir=tmp_path)
    leaf = root["group/a"]
    root.close()
    with pytest.raises(RuntimeError, match="already owned"):
        blosc2.RemoteStore(url, cache_dir=tmp_path)
    np.testing.assert_array_equal(leaf[:2], data[:2])
    leaf.close()
    with blosc2.RemoteStore(url, cache_dir=tmp_path):
        pass


def test_disk_manifest_rejects_corruption_and_releases_lock(hierarchy, tmp_path):
    import msgpack

    url, _ = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path) as store:
        path = store._owner.disk.path / "manifest.msgpack"
    value = msgpack.unpackb(path.read_bytes(), raw=False)
    value["caches"] = ["../escape"]
    path.write_bytes(msgpack.packb(value, use_bin_type=True))
    for _ in range(2):
        with pytest.raises(ValueError, match="Unsafe path"):
            blosc2.RemoteStore(url, cache_dir=tmp_path)


def test_disk_failed_refresh_and_portable_export(hierarchy, tmp_path, monkeypatch):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path, max_cache_bytes=None) as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            generation = store._owner.generation

            def fail(*args):
                raise OSError("manifest publication failed")

            with monkeypatch.context() as patch:
                patch.setattr(store._owner.disk, "publish", fail)
                with pytest.raises(OSError, match="publication failed"):
                    store.refresh()
            assert store._owner.generation == generation
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            exported = blosc2.from_cframe(a.to_cframe(include_cache=True))
    before = exported.traffic.nbytes
    np.testing.assert_array_equal(exported[:10, :10], data[:10, :10])
    assert exported.traffic.nbytes == before


def test_disk_lock_crash_release(tmp_path):
    import subprocess
    import sys

    from blosc2.remote_store_cache import StoreDiskCache

    code = """
import sys
from blosc2.remote_store_cache import StoreDiskCache
cache = StoreDiskCache(sys.argv[1], {"urlpath": "https://example.com/data.b2z"})
print("locked", flush=True)
sys.stdin.read()
"""
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="already owned"):
            StoreDiskCache(tmp_path, {"urlpath": "https://example.com/data.b2z"})
        process.kill()
        process.wait(timeout=10)
        cache = StoreDiskCache(tmp_path, {"urlpath": "https://example.com/data.b2z"})
        cache.close()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()


@pytest.fixture(params=["b2z", "h5", "zarr2", "zarr3"])
def hierarchy(request, tmp_path):
    backend = request.param
    data = np.arange(600, dtype="int32").reshape(30, 20)
    url = f"memory://remote-store/{backend}/hierarchy."
    fs = fsspec.filesystem("memory")
    if backend == "b2z":
        path = tmp_path / "hierarchy.b2z"
        with blosc2.TreeStore(path, mode="w", threshold=0) as root:
            root.attrs["title"] = "root"
            root.get_subtree("/group").attrs["title"] = "child"
            root.get_subtree("/group/empty").attrs["empty"] = True
            for name, values in (("a", data), ("b", data + 1)):
                root[f"/group/{name}"] = blosc2.asarray(values, chunks=(10, 10), blocks=(5, 5))
            root["/bad"] = blosc2.SChunk(data=b"unsupported")
        url += "b2z"
        fs.pipe_file(url, path.read_bytes())
    elif backend == "h5":
        h5py = pytest.importorskip("h5py")
        pytest.importorskip("kerchunk")
        path = tmp_path / "hierarchy.h5"
        with h5py.File(path, "w") as root:
            root.attrs["title"] = "root"
            group = root.create_group("group")
            group.attrs["title"] = "child"
            group.create_group("empty").attrs["empty"] = True
            group.create_dataset("a", data=data, chunks=(10, 10), compression="gzip")
            group.create_dataset("b", data=data + 1, chunks=(10, 10))
            root.create_dataset("bad", shape=None, dtype="i4")
        url += "h5"
        fs.pipe_file(url, path.read_bytes())
    else:
        zarr = pytest.importorskip("zarr")
        url += "zarr"
        root = zarr.open_group(url, mode="w", zarr_format=int(backend[-1]))
        root.attrs["title"] = "root"
        group = root.create_group("group")
        group.attrs["title"] = "child"
        group.create_group("empty").attrs["empty"] = True
        group.create_array("a", data=data, chunks=(10, 10))
        group.create_array("b", data=data + 1, chunks=(10, 10))
    return url, data


def test_discovery_aliases_sources_and_lifetime(hierarchy, tmp_path, monkeypatch):
    url, data = hierarchy
    translations = []
    if url.endswith(".h5"):
        import kerchunk.hdf

        translate = kerchunk.hdf.SingleHdf5ToZarr.translate

        def counted(self):
            translations.append(1)
            return translate(self)

        monkeypatch.setattr(kerchunk.hdf.SingleHdf5ToZarr, "translate", counted)

    with monkeypatch.context() as no_payload:

        def forbidden(*args, **kwargs):
            raise AssertionError("discovery constructed an array reader or cache")

        for cls in (blosc2.B2ZNDSource, blosc2.HDF5NDSource, blosc2.ZarrNDSource, blosc2.Proxy):
            no_payload.setattr(cls, "__init__", forbidden)
        root = blosc2.RemoteStore(url, cache_policy=blosc2.CachePolicy.NONE)
        owner = root._owner
        assert "group" in list(root)
        assert root.attrs == {"title": "root"}
        assert root.get_info("group").attrs == {"title": "child"}
        assert root.kind("group/empty") == "group"
        assert owner.sources == {}
    assert root.cache_policy is blosc2.CachePolicy.NONE
    assert root.cache_bytes == 0
    assert root.max_cache_bytes is None
    leaf = root["group/a"]
    group = root["group"]
    alias = group["a"]
    assert isinstance(leaf, blosc2.RemoteArray)
    assert leaf.src is alias.src
    assert leaf.traffic is alias.traffic is group.traffic is root.traffic
    assert leaf.source == alias.source
    assert group.keys() == ["a", "b", "empty"]
    with group["empty"] as empty:
        assert empty.keys() == []
        assert empty.attrs == {"empty": True}
    with pytest.raises(KeyError):
        root["missing"]
    if "bad" in root:
        assert root.get_info("bad").kind == "unsupported"
        assert root.get_info("bad").diagnostic
        with pytest.raises(NotImplementedError):
            root["bad"]
    for path in ("../group", "group/../a", "group//a", "group\\a"):
        with pytest.raises(ValueError):
            root[path]
    with pytest.raises(TypeError):
        root[1]
    with pytest.raises(TypeError):
        root.attrs["title"] = "changed"
    with group["b"] as other:
        np.testing.assert_array_equal(other[:2, :3], data[:2, :3] + 1)
        if owner.format == "b2z":
            assert other.src._archive is leaf.src._archive is owner.archive
            with pytest.raises(ValueError, match="does not match its archive"):
                blosc2.B2ZNDSource("memory://wrong.b2z", "group/a", _archive=owner.archive)
        elif owner.format == "hdf5":
            assert other.src._refs is leaf.src._refs is owner.refs
            assert translations == [1]
        else:
            assert other.src.array.store is leaf.src.array.store is owner.zstore
    np.testing.assert_array_equal(leaf[:2, :3], data[:2, :3])
    before = root.traffic.nbytes
    np.testing.assert_array_equal(alias[:2, :3], data[:2, :3])
    assert root.traffic.nbytes > before  # NONE fetches again.
    assert leaf.cache is None
    assert leaf.cache_bytes == root.cache_bytes == 0
    np.testing.assert_array_equal((leaf + 2)[:], data + 2)

    export = tmp_path / "leaf.b2nd"
    leaf.save(export)
    np.testing.assert_array_equal(blosc2.open(export)[:2, :3], data[:2, :3])
    if owner.format == "hdf5":
        assert translations == [1]  # The standalone export contains its own refs.

    root.close()
    root.close()
    with pytest.raises(RuntimeError, match="closed"):
        root.keys()
    assert group.attrs == {"title": "child"}
    group.close()
    leaf.close()
    with pytest.raises(RuntimeError, match="closed"):
        leaf[:]
    with pytest.raises(RuntimeError, match="closed"):
        _ = leaf.shape
    assert not owner._closed
    np.testing.assert_array_equal(alias[:2, :3], data[:2, :3])
    alias.close()
    assert owner._closed
    if owner.archive is not None:
        assert owner.archive.file.closed


def test_subgroup_and_garbage_collection(hierarchy):
    url, data = hierarchy
    root = blosc2.RemoteStore(url, dataset="group")
    assert root.source["dataset"] == "group"
    assert root.keys() == ["a", "b", "empty"]
    leaf = root["a"]
    owner = root._owner
    root_ref = weakref.ref(root)
    del root
    gc.collect()
    assert root_ref() is None
    assert not owner._closed
    np.testing.assert_array_equal(leaf[:2, :3], data[:2, :3])
    del leaf
    gc.collect()
    assert owner._closed


def test_zarr_direct_lookup_without_listing(hierarchy, monkeypatch):
    url, data = hierarchy
    if not url.endswith(".zarr"):
        pytest.skip("Zarr-specific listing")
    import zarr

    async def denied(self, prefix):
        raise PermissionError("LIST denied")
        yield

    monkeypatch.setattr(zarr.storage.FsspecStore, "list_dir", denied)
    with blosc2.RemoteStore(url) as store:
        with store["group/a"] as leaf:
            np.testing.assert_array_equal(leaf[:2, :3], data[:2, :3])
        with pytest.raises(OSError, match="LIST permission"):
            store.keys()


def test_zarr_unsupported_codec():
    zarr = pytest.importorskip("zarr")
    url = "memory://remote-store/unsupported.zarr"
    root = zarr.open_group(url, mode="w", zarr_format=2)
    root.create_array("bad", data=np.arange(10))
    fs = fsspec.filesystem("memory")
    key = url + "/bad/.zarray"
    metadata = json.loads(fs.cat_file(key))
    metadata["compressor"] = {"id": "no-such-codec"}
    fs.pipe_file(key, json.dumps(metadata).encode())
    with blosc2.RemoteStore(url) as store:
        assert store.keys() == ["bad"]
        assert store.kind("bad") == "unsupported"
        assert "codec" in store.get_info("bad").diagnostic
        with pytest.raises(NotImplementedError, match="codec"):
            store["bad"]


def test_store_validation():
    with pytest.raises(TypeError, match="dataset must be a string"):
        blosc2.RemoteStore("memory://a.b2z", dataset=1)
    with pytest.raises(ValueError, match="cache_dir"):
        blosc2.RemoteStore("memory://a.b2z", cache_policy=blosc2.CachePolicy.DISK)
    with pytest.raises(TypeError, match="CachePolicy"):
        blosc2.RemoteStore("memory://a.b2z", cache_policy="none")
    with pytest.raises(ValueError, match="user information"):
        blosc2.RemoteStore("https://user:password@example.com/a.b2z")
