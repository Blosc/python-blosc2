"""Public remote discovery, shared readers and dependent handle lifetime."""

import gc
import json
import sys
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
        b2d_dirs = list(parent.glob("**/*.b2d"))
        assert len(b2d_dirs) == 1
        assert b2d_dirs[0].name == f"{store._owner.generation}.b2d"
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


def test_disk_cache_partitions_storage_options(hierarchy, tmp_path):
    url, data = hierarchy
    cache = tmp_path / "cache"

    for endpoint in ("one", "two"):
        with blosc2.RemoteStore(url, cache_dir=cache, storage_options={"endpoint": endpoint}) as store:
            with store["group/a"] as a:
                np.testing.assert_array_equal(a[:2, :2], data[:2, :2])

    # Different backends must not share a manifest or its leaf payloads.
    assert len([path for path in cache.iterdir() if path.is_dir()]) == 2


def test_artifact_keeps_storage_options_identity(hierarchy, tmp_path):
    url, data = hierarchy
    artifact = tmp_path / "warm.b2z"
    with blosc2.RemoteStore(url, storage_options={"endpoint": "one"}) as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:2, :2], data[:2, :2])
        store.save(artifact)

    # The access-configuration fingerprint is recorded, not required: a warm
    # artifact stays portable and reopens without live credentials.
    with blosc2.open(artifact) as restored:
        with restored["group/a"] as a:
            np.testing.assert_array_equal(a[:2, :2], data[:2, :2])


def test_mutable_artifact_refresh_preserves_runtime_storage(hierarchy, tmp_path):
    url, data = hierarchy
    artifact = tmp_path / "refresh.b2z"
    with blosc2.RemoteStore(url) as store:
        store.save(artifact, mutable=True)
    with blosc2.open(artifact, max_cache_bytes=4096) as store:
        store.mutable = True
        cache_root = store._owner.disk.path
        cleanup = store._owner._cleanup_dir
        store.refresh()
        assert cache_root.exists()
        assert store._owner._cleanup_dir is cleanup
        assert store.mutable is True
        manifest = store._owner.disk.load()
        assert manifest["max_cache_bytes"] == 4096
        assert manifest["mutable"] is True
        with pytest.raises(ValueError, match="source artifact"):
            store.save(artifact, overwrite=True)
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:], data)
    assert not cache_root.exists()


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
    import json

    url, _ = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path) as store:
        active_path = store._owner.disk.path / "active_generation.json"
        gen = json.loads(active_path.read_text(encoding="utf-8"))["generation"]
        embed_path = store._owner.disk.path / f"{gen}.b2d" / "embed.b2e"
    carrier = blosc2.blosc2_ext.open(str(embed_path), "a", 0)
    manifest = dict(carrier.vlmeta["b2remote_manifest"])
    manifest["caches"] = ["../escape"]
    carrier.vlmeta["b2remote_manifest"] = manifest
    del carrier
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


def test_sparse_store_shared_handles(hierarchy, tmp_path):
    url, data = hierarchy
    parent = tmp_path / "shared"
    with blosc2.RemoteStore.with_sparse_cache(url, parent) as first:
        with blosc2.RemoteStore.with_sparse_cache(url, parent) as second:
            with first["group/a"] as a, second["group/a"] as b:
                np.testing.assert_array_equal(a[:], data)
                second.traffic.reset()
                np.testing.assert_array_equal(b[:], data)
                assert second.traffic.requests == 0
                assert first.cache_bytes == second.cache_bytes > 0
                assert a._proxy._cache.schunk.contiguous is False
            second.refresh()
            with pytest.raises(RuntimeError, match="stale"):
                first.keys()


def test_sparse_store_trim_export_recovery(hierarchy, tmp_path):
    url, data = hierarchy
    parent = tmp_path / "shared"
    with blosc2.RemoteStore.with_sparse_cache(url, parent) as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:], data)
        store.save(tmp_path / "warm.b2z")
        store.save(tmp_path / "cold.b2z", include_cache=False)
        source = store._owner.disk.source
        evicted, remaining = blosc2.RemoteStore.trim_sparse_cache(parent, source, 0)
        assert evicted
        assert remaining == 0
        assert store.cache_bytes == 0
    with blosc2.open(tmp_path / "warm.b2z") as restored:
        with restored["group/a"] as a:
            np.testing.assert_array_equal(a[:], data)


def test_sparse_store_reopen_reuses_payload(hierarchy, tmp_path):
    url, data = hierarchy
    for repeat in range(3):
        with blosc2.RemoteStore.with_sparse_cache(url, tmp_path / "cache") as store:
            with store["group/a"] as array:
                store.traffic.reset()
                np.testing.assert_array_equal(array[:], data)
                if repeat:
                    assert store.traffic.requests == 0


def test_sparse_store_imports_warm_artifact(hierarchy, tmp_path):
    url, data = hierarchy
    artifact = tmp_path / "seed.b2z"
    with blosc2.RemoteStore(url) as store:
        with store["group/a"] as array:
            np.testing.assert_array_equal(array[:], data)
        store.save(artifact)
    with blosc2.RemoteStore.with_sparse_cache(url, tmp_path / "cache", carrier=artifact) as store:
        with store["group/a"] as array:
            store.traffic.reset()
            np.testing.assert_array_equal(array[:], data)
            assert store.traffic.requests == 0


def test_sparse_store_missing_key_preserves_handles(hierarchy, tmp_path):
    url, data = hierarchy
    with blosc2.RemoteStore.with_sparse_cache(url, tmp_path / "cache") as store:
        with store["group/a"] as array:
            with pytest.raises(KeyError):
                store["missing"]
            np.testing.assert_array_equal(array[:], data)


def _shared_fs():
    from fsspec.implementations.local import LocalFileSystem

    fs = LocalFileSystem(skip_instance_cache=True)
    fs._strip_protocol = lambda path: LocalFileSystem._strip_protocol(
        str(path).removeprefix("https://fixture.example")
    )
    return fs


def _shared_reader(url, cache, barrier, results):
    try:
        with blosc2.RemoteStore.with_sparse_cache(
            url, cache, max_cache_bytes=1 << 20, _filesystem=_shared_fs()
        ) as store:
            with store["a"] as array:
                barrier.wait(timeout=30)
                np.testing.assert_array_equal(array[:], np.arange(10000, dtype="i4"))
                barrier.wait(timeout=30)
                store.traffic.reset()
                np.testing.assert_array_equal(array[:], np.arange(10000, dtype="i4"))
                results.put((store.traffic.requests, store.cache_bytes))
    except BaseException as exc:
        results.put(repr(exc))


def _shared_crash(url, cache):
    import os

    store = blosc2.RemoteStore.with_sparse_cache(url, cache, _filesystem=_shared_fs())
    array = store["a"]
    original = blosc2.Proxy._store_chunk

    def die(self, *args):
        original(self, *args)
        os._exit(17)

    blosc2.Proxy._store_chunk = die
    array[:]
    os._exit(18)


def test_sparse_store_processes_and_crash(tmp_path):
    import multiprocessing

    source = tmp_path / "source.b2z"
    with blosc2.TreeStore(source, mode="w", threshold=0) as tree:
        tree["a"] = blosc2.asarray(np.arange(10000, dtype="i4"), chunks=(1000,), blocks=(1000,))
    url, cache = "https://fixture.example" + str(source), str(tmp_path / "shared")
    ctx = multiprocessing.get_context("spawn")
    barrier, results = ctx.Barrier(4), ctx.Queue()
    workers = [ctx.Process(target=_shared_reader, args=(url, cache, barrier, results)) for _ in range(4)]
    for worker in workers:
        worker.start()
    try:
        answers = [results.get(timeout=60) for _ in workers]
        assert all(isinstance(answer, tuple) and answer[0] == 0 and answer[1] > 0 for answer in answers), (
            answers
        )
        for worker in workers:
            worker.join(timeout=10)
            assert worker.exitcode == 0
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join()
        results.close()
    crash_cache = str(tmp_path / "crash")
    worker = ctx.Process(target=_shared_crash, args=(url, crash_cache))
    worker.start()
    worker.join(timeout=30)
    if worker.is_alive():
        worker.terminate()
        worker.join()
    assert worker.exitcode == 17
    with (
        blosc2.RemoteStore.with_sparse_cache(url, crash_cache, _filesystem=_shared_fs()) as store,
        store["a"] as array,
    ):
        np.testing.assert_array_equal(array[:], np.arange(10000, dtype="i4"))


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


def test_discovery_node_limit(hierarchy, tmp_path):
    url, _ = hierarchy
    with pytest.raises(ValueError, match="node limit"):
        with blosc2.RemoteStore(url, _max_nodes=1) as store:
            store.keys()  # Zarr discovers children lazily.
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "cache") as store:
        store.keys()
    with pytest.raises(ValueError, match="node limit"):
        blosc2.RemoteStore(url, cache_dir=tmp_path / "cache", _max_nodes=1)


@pytest.mark.parametrize("key", ["../escape", "/absolute", "group/../../escape", "group\\escape"])
def test_cache_payload_path_rejects_unsafe_keys(tmp_path, key):
    from blosc2.remote_store_cache import StoreDiskCache

    cache = StoreDiskCache(tmp_path, {"urlpath": "https://example.com/data.b2z"})
    try:
        with pytest.raises(ValueError, match="Unsafe path"):
            cache.payload_path("a" * 32, key)
        assert not (cache.path / f"{'a' * 32}.b2d").exists()
    finally:
        cache.close()


def test_save_and_reopen_immutable_and_mutable(hierarchy, tmp_path):
    import hashlib

    url, data = hierarchy
    live_cache = tmp_path / "live_cache"
    with blosc2.RemoteStore(url, cache_dir=live_cache) as store:
        assert store.mutable is False
        assert store.is_cache_mutable is True
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        assert store.cache_bytes > 0
        snapshot_imm = tmp_path / "snapshot_imm.b2z"
        snapshot_mut = tmp_path / "snapshot_mut.b2z"
        snapshot_cold = tmp_path / "snapshot_cold.b2z"
        store.save(snapshot_imm)
        store.save(snapshot_mut, mutable=True)
        store.save(snapshot_cold, include_cache=False)

    # 1. Test immutable reopen
    with blosc2.open(snapshot_imm) as restored:
        assert isinstance(restored, blosc2.RemoteStore)
        assert restored.mutable is False
        assert restored.is_cache_mutable is False
        assert restored.cache_bytes > 0
        with restored["group/a"] as a:
            restored.traffic.reset()
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            assert restored.traffic.requests == 0
            assert a.get_chunk(0)
            assert restored.traffic.requests == 0
            before_bytes = restored.cache_bytes
            assert a.get_chunk(2)
            assert not a.cache_contains(nchunk=2)
            with pytest.raises(RuntimeError, match="immutable"):
                a.fetch()
            with pytest.raises(RuntimeError, match="immutable"):
                a.trim_cache(0)
            np.testing.assert_array_equal(a[10:20, :10], data[10:20, :10])
            assert restored.cache_bytes == before_bytes
        with pytest.raises(ValueError, match="immutable"):
            restored.refresh()

    # Verify read-only permissions (chmod 0o444) and byte preservation
    snapshot_imm.chmod(0o444)
    sha_before = hashlib.sha256(snapshot_imm.read_bytes()).hexdigest()
    with blosc2.open(snapshot_imm) as ro_store:
        with ro_store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            np.testing.assert_array_equal(a[10:20, :10], data[10:20, :10])
    sha_after = hashlib.sha256(snapshot_imm.read_bytes()).hexdigest()
    assert sha_after == sha_before
    snapshot_imm.chmod(0o644)

    # 2. Test budget rejection on immutable snapshot
    with pytest.raises(ValueError, match="smaller than retained immutable payload"):
        blosc2.open(snapshot_imm, max_cache_bytes=10)
    with pytest.raises(ValueError, match=r"CachePolicy\.NONE"):
        blosc2.open(snapshot_imm, cache_policy=blosc2.CachePolicy.NONE)

    # 3. Test mutable reopen
    mut_sha_before = hashlib.sha256(snapshot_mut.read_bytes()).hexdigest()
    with blosc2.open(snapshot_mut) as restored_mut:
        assert isinstance(restored_mut, blosc2.RemoteStore)
        assert restored_mut.mutable is False  # export default is False
        assert restored_mut.is_cache_mutable is True
        with restored_mut["group/a"] as a:
            restored_mut.traffic.reset()
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            assert restored_mut.traffic.requests == 0
            before_cbytes = restored_mut.cache_bytes
            np.testing.assert_array_equal(a[10:20, :10], data[10:20, :10])
            mut_cbytes = restored_mut.cache_bytes
            assert mut_cbytes > before_cbytes
    # Archive file on disk remains unchanged
    mut_sha_after = hashlib.sha256(snapshot_mut.read_bytes()).hexdigest()
    assert mut_sha_after == mut_sha_before

    # 4. Test mutable trim on smaller requested allowance
    with blosc2.open(snapshot_mut, max_cache_bytes=mut_cbytes // 2) as trimmed:
        assert trimmed.cache_bytes <= mut_cbytes // 2
        assert trimmed.max_cache_bytes == mut_cbytes // 2

    # 5. Test cold reopen
    with blosc2.open(snapshot_cold) as restored_cold:
        assert restored_cold.cache_bytes == 0
        with restored_cold["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])


def test_subtree_export(hierarchy, tmp_path):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        group = store["group"]
        subtree_path = tmp_path / "subtree.b2z"
        group.save(subtree_path)

    with blosc2.open(subtree_path) as sub:
        assert isinstance(sub, blosc2.RemoteStore)
        assert set(sub.keys()) >= {"a", "b", "empty"}
        assert sub.attrs.get("title") == "child"
        assert sub["empty"].attrs.get("empty") is True
        sub.traffic.reset()
        np.testing.assert_array_equal(sub["a"][:10, :10], data[:10, :10])
        assert sub.traffic.requests == 0
        np.testing.assert_array_equal(sub["b"][:10, :10], data[:10, :10] + 1)


def test_hdf5_single_ref_map_preserved(hierarchy, tmp_path, monkeypatch):
    url, data = hierarchy
    if not url.endswith(".h5"):
        pytest.skip("HDF5 specific test")
    import kerchunk.hdf

    translations = []
    orig_translate = kerchunk.hdf.SingleHdf5ToZarr.translate

    def counted(self):
        translations.append(1)
        return orig_translate(self)

    monkeypatch.setattr(kerchunk.hdf.SingleHdf5ToZarr, "translate", counted)
    snapshot = tmp_path / "h5_snap.b2z"
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        store.save(snapshot)
    assert len(translations) == 1

    translations.clear()
    with blosc2.open(snapshot) as restored:
        with restored["group/a"] as a, restored["group/b"] as b:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            np.testing.assert_array_equal(b[:10, :10], data[:10, :10] + 1)
            assert a.src._refs is b.src._refs is restored._owner.refs
    assert len(translations) == 0


def test_mutability_property_and_inheritance(hierarchy):
    url, _ = hierarchy
    with blosc2.RemoteStore(url) as store:
        assert store.mutable is False
        assert store["group"].mutable is False
        assert store["group/a"].mutable is False

        store.mutable = True
        assert store.mutable is True
        assert store["group"].mutable is True
        assert store["group/a"].mutable is True

        store["group"].mutable = False
        assert store.mutable is False
        assert store["group"].mutable is False
        assert store["group/a"].mutable is False

        with pytest.raises(TypeError, match="boolean"):
            store.mutable = "invalid"


def test_save_destination_and_budget_validation(hierarchy, tmp_path):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        with pytest.raises(ValueError, match=r"\.b2z extension"):
            store.save(tmp_path / "not_b2z.txt")
        dir_b2z = tmp_path / "somedir.b2z"
        dir_b2z.mkdir()
        with pytest.raises(ValueError, match="directory"):
            store.save(dir_b2z)
        target = tmp_path / "exist.b2z"
        target.write_text("x")
        with pytest.raises(FileExistsError):
            store.save(target)
        store.save(target, overwrite=True)
        assert target.exists()

        with pytest.raises(ValueError, match="inside live cache"):
            store.save(store._owner.disk.path / "nested.b2z")

        store._owner.max_cache_bytes = 10
        with pytest.raises(ValueError, match="exceeds max_cache_bytes"):
            store.save(tmp_path / "exceeded.b2z")


def test_memory_store_save_and_reopen(hierarchy, tmp_path):
    """MEMORY is the default policy; every backend must export through it."""
    import os

    url, data = hierarchy
    with blosc2.RemoteStore(url) as store:
        assert store.cache_policy is blosc2.CachePolicy.MEMORY
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        retained = store.cache_bytes
        assert retained > 0

        warm = tmp_path / "mem_warm.b2z"
        writable = tmp_path / "mem_mut.b2z"
        cold = tmp_path / "mem_cold.b2z"
        assert store.save(warm) == os.path.abspath(warm)
        store.save(writable, mutable=True)
        store.save(cold, include_cache=False)
        # A cold export references discovery metadata only and must not clear
        # the live cache it was taken from.
        assert store.cache_bytes == retained

    with blosc2.open(warm) as restored:
        assert restored.is_cache_mutable is False
        assert restored.cache_bytes > 0
        with restored["group/a"] as a:
            restored.traffic.reset()
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            assert restored.traffic.requests == 0
            before = restored.cache_bytes
            np.testing.assert_array_equal(a[20:30, :10], data[20:30, :10])
            assert restored.cache_bytes == before  # misses are transient

    with blosc2.open(writable) as restored:
        assert restored.is_cache_mutable is True
        with restored["group/a"] as a:
            restored.traffic.reset()
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            assert restored.traffic.requests == 0


def test_cold_artifact_reopens_under_none(hierarchy, tmp_path):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        cold = tmp_path / "cold.b2z"
        warm = tmp_path / "warm.b2z"
        store.save(cold, include_cache=False)
        store.save(warm)

    with blosc2.open(cold, cache_policy=blosc2.CachePolicy.NONE) as restored:
        assert restored.cache_policy is blosc2.CachePolicy.NONE
        assert restored.cache_bytes == 0
        with restored["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])

    with pytest.raises(ValueError, match=r"warm RemoteStore artifact with CachePolicy\.NONE"):
        blosc2.open(warm, cache_policy=blosc2.CachePolicy.NONE)


def test_artifact_reopen_modes_and_policies(hierarchy, tmp_path):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        immutable = tmp_path / "immutable.b2z"
        writable = tmp_path / "writable.b2z"
        store.save(immutable)
        store.save(writable, mutable=True)

    with pytest.raises(ValueError, match="only support modes"):
        blosc2.open(immutable, mode="w")
    with pytest.raises(ValueError, match="read-only"):
        blosc2.open(immutable, mode="a")
    with pytest.raises(ValueError, match=r"require CachePolicy\.DISK"):
        blosc2.open(writable, cache_policy=blosc2.CachePolicy.MEMORY)

    # Neither artifact may be saved over itself, mutable or not.
    with blosc2.open(immutable) as restored:
        with pytest.raises(ValueError, match="source artifact"):
            restored.save(immutable, overwrite=True)
    with blosc2.open(writable) as restored:
        with pytest.raises(ValueError, match="source artifact"):
            restored.save(writable, overwrite=True)


def _rewrite_artifact(path, *, compress=False, mutate_manifest=None):
    """Rewrite a .b2z artifact, optionally deflating its leaves or editing its manifest."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        members = {info.filename: zf.read(info.filename) for info in zf.infolist()}
    embed_bytes = members.pop("embed.b2e")
    if mutate_manifest is not None:
        embed_path = path.parent / "embed.b2e"
        embed_path.write_bytes(embed_bytes)
        embed = blosc2.blosc2_ext.open(str(embed_path), "a", 0)
        manifest = dict(embed.vlmeta["b2remote_manifest"])
        mutate_manifest(manifest)
        embed.vlmeta["b2remote_manifest"] = manifest
        del embed
        members["embed.b2e"] = embed_path.read_bytes()
        embed_path.unlink()
    else:
        members["embed.b2e"] = embed_bytes
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in members.items():
            # embed.b2e stays stored so dispatch still reaches _open_artifact.
            deflated = compress and name != "embed.b2e"
            zf.writestr(
                name,
                content,
                compress_type=zipfile.ZIP_DEFLATED if deflated else zipfile.ZIP_STORED,
            )


@pytest.mark.parametrize("bad_generation", ["absolute", "../escaped", "x" * 32, None])
def test_artifact_generation_validated_before_staging(hierarchy, tmp_path, bad_generation):
    url, _ = hierarchy
    artifact = tmp_path / "generation.b2z"
    with blosc2.RemoteStore(url) as store:
        store.save(artifact, mutable=True)
    if bad_generation == "absolute":
        bad_generation = str(tmp_path / "escaped")
    _rewrite_artifact(artifact, mutate_manifest=lambda m: m.update(generation=bad_generation))
    cache_dir = tmp_path / "destination"
    with pytest.raises(ValueError, match="generation"):
        blosc2.open(artifact, cache_dir=cache_dir)
    assert not cache_dir.exists()
    assert not (tmp_path / "escaped.b2d").exists()


def test_artifact_extraction_failure_releases_owner(hierarchy, tmp_path, monkeypatch):
    import zipfile

    url, _ = hierarchy
    artifact = tmp_path / "extract.b2z"
    cache_dir = tmp_path / "staged"
    with blosc2.RemoteStore(url) as store:
        store.save(artifact, mutable=True)

    def fail(*args, **kwargs):
        raise OSError("extraction failed")

    with monkeypatch.context() as patch:
        patch.setattr(zipfile.ZipFile, "extractall", fail)
        with pytest.raises(OSError, match="extraction failed"):
            blosc2.open(artifact, cache_dir=cache_dir)
    assert not list(cache_dir.rglob("active_generation.json"))
    with blosc2.open(artifact, cache_dir=cache_dir) as reopened:
        assert reopened.is_cache_mutable


def test_artifact_manifest_and_member_validation(hierarchy, tmp_path):
    import shutil

    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
        artifact = tmp_path / "artifact.b2z"
        store.save(artifact)

    def inject_credentials(manifest):
        manifest["source"] = dict(manifest["source"])
        manifest["source"]["urlpath"] = "https://user:pass@example.com/x.b2z"

    bad_url = tmp_path / "bad_url.b2z"
    shutil.copy2(artifact, bad_url)
    _rewrite_artifact(bad_url, mutate_manifest=inject_credentials)
    with pytest.raises(ValueError, match="user information"):
        blosc2.open(bad_url)

    compressed = tmp_path / "compressed.b2z"
    shutil.copy2(artifact, compressed)
    _rewrite_artifact(compressed, compress=True)
    with pytest.raises(ValueError, match="ZIP_STORED"):
        blosc2.open(compressed)


def test_save_failure_preserves_destination_and_live_cache(hierarchy, tmp_path, monkeypatch):
    url, data = hierarchy
    with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])

        def fail(self, *args, **kwargs):
            raise OSError("export failed")

        destination = tmp_path / "out.b2z"
        destination.write_bytes(b"sentinel")
        with monkeypatch.context() as patch:
            patch.setattr(blosc2.RemoteStore, "_copy_leaf_carrier", fail)
            with pytest.raises(OSError, match="export failed"):
                store.save(destination, overwrite=True)
        assert destination.read_bytes() == b"sentinel"
        assert not [p for p in tmp_path.iterdir() if p.name.startswith("b2z-export-")]

        fresh = tmp_path / "fresh.b2z"
        with monkeypatch.context() as patch:
            patch.setattr(blosc2.RemoteStore, "_copy_leaf_carrier", fail)
            with pytest.raises(OSError, match="export failed"):
                store.save(fresh)
        assert not fresh.exists()

        # The live store is still usable after a failed export.
        with store["group/a"] as a:
            np.testing.assert_array_equal(a[:10, :10], data[:10, :10])


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="in-process HTTP servers not supported on Windows",
)
def test_artifact_reopens_in_fresh_process(tmp_path):
    """A locally transported .b2z must be readable by another interpreter."""
    import functools
    import hashlib
    import http.server
    import subprocess
    import threading

    data = np.arange(600, dtype="int32").reshape(30, 20)
    served = tmp_path / "served"
    served.mkdir()
    tree = served / "hierarchy.b2z"
    with blosc2.TreeStore(tree, mode="w", threshold=0) as root:
        root["/group/a"] = blosc2.asarray(data, chunks=(10, 10), blocks=(5, 5))

    class Ranged(http.server.SimpleHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def do_GET(self):
            span = self.headers.get("Range")
            if not span:
                return super().do_GET()
            body = (served / self.path.lstrip("/")).read_bytes()
            first, _, last = span.removeprefix("bytes=").partition("-")
            first, last = int(first), int(last) if last else len(body) - 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(last - first + 1))
            self.send_header("ETag", hashlib.sha256(body).hexdigest())
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body[first : last + 1])
            return None

        def do_HEAD(self):
            body = (served / self.path.lstrip("/")).read_bytes()
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", hashlib.sha256(body).hexdigest())
            self.send_header("Connection", "close")
            self.end_headers()

    handler = functools.partial(Ranged, directory=str(served))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/{tree.name}"
        with blosc2.RemoteStore(url, cache_dir=tmp_path / "live") as store:
            with store["group/a"] as a:
                np.testing.assert_array_equal(a[:10, :10], data[:10, :10])
            artifact = tmp_path / "snapshot.b2z"
            store.save(artifact)
        script = (
            "import sys\n"
            "import numpy as np\n"
            "import blosc2\n"
            "with blosc2.open(sys.argv[1]) as store:\n"
            "    with store['group/a'] as a:\n"
            "        print(int(np.asarray(a[:10, :10]).sum()))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(artifact)],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
    finally:
        server.shutdown()
        server.server_close()
    assert int(result.stdout.strip()) == int(data[:10, :10].sum())
