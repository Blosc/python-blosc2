"""Public remote discovery, shared readers and dependent handle lifetime."""

import gc
import json
import weakref

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


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
        root = blosc2.RemoteStore(url)
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
    with pytest.raises(NotImplementedError, match="NONE only"):
        blosc2.RemoteStore("memory://a.b2z", cache_policy=blosc2.CachePolicy.MEMORY)
    with pytest.raises(TypeError, match="CachePolicy"):
        blosc2.RemoteStore("memory://a.b2z", cache_policy="none")
    with pytest.raises(ValueError, match="user information"):
        blosc2.RemoteStore("https://user:password@example.com/a.b2z")
