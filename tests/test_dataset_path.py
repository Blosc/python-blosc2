"""Public node selectors remain aliases, not new source identities."""

import dataclasses

import numpy as np
import pytest

import blosc2

fsspec = pytest.importorskip("fsspec")


@pytest.fixture
def container(tmp_path):
    @dataclasses.dataclass
    class Row:
        x: int

    filename = tmp_path / "selectors.b2z"
    with blosc2.TreeStore(filename, mode="w", threshold=0) as tree:
        tree["group/array"] = blosc2.arange(8)
        tree["group/table"] = blosc2.CTable(Row, [(1,), (2,)], create_summary_index=False)
    url = f"memory://{tmp_path.name}/selectors.b2z"
    fsspec.filesystem("memory").pipe(url, filename.read_bytes())
    return filename, url


@pytest.mark.parametrize("api", ["open", "array", "store", "table", "sparse_store", "sparse_table"])
def test_path_alias(container, tmp_path, api):
    filename, url = container
    target = "group/array" if api in {"open", "array"} else "group/table" if "table" in api else "group"
    opener = {
        "open": blosc2.open,
        "array": blosc2.RemoteArray,
        "store": blosc2.RemoteStore,
        "table": blosc2.RemoteCTable,
        "sparse_store": blosc2.RemoteStore.with_sparse_cache,
        "sparse_table": blosc2.RemoteCTable.with_sparse_cache,
    }[api]
    args = (url, tmp_path / "sparse") if api.startswith("sparse") else (url,)
    for options in ({"path": target}, {"dataset": target}, {"path": f"/{target}/", "dataset": target}):
        with opener(*args, **options) as obj:
            if api in {"open", "array"}:
                np.testing.assert_array_equal(obj[:], np.arange(8))
                assert obj.dataset == target
            elif "table" in api:
                np.testing.assert_array_equal(obj["x"][:], [1, 2])
            else:
                assert obj.keys() == ["array", "table"]
    with pytest.raises(ValueError, match="Conflicting dataset and path"):
        opener(*args, path=target, dataset="different")
    with pytest.raises(TypeError, match="path must be a string"):
        opener(*args, path=123)
    for path in (target, "", "/"):
        with pytest.raises(ValueError, match="both URL path and dataset"):
            opener(url + "::" + target, *args[1:], path=path)
    if api == "open":
        with blosc2.open(filename, path=target) as obj:
            np.testing.assert_array_equal(obj[:], np.arange(8))
        # The old positional dataset argument keeps its meaning.
        with blosc2.open(url, "r", 0, target) as obj:
            np.testing.assert_array_equal(obj[:], np.arange(8))


@pytest.mark.parametrize(
    "options", [{"path": None}, {"path": ""}, {"path": "/"}, {"path": "/", "dataset": ""}]
)
def test_path_root(container, options):
    _, url = container
    for opener in (blosc2.open, blosc2.RemoteStore):
        with opener(url, **options) as store:
            assert store.keys() == ["group"]


def test_path_cache_and_artifact_compatibility(container, tmp_path):
    _, url = container
    cache = tmp_path / "cache"
    artifact = tmp_path / "reference.b2nd"
    with blosc2.open(url, path="group/array", cache_dir=cache) as array:
        np.testing.assert_array_equal(array[:], np.arange(8))
        source = array._source.copy()
        array.save(artifact)
    with blosc2.open(url, dataset="group/array", cache_dir=cache) as array:
        assert array._source == source
        np.testing.assert_array_equal(array[:], np.arange(8))
        assert array.traffic.requests == 0
    with blosc2.open(artifact) as array:
        assert array._source == source
        np.testing.assert_array_equal(array[:], np.arange(8))


def test_path_b2z_source(container):
    _, url = container
    source = blosc2.B2ZNDSource(url, path="/group/array/", dataset="group/array")
    np.testing.assert_array_equal(blosc2.Proxy(source)[:], np.arange(8))
    with pytest.raises(ValueError, match="Conflicting dataset and path"):
        blosc2.B2ZNDSource(url, path="group/array", dataset="different")


@pytest.mark.parametrize("format", ["hdf5", "zarr2", "zarr3"])
def test_path_other_formats(tmp_path, format):
    data = np.arange(8)
    if format == "hdf5":
        h5py = pytest.importorskip("h5py")
        filename = tmp_path / "selectors.h5"
        with h5py.File(filename, "w") as file:
            file.create_dataset("group/array", data=data, chunks=(4,))
        index = blosc2.scan_hdf5_index(filename, path="group")
        assert index == blosc2.scan_hdf5_index(filename, dataset="group")
        assert blosc2.validate_hdf5_index(index, path="group") is index
        source = blosc2.HDF5NDSource(filename, path="group/array")
        try:
            np.testing.assert_array_equal(blosc2.Proxy(source)[:], data)
        finally:
            source.close()
        for opener, args in (
            (blosc2.HDF5NDSource, (filename,)),
            (blosc2.scan_hdf5_index, (filename,)),
            (blosc2.validate_hdf5_index, (index,)),
        ):
            with pytest.raises(ValueError, match="Conflicting dataset and path"):
                opener(*args, path="group/array", dataset="different")
    else:
        zarr = pytest.importorskip("zarr")
        filename = tmp_path / "selectors.zarr"
        group = zarr.open_group(filename, mode="w", zarr_format=int(format[-1]))
        group.create_group("group").create_array("array", data=data, chunks=(4,))
    with blosc2.open(filename, path="group/array") as array:
        np.testing.assert_array_equal(array[:], data)
    url = f"memory://{tmp_path.name}/{filename.name}"
    fs = fsspec.filesystem("memory")
    if filename.is_file():
        fs.pipe(url, filename.read_bytes())
    else:
        for entry in filename.rglob("*"):
            if entry.is_file():
                fs.pipe(url + "/" + entry.relative_to(filename).as_posix(), entry.read_bytes())
    for opener in (blosc2.open, blosc2.RemoteArray):
        with opener(url, path="/group/array/", dataset="group/array") as array:
            np.testing.assert_array_equal(array[:], data)
    with blosc2.RemoteStore(url, path="group") as store:
        assert store.keys() == ["array"]


@pytest.mark.parametrize("cached", [False, True])
def test_remote_hdf5_group_dispatch(tmp_path, cached):
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "groups.h5"
    with h5py.File(source, "w") as file:
        file.create_dataset("group/data", data=np.arange(4))
    url = f"memory://{tmp_path.name}-groups.h5"
    fsspec.filesystem("memory").pipe(url, source.read_bytes())
    options = {"cache_dir": tmp_path / "cache"} if cached else {}
    for _ in range(2):
        with blosc2.open(url, path="group", **options) as group:
            assert isinstance(group, blosc2.RemoteStore)
            assert group.keys() == ["data"]
            with group["data"] as array:
                np.testing.assert_array_equal(array[:], np.arange(4))
    for target in (url, source):
        for root in (None, "", "/"):
            with blosc2.open(target, path=root, **options) as group:
                assert isinstance(group, blosc2.RemoteStore)
                assert group.keys() == ["group"]
    with pytest.raises(ValueError, match="not found"):
        blosc2.open(url, path="missing")
