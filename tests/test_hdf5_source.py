#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import builtins
import io
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.hdf5_source import check_hdf5_dependencies

h5py = pytest.importorskip("h5py")
kerchunk = pytest.importorskip("kerchunk")
zarr = pytest.importorskip("zarr")
fsspec = pytest.importorskip("fsspec")


def make_memory_h5(name: str = "test.h5", **datasets) -> str:
    """Create an HDF5 file in fsspec memory filesystem with the specified datasets."""
    fs = fsspec.filesystem("memory")
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        for ds_path, val in datasets.items():
            if isinstance(val, tuple):
                data, chunks = val
                f.create_dataset(ds_path, data=data, chunks=chunks)
            elif isinstance(val, dict):
                f.create_dataset(ds_path, **val)
            else:
                f.create_dataset(ds_path, data=val)
    fs.pipe_file(name, buf.getvalue())
    return f"memory://{name}"


def test_hdf5_scan_resets_zarr_resources_after_timeout(monkeypatch):
    import blosc2.hdf5_source as hdf5_source

    attempts = []
    resets = []

    def translate(*args):
        attempts.append(None)
        if len(attempts) == 1:
            raise TimeoutError("wedged sync bridge")
        return {"refs": {}}

    monkeypatch.setattr(hdf5_source, "_translate_hdf5", translate)
    monkeypatch.setattr(hdf5_source, "_reset_zarr_sync_resources", lambda: resets.append(None))

    assert hdf5_source.scan_hdf5_refs("memory://unused.h5") == {"refs": {}}
    assert len(attempts) == 2
    assert len(resets) == 2


# ---------------------------------------------------------------------------
# Adapter tests (HDF5NDSource directly)
# ---------------------------------------------------------------------------


def test_hdf5_reference_and_expression_roundtrip(tmp_path):
    data = np.arange(20, dtype=np.int32)
    url = make_memory_h5("reference.h5", **{"group/data": (data, (5,))})
    array = blosc2.RemoteArray(url, dataset="group/data")
    reference = blosc2.Ref.from_dict(blosc2.Ref.from_object(array).to_dict())
    np.testing.assert_array_equal(reference.open()[:], data)
    expression = array + 2
    destination = tmp_path / "expression.b2nd"
    expression.save(destination)
    np.testing.assert_array_equal(blosc2.open(destination)[:], data + 2)


@pytest.mark.parametrize("extension", ["h5", "hdf5", "H5"])
def test_hdf5_url_preserves_query(extension):
    from blosc2.core import parse_container_url

    url = f"https://host/data.{extension}/group/data?version=1"
    assert parse_container_url(url) == (f"https://host/data.{extension}?version=1", "group/data", "hdf5")
    assert parse_container_url(f"https://host/frame?next=data.{extension}/group/data")[2] is None


def test_hdf5_source_through_proxy(tmp_path):
    path = str(tmp_path / "through_proxy.h5")
    data = np.arange(100, dtype=np.int32).reshape(10, 10)
    with h5py.File(path, "w") as f:
        f.create_dataset("d0/data", data=data, chunks=(5, 5))

    src = blosc2.HDF5NDSource(path, "d0/data")
    proxy = blosc2.Proxy(src)
    assert proxy.shape == (10, 10)
    assert proxy.dtype == np.dtype("int32")
    assert proxy.chunks == (5, 5)

    np.testing.assert_array_equal(proxy[1:4, 2:5], data[1:4, 2:5])
    np.testing.assert_array_equal(proxy[:], data)

    cache_path = tmp_path / "proxy.b2nd"
    cached = blosc2.Proxy(src, urlpath=cache_path)
    np.testing.assert_array_equal(cached[:], data)
    del cached
    np.testing.assert_array_equal(blosc2.open(cache_path)[:], data)


@pytest.mark.parametrize(
    ("dtype", "data"),
    [
        (np.int32, np.arange(20, dtype=np.int32)),
        (np.int64, np.arange(20, dtype=np.int64)),
        (np.float32, np.linspace(0.0, 1.0, 20, dtype=np.float32)),
        (np.float64, np.linspace(0.0, 1.0, 20, dtype=np.float64)),
        (np.bool_, np.array([True, False] * 10, dtype=np.bool_)),
        (np.complex64, np.array([1 + 2j, 3 + 4j] * 10, dtype=np.complex64)),
        (np.dtype("S6"), np.array([b"hello", b"world"] * 10, dtype="S6")),
    ],
)
def test_hdf5_source_dtypes(dtype, data):
    url = make_memory_h5(f"dtypes_{dtype}.h5", ds=(data, (5,)))
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    assert proxy.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(proxy[:], data)


def test_hdf5_source_edge_chunks():
    # 10 is not divisible by 3, 11 is not divisible by 4
    data = np.arange(110, dtype=np.int32).reshape(10, 11)
    url = make_memory_h5("edge_chunks.h5", ds=(data, (3, 4)))
    proxy = blosc2.open(url, lazy=True, dataset="ds")

    np.testing.assert_array_equal(proxy[:], data)
    np.testing.assert_array_equal(proxy[-2:, -3:], data[-2:, -3:])
    np.testing.assert_array_equal(proxy[2:8, 3:9], data[2:8, 3:9])


def test_hdf5_source_fill_value():
    url = make_memory_h5(
        "fill_value.h5",
        ds={"shape": (10,), "dtype": "int32", "chunks": (2,), "fillvalue": -999},
    )
    # Write only first chunk
    fs = fsspec.filesystem("memory")
    buf = io.BytesIO(fs.cat("fill_value.h5"))
    with h5py.File(buf, "a") as f:
        f["ds"][:2] = [10, 20]
    fs.pipe_file("fill_value.h5", buf.getvalue())

    proxy = blosc2.open(url, lazy=True, dataset="ds")
    expected = np.full(10, -999, dtype=np.int32)
    expected[:2] = [10, 20]
    np.testing.assert_array_equal(proxy[:], expected)


def test_hdf5_source_scalar_and_empty():
    url = make_memory_h5("scalar.h5", scalar={"data": 42})
    proxy = blosc2.open(url, lazy=True, dataset="scalar")
    assert proxy.shape == ()
    assert proxy[()] == 42


def test_hdf5_source_multidim():
    d1 = np.arange(50, dtype=np.float32)
    d2 = np.arange(120, dtype=np.int32).reshape(10, 12)
    d3 = np.arange(60, dtype=np.int16).reshape(3, 4, 5)
    url = make_memory_h5("multidim.h5", a1=(d1, (10,)), a2=(d2, (5, 4)), a3=(d3, (1, 2, 5)))

    p1 = blosc2.open(url, lazy=True, dataset="a1")
    p2 = blosc2.open(url, lazy=True, dataset="a2")
    p3 = blosc2.open(url, lazy=True, dataset="a3")

    np.testing.assert_array_equal(p1[:], d1)
    np.testing.assert_array_equal(p2[:], d2)
    np.testing.assert_array_equal(p3[:], d3)


def test_hdf5_source_gzip_compression():
    data = np.arange(100, dtype=np.int32)
    url = make_memory_h5(
        "gzip.h5",
        ds={"data": data, "chunks": (20,), "compression": "gzip", "compression_opts": 4},
    )
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    np.testing.assert_array_equal(proxy[:], data)


def test_hdf5_source_group_error():
    url = make_memory_h5("group_err.h5", **{"d0/d1/arr": np.arange(10)})
    with pytest.raises(ValueError, match="is an HDF5 group; pass the path of a dataset"):
        blosc2.HDF5NDSource(url, "d0/d1")


def test_hdf5_source_missing_dataset():
    url = make_memory_h5("missing_ds.h5", ds1=np.arange(10), ds2=np.arange(5))
    with pytest.raises(ValueError, match="dataset 'nonexistent' not found"):
        blosc2.HDF5NDSource(url, "nonexistent")


def test_hdf5_source_available_datasets():
    url = make_memory_h5("available.h5", **{"d0/a0": [1], "d0/d1/a1": [2], "a_root": [3]})
    datasets = blosc2.available_datasets(url)
    assert datasets == ["a_root", "d0/a0", "d0/d1/a1"]


# ---------------------------------------------------------------------------
# RemoteArray integration tests
# ---------------------------------------------------------------------------


def test_open_hdf5_as_remote_array(tmp_path):
    path = str(tmp_path / "open_remote.h5")
    data = np.arange(50, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, chunks=(10,))

    proxy = blosc2.open(path, lazy=True, source_format="hdf5", dataset="data")
    assert isinstance(proxy, blosc2.RemoteArray)
    assert proxy.dataset == "data"
    np.testing.assert_array_equal(proxy[:], data)


def test_hdf5_auto_detection(tmp_path):
    path = str(tmp_path / "auto_detect.h5")
    data = np.arange(30, dtype=np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=data, chunks=(10,))

    # Without source_format="hdf5", suffix should trigger it
    proxy = blosc2.open(path, lazy=True, dataset="data")
    assert isinstance(proxy, blosc2.RemoteArray)
    assert isinstance(proxy.src, blosc2.HDF5NDSource)
    assert proxy.dataset == "data"
    np.testing.assert_array_equal(proxy[:], data)

    # With remote URL, proxy.source also works
    mem_url = make_memory_h5("auto_detect_mem.h5", data=(data, (10,)))
    mem_proxy = blosc2.open(mem_url, lazy=True, dataset="data")
    assert mem_proxy.source["kind"] == "hdf5"


def test_hdf5_requires_dataset():
    url = make_memory_h5("no_ds.h5", data=np.arange(10))
    with pytest.raises(ValueError, match="HDF5 sources require a dataset path"):
        blosc2.open(url, lazy=True)


def test_hdf5_url_syntax_variants(tmp_path):
    data = np.arange(50, dtype=np.int32)
    path = str(tmp_path / "variants.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sub/data", data=data, chunks=(10,))

    p1 = blosc2.open(f"{path}/sub/data", lazy=True)
    p2 = blosc2.open(f"{path}::sub/data", lazy=True)
    p3 = blosc2.open(f"{path}::/sub/data", lazy=True)
    p4 = blosc2.open(path, lazy=True, dataset="sub/data")

    for p in (p1, p2, p3, p4):
        assert p.dataset == "sub/data"
        np.testing.assert_array_equal(p[:], data)


def test_hdf5_url_syntax_variants_memory():
    data = np.arange(50, dtype=np.int32)
    url = make_memory_h5("variants_mem.h5", **{"sub/data": (data, (10,))})

    p1 = blosc2.open(f"{url}/sub/data", lazy=True)
    p2 = blosc2.open(f"{url}::sub/data", lazy=True)
    p3 = blosc2.open(f"{url}::/sub/data", lazy=True)
    p4 = blosc2.open(url, lazy=True, dataset="sub/data")

    for p in (p1, p2, p3, p4):
        assert p.dataset == "sub/data"
        np.testing.assert_array_equal(p[:], data)


def test_hdf5_url_syntax_conflicts(tmp_path):
    path = str(tmp_path / "conflicts.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])

    with pytest.raises(ValueError, match="Cannot specify dataset in both URL path and dataset parameter"):
        blosc2.open(f"{path}/data", lazy=True, dataset="other")

    with pytest.raises(ValueError, match="Cannot specify dataset in both URL path and dataset parameter"):
        blosc2.open(f"{path}::data", lazy=True, dataset="other")


def test_hdf5_requires_lazy(tmp_path):
    path = str(tmp_path / "not_lazy.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("data", data=[1, 2, 3])
    with pytest.raises(ValueError, match="HDF5 sources require lazy=True"):
        blosc2.open(path, lazy=False)
    with pytest.raises(ValueError, match="dataset requires lazy=True"):
        blosc2.open(path, lazy=False, dataset="data")


def test_hdf5_rejects_mutable():
    url = make_memory_h5("mutable.h5", data=np.arange(10))
    with pytest.raises(NotImplementedError, match="mutable HDF5 sources are not supported"):
        blosc2.open(url, lazy=True, dataset="data", assume_immutable=False)


def test_hdf5_memory_cache():
    data = np.arange(40, dtype=np.int32)
    url = make_memory_h5("mem_cache.h5", data=(data, (10,)))
    proxy = blosc2.open(url, lazy=True, dataset="data", cache_policy=blosc2.CachePolicy.MEMORY)

    slice1 = proxy[:10]
    traffic1 = proxy.traffic.nbytes
    assert traffic1 > 0

    slice2 = proxy[:10]
    traffic2 = proxy.traffic.nbytes
    assert traffic2 == traffic1
    np.testing.assert_array_equal(slice1, slice2)


def test_hdf5_disk_cache(tmp_path):
    data = np.arange(40, dtype=np.int32)
    url = make_memory_h5("disk_cache.h5", data=(data, (10,)))
    cache_path = tmp_path / "carrier.b2nd"
    proxy = blosc2.open(
        url,
        lazy=True,
        dataset="data",
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
    )
    np.testing.assert_array_equal(proxy[:10], data[:10])

    assert "hdf5-refs" in proxy.schunk.vlmeta
    reopened = blosc2.open(cache_path)
    assert isinstance(reopened, blosc2.RemoteArray)
    assert reopened.dataset == "data"
    np.testing.assert_array_equal(reopened[:], data)


def test_hdf5_traffic_accounting():
    data = np.arange(100, dtype=np.int32)
    url = make_memory_h5("traffic.h5", data=(data, (20,)))
    proxy = blosc2.open(url, lazy=True, dataset="data")

    # Initial traffic should only be metadata scanning
    initial_traffic = proxy.traffic.nbytes
    assert initial_traffic > 0

    # Cold chunk fetch
    _ = proxy[20:40]
    after_chunk = proxy.traffic.nbytes
    assert after_chunk > initial_traffic

    # Warm chunk hit
    _ = proxy[20:40]
    assert proxy.traffic.nbytes == after_chunk


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------


def test_hdf5_carrier_reopens_warm(tmp_path):
    data = np.arange(60, dtype=np.int32)
    url = make_memory_h5("warm_reopen.h5", data=(data, (20,)))
    cache_path = tmp_path / "warm_carrier.b2nd"

    creator = blosc2.RemoteArray(
        url,
        dataset="data",
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
    )
    _ = creator[:]
    assert "hdf5-refs" in creator.schunk.vlmeta

    reopened = blosc2.open(cache_path)
    reopened.src.get_chunk = lambda nchunk: (_ for _ in ()).throw(AssertionError("cache miss"))
    np.testing.assert_array_equal(reopened[:], data)


def test_hdf5_carrier_save_load(tmp_path):
    data = np.arange(30, dtype=np.int64)
    url = make_memory_h5("save_load.h5", data=(data, (10,)))
    proxy = blosc2.open(url, lazy=True, dataset="data")
    save_path = tmp_path / "saved.b2nd"
    proxy.save(save_path)

    reopened = blosc2.open(save_path)
    np.testing.assert_array_equal(reopened[:], data)


def test_hdf5_source_descriptor():
    data = np.arange(20, dtype=np.int32)
    url = make_memory_h5("descriptor.h5", data=(data, (10,)))
    proxy = blosc2.open(url, lazy=True, dataset="data")
    expected = {
        "kind": "hdf5",
        "version": 1,
        "urlpath": url,
        "dataset": "data",
        "assume_immutable": True,
    }
    assert proxy.source == expected


def test_hdf5_geometry_mismatch(tmp_path):
    url1 = make_memory_h5("geo1.h5", data=(np.arange(20, dtype=np.int32), (10,)))
    cache_path = tmp_path / "geo_carrier.b2nd"
    proxy1 = blosc2.open(url1, lazy=True, dataset="data", cache_path=cache_path)
    _ = proxy1[:10]

    # Reopen against different geometry
    url2 = make_memory_h5("geo2.h5", data=(np.arange(40, dtype=np.int32), (10,)))
    with pytest.raises(ValueError, match="specification"):
        blosc2.open(url2, lazy=True, dataset="data", cache_path=cache_path)


def test_hdf5_refs_in_vlmeta(tmp_path):
    data = np.arange(20, dtype=np.int32)
    url = make_memory_h5("vlmeta_refs.h5", data=(data, (10,)))
    cache_path = tmp_path / "refs_carrier.b2nd"
    proxy = blosc2.open(url, lazy=True, dataset="data", cache_path=cache_path)
    raw_refs = proxy.schunk.vlmeta.get("hdf5-refs")
    assert raw_refs is not None
    try:
        import ujson as json_mod
    except ImportError:
        import json as json_mod
    decompressed = json_mod.loads(blosc2.decompress(raw_refs).decode("utf-8"))
    assert "refs" in decompressed


# ---------------------------------------------------------------------------
# Dependency isolation tests
# ---------------------------------------------------------------------------


def test_hdf5_missing_kerchunk_error(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "kerchunk.hdf" or name == "kerchunk":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"blosc2\[hdf5\]"):
        check_hdf5_dependencies()


def test_hdf5_missing_h5py_error(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "h5py":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"blosc2\[hdf5\]"):
        check_hdf5_dependencies()


def test_blosc2_import_without_hdf5():
    assert hasattr(blosc2, "HDF5NDSource")
    assert hasattr(blosc2, "available_datasets")


# ---------------------------------------------------------------------------
# Offline Moto S3 suite
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def s3_server():
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"
    yield endpoint
    server.stop()


def test_moto_s3_hdf5_read(s3_server):
    s3_opts = {
        "endpoint_url": s3_server,
        "key": "testing",
        "secret": "testing",
        "client_kwargs": {"region_name": "eu-west-1"},
    }
    fs = fsspec.filesystem("s3", **s3_opts)
    fs.mkdir("moto-bucket")

    data = np.arange(50, dtype=np.int32)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("test_ds", data=data, chunks=(10,))
    fs.pipe_file("moto-bucket/test.h5", buf.getvalue())

    url = "s3://moto-bucket/test.h5"
    proxy = blosc2.open(url, lazy=True, dataset="test_ds", storage_options=s3_opts)
    assert isinstance(proxy, blosc2.RemoteArray)
    assert proxy.shape == (50,)
    np.testing.assert_array_equal(proxy[:20], data[:20])
    np.testing.assert_array_equal(proxy[:], data)


def test_moto_s3_hdf5_caching(s3_server, tmp_path):
    s3_opts = {
        "endpoint_url": s3_server,
        "key": "testing",
        "secret": "testing",
        "client_kwargs": {"region_name": "eu-west-1"},
    }
    fs = fsspec.filesystem("s3", **s3_opts)
    if not fs.exists("moto-cache-bucket"):
        fs.mkdir("moto-cache-bucket")

    data = np.arange(40, dtype=np.float64)
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        f.create_dataset("cache_ds", data=data, chunks=(10,))
    fs.pipe_file("moto-cache-bucket/test.h5", buf.getvalue())

    url = "s3://moto-cache-bucket/test.h5"
    cache_path = tmp_path / "s3_carrier.b2nd"
    proxy = blosc2.open(
        url,
        lazy=True,
        dataset="cache_ds",
        storage_options=s3_opts,
        cache_path=cache_path,
    )
    assert proxy.traffic.nbytes > 0
    np.testing.assert_array_equal(proxy[:], data)

    reopened = blosc2.open(cache_path)
    np.testing.assert_array_equal(reopened[:], data)


# ---------------------------------------------------------------------------
# Network suite (Backblaze B2: s3://blosc2/hierarchy.h5)
# ---------------------------------------------------------------------------

STORAGE_OPTIONS = {
    "profile": "blosc2",
    "endpoint_url": "https://s3.us-west-001.backblazeb2.com",
}
LOCAL_HIERARCHY = Path(__file__).resolve().parents[1] / "hierarchy.h5"


@pytest.mark.network
def test_s3_hdf5_open_and_slice():
    pytest.importorskip("s3fs")
    url = "s3://blosc2/hierarchy.h5"
    remote = blosc2.open(url, lazy=True, dataset="d0/d1/a2", storage_options=STORAGE_OPTIONS)
    assert remote.shape == (10, 1000, 1000)
    assert remote.dtype == np.dtype("int32")
    assert remote.chunks == (2, 500, 500)

    slice_remote = remote[0, :3, :3]
    if LOCAL_HIERARCHY.exists():
        with h5py.File(LOCAL_HIERARCHY, "r") as f:
            local_slice = f["d0/d1/a2"][0, :3, :3]
        np.testing.assert_array_equal(slice_remote, local_slice)
    else:
        assert list(slice_remote[0]) == [0, 1, 2]


@pytest.mark.network
def test_s3_hdf5_cache_hit():
    pytest.importorskip("s3fs")
    url = "s3://blosc2/hierarchy.h5"
    remote = blosc2.open(url, lazy=True, dataset="d0/d1/a2", storage_options=STORAGE_OPTIONS)
    _ = remote[0, :3, :3]
    traffic_after_first = remote.traffic.nbytes
    assert traffic_after_first > 0

    _ = remote[0, :3, :3]
    assert remote.traffic.nbytes == traffic_after_first


@pytest.mark.network
def test_s3_hdf5_disk_carrier(tmp_path):
    pytest.importorskip("s3fs")
    url = "s3://blosc2/hierarchy.h5"
    cache_path = tmp_path / "hierarchy_carrier.b2nd"
    remote = blosc2.open(
        url,
        lazy=True,
        dataset="d0/d1/a2",
        storage_options=STORAGE_OPTIONS,
        cache_path=cache_path,
    )
    val = remote[0, :3, :3]
    assert list(val[0]) == [0, 1, 2]

    reopened = blosc2.open(cache_path)
    np.testing.assert_array_equal(reopened[0, :3, :3], val)


@pytest.mark.network
def test_s3_hdf5_nested_datasets():
    pytest.importorskip("s3fs")
    url = "s3://blosc2/hierarchy.h5"
    for ds_path in ["d0/a0", "d0/d1/a1", "d0/d1/d2/a3"]:
        proxy = blosc2.open(url, lazy=True, dataset=ds_path, storage_options=STORAGE_OPTIONS)
        assert proxy.shape == (10, 1000, 1000)
        assert proxy.dtype == np.dtype("int32")
        val = proxy[0, :3, :3]
        assert val.shape == (3, 3)


def test_hdf5_vlmeta(tmp_path):
    path = str(tmp_path / "test_attrs.h5")
    data = np.arange(100, dtype=np.int32).reshape(10, 10)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("d0/data", data=data, chunks=(5, 5))
        ds.attrs["description"] = "hdf5 dataset"
        ds.attrs["sampling_rate"] = 250

    src = blosc2.HDF5NDSource(path, "d0/data")
    assert src.vlmeta["description"] == "hdf5 dataset"
    assert src.vlmeta["sampling_rate"] == 250
    assert "_ARRAY_DIMENSIONS" not in src.vlmeta
    assert "_ARRAY_DIMENSIONS" in src.array.attrs

    url = "memory://test_attrs.h5"
    fsspec.filesystem("memory").pipe_file(url, Path(path).read_bytes())
    proxy = blosc2.RemoteArray(url, source_format="hdf5", dataset="d0/data")
    assert proxy.attrs is proxy.vlmeta
    assert proxy.attrs[:] == {"description": "hdf5 dataset", "sampling_rate": 250}


def test_zarr_sync_reset_waits_for_inflight_read(monkeypatch):
    import threading

    from zarr.core import sync

    from blosc2 import hdf5_source
    from blosc2.zarr_source import ZARR_SYNC_LOCK

    # Keep the reset from touching real Zarr runtime state.
    monkeypatch.setattr(sync, "loop", [None])
    monkeypatch.setattr(sync, "iothread", [None])
    monkeypatch.setattr(sync, "_executor", None)

    reading, release = threading.Event(), threading.Event()

    def read():
        with ZARR_SYNC_LOCK:
            reading.set()
            release.wait(5)

    reader = threading.Thread(target=read)
    reader.start()
    resetter = threading.Thread(target=hdf5_source._reset_zarr_sync_resources)
    try:
        assert reading.wait(5)
        resetter.start()
        resetter.join(0.2)
        # The reset must not stop Zarr's loop while a read holds the sync lock.
        assert resetter.is_alive()
        release.set()
        resetter.join(5)
        assert not resetter.is_alive()
    finally:
        release.set()
        resetter.join(5)
        reader.join(5)


@pytest.mark.network
def test_s3_hdf5_matches_zarr():
    pytest.importorskip("s3fs")
    h5_url = "s3://blosc2/hierarchy.h5"
    zarr_url = "s3://blosc2/hierarchy.zarr/d0/d1/a2"
    h5_proxy = blosc2.open(h5_url, lazy=True, dataset="d0/d1/a2", storage_options=STORAGE_OPTIONS)
    zarr_proxy = blosc2.open(zarr_url, lazy=True, storage_options=STORAGE_OPTIONS)

    assert h5_proxy.shape == zarr_proxy.shape
    assert h5_proxy.dtype == zarr_proxy.dtype
    np.testing.assert_array_equal(h5_proxy[0, :5, :5], zarr_proxy[0, :5, :5])


@pytest.mark.network
def test_s3_hdf5_traffic():
    pytest.importorskip("s3fs")
    url = "s3://blosc2/hierarchy.h5"
    proxy = blosc2.open(url, lazy=True, dataset="d0/d1/a2", storage_options=STORAGE_OPTIONS)
    initial_traffic = proxy.traffic.nbytes
    assert initial_traffic > 0

    _ = proxy[2:4, :5, :5]
    traffic_after_read = proxy.traffic.nbytes
    assert traffic_after_read > initial_traffic

    _ = proxy[2:4, :5, :5]
    assert proxy.traffic.nbytes == traffic_after_read
