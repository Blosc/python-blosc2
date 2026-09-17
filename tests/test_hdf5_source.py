#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import builtins
import gc
import io
import json
import zlib
from pathlib import Path

import numpy as np
import pytest

import blosc2
from blosc2.hdf5_source import check_hdf5_dependencies

h5py = pytest.importorskip("h5py")
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


def test_hdf5_native_index():
    from blosc2.hdf5_source import HDF5_INDEX_FORMAT, scan_hdf5_index, validate_hdf5_index

    url = make_memory_h5("native-index.h5", data=(np.arange(12, dtype="i4"), (4,)))
    index = scan_hdf5_index(url)
    assert index["format"] == HDF5_INDEX_FORMAT
    assert index["datasets"]["data"]["direct"] is True
    assert len(index["datasets"]["data"]["allocated"]) == 3

    malformed = json.loads(json.dumps(index))
    malformed["datasets"]["data"]["allocated"] = "not-a-list"
    with pytest.raises(ValueError, match="allocation table"):
        validate_hdf5_index(malformed)
    incomplete = json.loads(json.dumps(index))
    del incomplete["datasets"]["data"]["allocated"]
    with pytest.raises(ValueError, match="Incomplete"):
        validate_hdf5_index(incomplete)
    for field in ("shape", "chunks", "fill_value", "attrs", "dtype"):
        incomplete = json.loads(json.dumps(index))
        del incomplete["datasets"]["data"][field]
        with pytest.raises(ValueError, match="Incomplete"):
            validate_hdf5_index(incomplete)


def test_hdf5_object_ndarray_reconstruction():
    from blosc2.hdf5_source import _from_json_value, _json_value

    value = np.empty(2, dtype=object)
    value[0] = np.array([1, 2])
    value[1] = np.array([3, 4])
    restored = _from_json_value(_json_value(value))
    assert restored.shape == (2,)
    np.testing.assert_array_equal(restored[0], [1, 2])
    np.testing.assert_array_equal(restored[1], [3, 4])

    ragged = np.empty(2, dtype=object)
    ragged[0] = np.array([1, 2])
    ragged[1] = np.array([3, 4, 5])
    restored = _from_json_value(_json_value(ragged))
    assert restored.shape == (2,)
    np.testing.assert_array_equal(restored[0], [1, 2])
    np.testing.assert_array_equal(restored[1], [3, 4, 5])


def test_hdf5_titled_dtype_roundtrip():
    from blosc2.hdf5_source import dtype_from_value, dtype_value

    dtype = np.dtype([(("title", "value"), "<i4"), ("point", "<f4", (2,))])
    assert dtype_from_value(json.loads(json.dumps(dtype_value(dtype)))) == dtype


def test_hdf5_index_rejects_malformed_filter_values():
    from blosc2.hdf5_source import scan_hdf5_index, validate_hdf5_index

    url = make_memory_h5(
        "shuffle-index.h5",
        ds={"data": np.arange(20, dtype="i4"), "chunks": (10,), "shuffle": True},
    )
    index = scan_hdf5_index(url)
    validate_hdf5_index(index)
    shuffle = next(item for item in index["datasets"]["ds"]["filters"] if item["id"] == 2)
    shuffle["values"] = [0]
    with pytest.raises(ValueError, match="shuffle"):
        validate_hdf5_index(index)
    shuffle["values"] = []
    with pytest.raises(ValueError, match="shuffle"):
        validate_hdf5_index(index)
    shuffle["values"] = ["wrong"]
    with pytest.raises(ValueError, match="filter values"):
        validate_hdf5_index(index)


def test_hdf5_deflate_decode_is_bounded():
    from blosc2.hdf5_source import _decompress_deflate

    compressed = zlib.compress(b"a" * 100)
    assert _decompress_deflate(compressed, 100) == b"a" * 100
    with pytest.raises(ValueError, match="deflate"):
        _decompress_deflate(compressed, 10)
    with pytest.raises(ValueError, match="deflate"):
        _decompress_deflate(compressed + b"junk", 100)
    # A truncated stream that yields the right length is still invalid.
    with pytest.raises(ValueError, match="deflate"):
        _decompress_deflate(compressed[:-4], 100)


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

    base, dataset = blosc2.HDF5NDSource._parse_url(url, None)
    assert base == f"https://host/data.{extension}?version=1"
    assert dataset == "group/data"
    with pytest.raises(ValueError, match="both URL path and dataset"):
        blosc2.HDF5NDSource._parse_url(url, "other/data")


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


@pytest.mark.parametrize("layout", ["contiguous", "gzip"])
@pytest.mark.parametrize("syntax", ["path", "slash", "separator", "file_url"])
def test_local_hdf5_without_remote_dependencies(tmp_path, monkeypatch, layout, syntax):
    path = tmp_path / "native.h5"
    data = np.arange(110, dtype=">i4").reshape(10, 11)
    options = {} if layout == "contiguous" else {"chunks": (3, 4), "compression": "gzip"}
    with h5py.File(path, "w") as file:
        array = file.create_dataset("d0/a2", data=data, **options)
        array.attrs["description"] = "native hdf5"
        array.attrs["sampling_rate"] = 250
        array.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"zarr", "fsspec"}:
            raise AssertionError(f"Local HDF5 must not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    if syntax == "path":
        proxy = blosc2.open(path, dataset="d0/a2")
    else:
        target = path.as_uri() + "::/d0/a2" if syntax == "file_url" else str(path)
        if syntax != "file_url":
            target += "/d0/a2" if syntax == "slash" else "::/d0/a2"
        proxy = blosc2.open(target)
    assert isinstance(proxy.src.array, h5py.Dataset)
    assert proxy.src._hdf5_index is None
    assert proxy.dtype == data.dtype
    if layout == "gzip":
        assert proxy.chunks == (3, 4)
    assert "native.h5" in repr(proxy.info)
    assert proxy.attrs["description"] == "native hdf5"
    assert proxy.attrs["sampling_rate"] == 250
    np.testing.assert_array_equal(proxy.attrs["_ARRAY_DIMENSIONS"], ["x", "y"])
    np.testing.assert_array_equal(proxy[2:8:2, 3:10:2], data[2:8:2, 3:10:2])
    np.testing.assert_array_equal(proxy[:], data)
    np.testing.assert_array_equal((proxy + 2)[:], data + 2)
    assert blosc2.available_datasets(path) == ["d0/a2"]
    file_id = proxy.src.array.file.id
    del proxy
    gc.collect()
    assert not file_id.valid


def test_local_hdf5_special_datasets_and_errors(tmp_path, monkeypatch):
    path = tmp_path / "special.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("scalar", data=42)
        file.create_dataset("empty", shape=(0, 5), dtype="f8")
        file.create_dataset("fill", shape=(10,), chunks=(3,), dtype="i4", fillvalue=-999)
        file["fill"][:2] = [10, 20]
        file.create_dataset("null", dtype="i4")
        file.create_dataset("variable", data=["hello", "world"])
        file.create_group("group")
    assert blosc2.open(path, dataset="scalar")[()] == 42
    np.testing.assert_array_equal(blosc2.open(path, dataset="empty")[:], np.empty((0, 5)))
    np.testing.assert_array_equal(blosc2.open(path, dataset="fill")[:], [10, 20] + [-999] * 8)

    opened = []
    original = h5py.File

    def track_file(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file.id)
        return file

    monkeypatch.setattr(h5py, "File", track_file)
    for dataset, error, message in [
        ("missing", ValueError, "not found"),
        ("group", ValueError, "is an HDF5 group"),
        ("/", ValueError, "is an HDF5 group"),
        ("null", TypeError, "null datasets"),
        ("variable", TypeError, "fixed-size dtypes"),
    ]:
        with pytest.raises(error, match=message):
            blosc2.HDF5NDSource(path, dataset)
        assert not opened[-1].valid


def test_available_datasets_local_json_without_fsspec(tmp_path, monkeypatch):
    from blosc2.hdf5_source import scan_hdf5_index

    path = tmp_path / "indexed.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=np.arange(8, dtype="i4"), chunks=(4,))
    index_path = tmp_path / "indexed.json"
    index_path.write_text(json.dumps(scan_hdf5_index(str(path))))

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"zarr", "fsspec"}:
            raise AssertionError(f"Local HDF5 must not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert blosc2.available_datasets(str(index_path)) == ["data"]


def test_local_hdf5_explicit_index(tmp_path):
    from blosc2.hdf5_source import scan_hdf5_index

    path = tmp_path / "explicit.h5"
    data = np.arange(20, dtype=np.int32)
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=data, chunks=(5,))
    hdf5_index = scan_hdf5_index(str(path))
    proxy = blosc2.open(path, dataset="data", hdf5_index=hdf5_index)
    assert proxy.src._hdf5_index is hdf5_index
    np.testing.assert_array_equal(proxy[:], data)


def test_local_hdf5_bad_explicit_index_closes_file(tmp_path, monkeypatch):
    from blosc2.hdf5_source import scan_hdf5_index

    path = tmp_path / "bad-index.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=np.arange(8, dtype="i4"), chunks=(4,))
    good = scan_hdf5_index(str(path))
    opened = []
    original = h5py.File

    def track_file(*args, **kwargs):
        file = original(*args, **kwargs)
        opened.append(file.id)
        return file

    monkeypatch.setattr(h5py, "File", track_file)
    bad = dict(good, urlpath=str(path) + "-other")
    with pytest.raises(ValueError, match="does not match") as excinfo:
        blosc2.HDF5NDSource(path, "data", hdf5_index=bad)
    assert "does not match" in str(excinfo.value)
    assert opened
    assert not opened[-1].valid


def test_local_hdf5_explicit_index_without_fsspec(tmp_path, monkeypatch):
    from blosc2.hdf5_source import scan_hdf5_index

    path = tmp_path / "explicit-no-fsspec.h5"
    data = np.arange(20, dtype=np.int32)
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=data, chunks=(5,))
    hdf5_index = scan_hdf5_index(str(path))

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.split(".")[0] in {"zarr", "fsspec"}:
            raise AssertionError(f"Local HDF5 must not import {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    proxy = blosc2.open(path, dataset="data", hdf5_index=hdf5_index)
    assert proxy.src._hdf5_index is hdf5_index
    np.testing.assert_array_equal(proxy[:], data)


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
        (np.dtype(">i4"), np.arange(20, dtype=">i4")),
        (
            np.dtype([("value", "<i4"), ("point", "<f4", (2,))]),
            np.array(
                [(i, (i + 0.5, i + 1.5)) for i in range(20)],
                dtype=[("value", "<i4"), ("point", "<f4", (2,))],
            ),
        ),
    ],
)
def test_hdf5_source_dtypes(dtype, data):
    url = make_memory_h5(f"dtypes_{abs(hash(np.dtype(dtype).str))}.h5", ds=(data, (5,)))
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


@pytest.mark.parametrize("shuffle", [False, True])
def test_hdf5_direct_deflate_pipeline(shuffle):
    data = np.arange(105, dtype=np.int32).reshape(15, 7)
    url = make_memory_h5(
        f"deflate-shuffle-{shuffle}.h5",
        ds={"data": data, "chunks": (6, 4), "compression": "gzip", "shuffle": shuffle},
    )
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    assert proxy.src._metadata["direct"] is True
    np.testing.assert_array_equal(proxy[:], data)
    assert proxy.src._fallback_h5 is None


def test_hdf5_direct_blosc2_pipeline():
    plugin = pytest.importorskip("hdf5plugin")
    data = np.arange(105, dtype=np.int32).reshape(15, 7)
    url = make_memory_h5(
        "direct-blosc2.h5",
        ds={"data": data, "chunks": (6, 4), **plugin.Blosc2()},
    )
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    assert proxy.src._metadata["direct"] is True
    np.testing.assert_array_equal(proxy[:], data)
    assert proxy.src._fallback_h5 is None


def test_hdf5_lzf_uses_reused_fallback():
    data = np.arange(105, dtype=np.int32).reshape(15, 7)
    url = make_memory_h5(
        "fallback-lzf.h5",
        ds={"data": data, "chunks": (6, 4), "compression": "lzf"},
    )
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    assert proxy.src._metadata["direct"] is False
    np.testing.assert_array_equal(proxy[:6], data[:6])
    fallback = proxy.src._fallback_h5
    assert fallback is not None
    np.testing.assert_array_equal(proxy[6:12], data[6:12])
    assert proxy.src._fallback_h5 is fallback
    proxy.close()
    assert proxy.src._fallback_h5 is None
    assert proxy.src._fallback_file is None
    proxy.close()  # Closing is idempotent.
    with pytest.raises(RuntimeError, match="closed"):
        proxy[:]


def test_hdf5_plugin_filter_uses_fallback():
    plugin = pytest.importorskip("hdf5plugin")
    data = np.arange(105, dtype=np.int32).reshape(15, 7)
    url = make_memory_h5(
        "fallback-hdf5plugin.h5",
        ds={"data": data, "chunks": (6, 4), **plugin.Blosc()},
    )
    proxy = blosc2.open(url, lazy=True, dataset="ds")
    assert proxy.src._metadata["direct"] is False
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
    # lazy=True is auto-inferred for HDF5 sources when omitted.
    proxy = blosc2.open(path, dataset="data")
    np.testing.assert_array_equal(proxy[:], [1, 2, 3])
    with pytest.raises(NotImplementedError, match="requires lazy=True"):
        blosc2.open(path, dataset="data", lazy=False)


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

    assert "hdf5-index" in proxy.schunk.vlmeta
    reopened = blosc2.open(cache_path)
    assert isinstance(reopened, blosc2.RemoteArray)
    assert reopened.dataset == "data"
    np.testing.assert_array_equal(reopened[:], data)


def test_hdf5_disk_cache_reuses_index(tmp_path, monkeypatch):
    import blosc2.hdf5_source as hdf5_source

    data = np.arange(40, dtype=np.int32)
    url = make_memory_h5("reuse_index.h5", data=(data, (10,)))
    cache_dir = tmp_path / "cache"
    scans = []
    original = hdf5_source.scan_hdf5_index

    def counting_scan(*args, **kwargs):
        scans.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(hdf5_source, "scan_hdf5_index", counting_scan)

    first = blosc2.open(url, lazy=True, dataset="data", cache_dir=cache_dir)
    np.testing.assert_array_equal(first[:10], data[:10])
    assert len(scans) == 1

    second = blosc2.open(url, lazy=True, dataset="data", cache_dir=cache_dir)
    assert second._cache_status == "reused"
    assert len(scans) == 1  # the cached native index replaced the rescan
    np.testing.assert_array_equal(second[:10], data[:10])


def test_publish_hdf5_index_skips_carriers_without_a_snapshot(tmp_path):
    """Local h5py readers (e.g. Windows drive-letter paths) have no index to share."""
    from blosc2.remote_array import _publish_hdf5_index

    carrier = blosc2.empty((4,), dtype="i4", cparams=blosc2.CParams(nthreads=1))
    path = tmp_path / "shared.hdf5-index.b2"
    _publish_hdf5_index(path, carrier, scanned=True)
    assert not path.exists()


@pytest.mark.parametrize("snapshot", ["new", "legacy", "damaged"])
def test_hdf5_disk_cache_shares_index_between_leaves(tmp_path, monkeypatch, snapshot):
    import blosc2.hdf5_source as hdf5_source

    data = np.arange(40, dtype=np.int32)
    url = make_memory_h5("sibling_index.h5", a=(data, (10,)), b=(data + 1, (10,)))
    scans = []
    original = hdf5_source.scan_hdf5_index

    def counting_scan(*args, **kwargs):
        scans.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(hdf5_source, "scan_hdf5_index", counting_scan)
    with blosc2.open(url + "::a", cache_dir=tmp_path) as first:
        np.testing.assert_array_equal(first[:], data)
    shared = next(tmp_path.rglob("*.hdf5-index.b2"))
    if snapshot == "legacy":
        shared.unlink()
        with blosc2.open(url + "::a", cache_dir=tmp_path):
            pass
        assert shared.exists()
    elif snapshot == "damaged":
        shared.write_bytes(b"broken")
    with blosc2.open(url + "::b", cache_dir=tmp_path) as sibling:
        np.testing.assert_array_equal(sibling[:], data + 1)
    assert len(scans) == (2 if snapshot == "damaged" else 1)
    assert "b" in json.loads(blosc2.decompress(shared.read_bytes()))["datasets"]

    # Different access configurations must not share a container snapshot.
    with blosc2.open(url + "::b", cache_dir=tmp_path, storage_options={"skip_instance_cache": True}):
        pass
    assert len(scans) == (3 if snapshot == "damaged" else 2)


def test_hdf5_blosc2_filter_decodes_super_chunk():
    from blosc2.hdf5_source import _decode_blosc2

    data = np.arange(100, dtype=np.int32)
    schunk = blosc2.SChunk(chunksize=200)
    schunk.append_data(data[:50].tobytes())
    schunk.append_data(data[50:].tobytes())
    # blosc2.decompress() rejects a multi-chunk super-chunk frame, so the Blosc2
    # HDF5 filter codec must fall back to from_cframe().
    decoded = _decode_blosc2(schunk.to_cframe())
    assert bytes(decoded) == data.tobytes()


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
    assert "hdf5-index" in creator.schunk.vlmeta

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


def test_hdf5_index_in_vlmeta(tmp_path):
    data = np.arange(20, dtype=np.int32)
    url = make_memory_h5("vlmeta_index.h5", data=(data, (10,)))
    cache_path = tmp_path / "index_carrier.b2nd"
    proxy = blosc2.open(url, lazy=True, dataset="data", cache_path=cache_path)
    raw_hdf5_index = proxy.schunk.vlmeta.get("hdf5-index")
    assert raw_hdf5_index is not None
    try:
        import ujson as json_mod
    except ImportError:
        import json as json_mod
    decompressed = json_mod.loads(blosc2.decompress(raw_hdf5_index).decode("utf-8"))
    assert decompressed["format"] == "blosc2-hdf5-index"


# ---------------------------------------------------------------------------
# Dependency isolation tests
# ---------------------------------------------------------------------------


def test_hdf5_rejects_virtual_and_external_datasets(tmp_path):
    from blosc2.hdf5_source import scan_hdf5_index

    path = tmp_path / "special-layout.h5"
    data = np.arange(10, dtype="i4")
    with h5py.File(path, "w") as file:
        file.create_dataset("data", data=data, chunks=(5,))
        layout = h5py.VirtualLayout(shape=(4,), dtype="i4")
        layout[:] = h5py.VirtualSource(path, "data", shape=(4,))
        file.create_virtual_dataset("virt", layout)
        file.create_dataset("ext", shape=(4,), dtype="i4", external=str(path))
    url = "memory://special-layout.h5"
    fsspec.filesystem("memory").pipe_file(url, path.read_bytes())
    unsupported = {}
    index = scan_hdf5_index(url, unsupported=unsupported)
    assert "data" in index["datasets"]
    assert "virt" not in index["datasets"]
    assert "ext" not in index["datasets"]
    assert "virtual datasets are not supported" in unsupported["virt"]
    assert "externally stored datasets are not supported" in unsupported["ext"]


def test_hdf5_rejects_legacy_reference_map():
    url = make_memory_h5("legacy-index.h5", data=(np.arange(10), (5,)))
    for index in (
        {"version": 1, "refs": {}},
        {".zarray": {}, "data/.zgroup": {}},
        {".zgroup": {}},
    ):
        with pytest.raises(ValueError, match="Legacy HDF5 reference maps"):
            blosc2.HDF5NDSource(url, "data", hdf5_index=index)


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


@pytest.mark.network
def test_https_hdf5_native_blosc2_reader():
    url = "https://f001.backblazeb2.com/file/blosc2/hierarchy.h5"
    proxy = blosc2.open(url, lazy=True, dataset="d0/d1/a2")
    assert proxy.shape == (1000, 1000)
    assert proxy.chunks == (500, 500)
    assert proxy.dtype == np.dtype("int32")
    assert proxy.src._metadata["direct"] is True
    np.testing.assert_array_equal(
        proxy[:3, :3],
        [[0, 1, 2], [1000, 1001, 1002], [2000, 2001, 2002]],
    )


def test_hdf5_vlmeta(tmp_path):
    path = str(tmp_path / "test_attrs.h5")
    data = np.arange(100, dtype=np.int32).reshape(10, 10)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("d0/data", data=data, chunks=(5, 5))
        ds.attrs["description"] = "hdf5 dataset"
        ds.attrs["sampling_rate"] = 250
        ds.attrs["_ARRAY_DIMENSIONS"] = ["x", "y"]
        ds.attrs["numbers"] = np.arange(3, dtype="i8")
        ds.attrs["vlen"] = np.array(["one", "two"], dtype=object)

    src = blosc2.HDF5NDSource(path, "d0/data")
    assert src.vlmeta["description"] == "hdf5 dataset"
    assert src.vlmeta["sampling_rate"] == 250
    np.testing.assert_array_equal(src.vlmeta["_ARRAY_DIMENSIONS"], ["x", "y"])
    np.testing.assert_array_equal(src.vlmeta["numbers"], np.arange(3, dtype="i8"))
    np.testing.assert_array_equal(src.vlmeta["vlen"], ["one", "two"])
    assert isinstance(src.array, h5py.Dataset)

    url = "memory://test_attrs.h5"
    fsspec.filesystem("memory").pipe_file(url, Path(path).read_bytes())
    proxy = blosc2.RemoteArray(url, source_format="hdf5", dataset="d0/data")
    assert not hasattr(proxy.src, "array")
    assert proxy.attrs is proxy.vlmeta
    attrs = proxy.attrs[:]
    assert attrs["description"] == "hdf5 dataset"
    assert attrs["sampling_rate"] == 250
    np.testing.assert_array_equal(attrs["_ARRAY_DIMENSIONS"], ["x", "y"])
    np.testing.assert_array_equal(attrs["numbers"], np.arange(3, dtype="i8"))
    np.testing.assert_array_equal(attrs["vlen"], ["one", "two"])

    # Array-valued attributes must also survive a portable carrier export.
    destination = tmp_path / "attrs.b2nd"
    proxy.save(destination)
    with blosc2.open(destination) as reopened:
        attrs = reopened.attrs[:]
        assert attrs["description"] == "hdf5 dataset"
        np.testing.assert_array_equal(attrs["numbers"], np.arange(3, dtype="i8"))
        np.testing.assert_array_equal(attrs["vlen"], ["one", "two"])


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
