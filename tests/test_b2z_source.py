"""Native remote reads inside B2Z archives."""

import io
import subprocess
import sys
import zipfile

import numpy as np
import pytest

import blosc2
from blosc2.core import parse_container_url

fsspec = pytest.importorskip("fsspec")


def memory_archive(data=None, *, compression=zipfile.ZIP_STORED, zip64=False):
    if data is None:
        data = np.random.default_rng(42).integers(0, 256, (200, 1000), dtype="uint8")
    array = blosc2.asarray(data, chunks=(40, 250), blocks=(10, 50))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        with archive.open("d0/a.b2nd", "w", force_zip64=zip64) as member:
            member.write(array.to_cframe())
        archive.writestr("d0/b.b2nd", blosc2.asarray(data[::-1].copy()).to_cframe())
    fs = fsspec.filesystem("memory")
    fs.pipe_file("v10.b2z", buffer.getvalue())
    return "memory://v10.b2z", data


@pytest.mark.parametrize("address", ["::/d0/a", "/d0/a", "keyword"])
def test_addressing_and_hits(address, monkeypatch):
    url, data = memory_archive()
    fs = fsspec.filesystem("memory")
    reads = []
    original = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return original(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    arr = (
        blosc2.open(url, lazy=True, dataset="/d0/a/")
        if address == "keyword"
        else blosc2.open(url + address, lazy=True)
    )
    assert arr.dataset == "d0/a"
    assert arr.source["kind"] == "b2z"
    assert arr.chunks == (40, 250)
    assert arr.blocks == (10, 50)
    assert arr.traffic.nbytes == sum(end - start for start, end in reads)
    assert arr.traffic.nbytes < data.nbytes // 4
    assert len(reads) == 2  # ZIP tail, then local header plus native frame prefix.
    assert not arr.src._opening_ranges  # Discovery buffers do not become a second payload cache.
    reads.clear()
    np.testing.assert_array_equal(arr[1:6, :5], data[1:6, :5])
    assert reads
    lo, hi = arr.src.member_offset, arr.src.member_offset + arr.src.member_length
    assert all(lo <= start < end <= hi for start, end in reads)
    reads.clear()
    np.testing.assert_array_equal(arr[1:6, :5], data[1:6, :5])
    assert not reads
    np.testing.assert_array_equal(arr[-3:, -4:], data[-3:, -4:])


@pytest.mark.parametrize("limit", [None, 1000, 300_000])
def test_disk_persistence_and_eviction(tmp_path, limit):
    url, data = memory_archive()
    path = tmp_path / "cache.b2nd"
    arr = blosc2.open(url, dataset="d0/a", lazy=True, cache_path=path, max_cache_bytes=limit)
    np.testing.assert_array_equal(arr[:40], data[:40])
    if limit is not None:
        assert arr.cache_bytes <= limit
    reopened = blosc2.open(path, mode="a")
    before = reopened.traffic.nbytes
    np.testing.assert_array_equal(reopened[:40], data[:40])
    assert (reopened.traffic.nbytes == before) == (limit != 1000)
    restored = blosc2.from_cframe(arr.to_cframe())
    np.testing.assert_array_equal(restored[-2:], data[-2:])
    assert restored.dataset == "d0/a"


def test_policies_exports_and_identity(tmp_path):
    url, data = memory_archive()
    none = blosc2.RemoteArray(url, dataset="d0/a")
    np.testing.assert_array_equal(none[:2], data[:2])
    before = none.traffic.nbytes
    np.testing.assert_array_equal(none[:2], data[:2])
    assert none.traffic.nbytes > before
    memory = blosc2.open(url, dataset="d0/a", lazy=True)
    memory[:2]
    memory.save(tmp_path / "cold.b2nd")
    cold = blosc2.open(tmp_path / "cold.b2nd")
    assert cold.cache_bytes == 0
    np.testing.assert_array_equal(cold.materialize((slice(0, 2),))[:], data[:2])
    a = blosc2.open(url, dataset="d0/a", lazy=True, cache_dir=tmp_path / "caches")
    b = blosc2.open(url, dataset="d0/b", lazy=True, cache_dir=tmp_path / "caches")
    assert a.cache_path != b.cache_path
    np.testing.assert_array_equal(b[:2], data[::-1][:2])
    src = blosc2.B2ZNDSource(url, "d0/a")
    proxy = blosc2.Proxy(src, urlpath=str(tmp_path / "legacy.b2nd"), mode="w")
    proxy[:2]
    np.testing.assert_array_equal(blosc2.open(tmp_path / "legacy.b2nd")[-2:], data[-2:])
    sparse = blosc2.RemoteArray.with_sparse_cache(src, tmp_path / "sparse", source_descriptor=a.source)
    np.testing.assert_array_equal(sparse[:2], data[:2])
    assert sparse.cache_bytes > 0


def test_zip64_and_range_bounds():
    url, data = memory_archive(zip64=True)
    src = blosc2.B2ZNDSource(url, "d0/a")
    np.testing.assert_array_equal(blosc2.Proxy(src)[:], data)
    assert src.read_range(src.member_length - 2, 100) == src.read_range(src.member_length - 2, 2)
    assert src.read_range(src.member_length + 10, 100) == b""
    with pytest.raises(ValueError, match="range"):
        src.read_range(-1, 10)


@pytest.mark.parametrize("variant", ["directory", "comment", "extra"])
def test_opening_buffer_fallbacks(variant):
    url, data = memory_archive()
    fs = fsspec.filesystem("memory")
    buffer = io.BytesIO(fs.cat_file("v10.b2z"))
    if variant == "extra":
        with zipfile.ZipFile(buffer) as archive:
            frame = archive.read("d0/a.b2nd")
        buffer = io.BytesIO()
        info = zipfile.ZipInfo("d0/a.b2nd")
        info.extra = b"\x00\xf0" + (20000).to_bytes(2, "little") + b"x" * 20000
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(info, frame)
    else:
        with zipfile.ZipFile(buffer, "a") as archive:
            if variant == "comment":
                archive.comment = b"x" * 60000
            else:
                for index in range(300):
                    archive.writestr(f"directory/padding-{index}", b"")
    fs.pipe_file("v10.b2z", buffer.getvalue())
    arr = blosc2.open(url, dataset="d0/a", lazy=True)
    np.testing.assert_array_equal(arr[:], data)


@pytest.mark.parametrize("dataset", [None, "", "/", "d0", "missing", "../d0/a", "d0//a", "d0/./a", "d0/\na"])
def test_bad_datasets(dataset):
    url, _ = memory_archive()
    with pytest.raises(ValueError):
        blosc2.open(url, lazy=True, dataset=dataset)


def test_options_and_bad_archives():
    url, _ = memory_archive()
    with pytest.raises(NotImplementedError, match="mutable B2Z"):
        blosc2.open(url, lazy=True, dataset="d0/a", assume_immutable=False)
    with pytest.raises(ValueError, match="both"):
        blosc2.open(url + "::d0/a", dataset="d0/a", lazy=True)
    with pytest.raises(ValueError, match="lazy=True"):
        blosc2.open(url, dataset="d0/a")
    fs = fsspec.filesystem("memory")
    fs.pipe_file("suffix-free", fs.cat_file("v10.b2z"))
    assert (
        blosc2.open("memory://suffix-free", source_format="b2z", dataset="d0/a", lazy=True).dataset == "d0/a"
    )
    url, _ = memory_archive(compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(NotImplementedError, match="ZIP_STORED"):
        blosc2.open(url, dataset="d0/a", lazy=True)
    fs.pipe_file("v10.b2z", b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        blosc2.open(url, dataset="d0/a", lazy=True)


def test_parser_query_and_local_store(tmp_path):
    assert parse_container_url("zip://a.b2nd::memory://h.b2z")[2] is None
    assert parse_container_url("memory://h.b2z/group.h5/a") == ("memory://h.b2z", "group.h5/a", "b2z")
    assert parse_container_url("https://host/h.b2z/d0/a?x=1") == ("https://host/h.b2z?x=1", "d0/a", "b2z")
    assert parse_container_url("https://host/frame?x=h.b2z/d0/a")[2] is None
    path = tmp_path / "local.b2z"
    with blosc2.TreeStore(path, mode="w") as store:
        store["/a"] = np.arange(10)
    with blosc2.open(path) as store:
        np.testing.assert_array_equal(store["/a"][:], np.arange(10))


@pytest.mark.parametrize("dtype", ["int32", "float64", "complex64", "S8", "datetime64[s]", ">i4"])
def test_dtypes_and_edges(dtype):
    data = np.arange(41 * 253).reshape(41, 253).astype(dtype)
    url, _ = memory_archive(data)
    arr = blosc2.open(url, dataset="d0/a", lazy=True)
    np.testing.assert_array_equal(arr[:], data)


@pytest.mark.parametrize("shape", [(), (0,), (0, 5)])
def test_scalar_and_empty(shape):
    data = np.zeros(shape, dtype="int32")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.b2nd", blosc2.asarray(data).to_cframe())
    fsspec.filesystem("memory").pipe_file("empty.b2z", buffer.getvalue())
    arr = blosc2.open("memory://empty.b2z", dataset="a", lazy=True)
    np.testing.assert_array_equal(arr[()], data)


@pytest.mark.parametrize(
    "damage", ["duplicate", "encrypted", "header", "frame_length", "truncated", "object"]
)
def test_invalid_members(damage):
    url, _ = memory_archive()
    fs = fsspec.filesystem("memory")
    raw = bytearray(fs.cat_file("v10.b2z"))
    if damage == "duplicate":
        buffer = io.BytesIO(raw)
        with zipfile.ZipFile(buffer, "a") as archive, pytest.warns(UserWarning, match="Duplicate"):
            archive.writestr("d0/a.b2nd", b"duplicate")
        raw = buffer.getvalue()
    elif damage == "encrypted":
        central = raw.index(b"PK\x01\x02")
        raw[central + 8] |= 1
        raw[6] |= 1
    elif damage == "header":
        raw[0] = 0
    elif damage == "frame_length":
        offset = 30 + len("d0/a.b2nd")
        raw[offset + 16 : offset + 24] = (len(raw) * 2).to_bytes(8, "big")
    elif damage == "truncated":
        raw = raw[:-100]
    else:
        buffer = io.BytesIO()
        from blosc2.b2objects import make_b2object_carrier

        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("d0/a.b2nd", make_b2object_carrier("lazyexpr", (10,), "int32").to_cframe())
        raw = buffer.getvalue()
    fs.pipe_file("v10.b2z", raw)
    with pytest.raises((ValueError, NotImplementedError, zipfile.BadZipFile)):
        blosc2.open(url, dataset="d0/a", lazy=True)


def test_geometry_change_and_expression(tmp_path):
    url, data = memory_archive()
    path = tmp_path / "geometry.b2nd"
    arr = blosc2.open(url, dataset="d0/a", lazy=True, cache_path=path)
    expr = arr + 2
    expr.save(tmp_path / "expr.b2nd")
    np.testing.assert_array_equal(blosc2.open(tmp_path / "expr.b2nd")[:2], data[:2] + 2)
    memory_archive(np.zeros((201, 1000), dtype="uint8"))
    with pytest.raises(ValueError, match="geometry"):
        blosc2.open(path)


def test_b2z_metadata_and_caching():
    data = np.arange(100, dtype=np.int32)
    array = blosc2.asarray(
        data,
        chunks=(20,),
        blocks=(10,),
        meta={"sensor_info": {"model": "X1", "rate": 100}},
    )
    array.vlmeta["experiment_notes"] = {"comment": "test run", "valid": True}
    plain_array = blosc2.asarray(data, chunks=(20,), blocks=(10,))

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("d0/with_meta.b2nd", array.to_cframe())
        archive.writestr("d0/plain.b2nd", plain_array.to_cframe())

    fs = fsspec.filesystem("memory")
    fs.pipe_file("meta_test.b2z", buffer.getvalue())

    # 1. Test B2ZNDSource directly
    src = blosc2.B2ZNDSource("memory://meta_test.b2z", "d0/with_meta")
    assert src.has_vlmetalayers
    assert "b2nd" in src.meta
    assert src.meta["sensor_info"] == {"model": "X1", "rate": 100}
    assert src.vlmeta["experiment_notes"] == {"comment": "test run", "valid": True}

    plain_src = blosc2.B2ZNDSource("memory://meta_test.b2z", "d0/plain")
    assert not plain_src.has_vlmetalayers
    assert "b2nd" in plain_src.meta
    assert "sensor_info" not in plain_src.meta
    plain_src.traffic.reset()
    assert plain_src.vlmeta == {}
    assert plain_src.traffic.requests == 0

    # 2. Test RemoteArray over B2Z
    proxy = blosc2.open("memory://meta_test.b2z", dataset="d0/with_meta", lazy=True)
    assert isinstance(proxy.meta, blosc2.RemoteMetadataMapping)
    assert isinstance(proxy.vlmeta, blosc2.RemoteMetadataMapping)
    assert proxy.meta["sensor_info"] == {"model": "X1", "rate": 100}
    assert proxy.meta.get("sensor_info") == {"model": "X1", "rate": 100}
    assert proxy.meta.get("nonexistent", "fallback") == "fallback"
    assert "b2nd" in proxy.meta
    assert "sensor_info" in proxy.meta
    assert len(proxy.meta) >= 2
    assert proxy.meta[:] == proxy.meta.getall()
    with pytest.raises(TypeError):
        proxy.meta["new_meta"] = 123
    with pytest.raises(TypeError):
        del proxy.meta["sensor_info"]

    assert proxy.vlmeta["experiment_notes"] == {"comment": "test run", "valid": True}
    assert proxy.vlmeta.get("experiment_notes") == {"comment": "test run", "valid": True}
    assert "experiment_notes" in proxy.vlmeta
    assert len(proxy.vlmeta) == 1
    assert proxy.vlmeta[:] == {"experiment_notes": {"comment": "test run", "valid": True}}
    with pytest.raises(TypeError):
        proxy.vlmeta["new_vlmeta"] = 123
    with pytest.raises(TypeError):
        del proxy.vlmeta["experiment_notes"]

    # In-memory caching: subsequent accesses issue 0 network traffic
    proxy.traffic.reset()
    _ = proxy.meta["sensor_info"]
    _ = proxy.vlmeta["experiment_notes"]
    _ = proxy.meta[:]
    _ = proxy.vlmeta[:]
    assert proxy.traffic.requests == 0


def test_optional_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in {'zarr', 'kerchunk', 'h5py', 'hdf5plugin'}:
        raise ImportError('blocked optional dependency')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import blosc2
import fsspec
import io
import zipfile
a = blosc2.arange(10)
fs = fsspec.filesystem('memory')
fs.pipe_file('plain.b2nd', a.to_cframe())
assert blosc2.open('memory://plain.b2nd', lazy=True)[3] == 3
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w') as archive:
    archive.writestr('a.b2nd', a.to_cframe())
fs.pipe_file('optional.b2z', buffer.getvalue())
assert blosc2.open('memory://optional.b2z', lazy=True, dataset='a')[3] == 3
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.network
def test_s3_b2z_slice():
    pytest.importorskip("s3fs")
    arr = blosc2.open(
        "s3://blosc2/hierarchy.b2z::/d0/a3",
        lazy=True,
        storage_options={"profile": "blosc2", "endpoint_url": "https://s3.us-west-001.backblazeb2.com"},
    )
    expected = np.arange(10)[:, None] * 1_000_000 + np.arange(5)
    np.testing.assert_array_equal(arr[:10, 0, :5], expected)
    before = arr.traffic.nbytes
    np.testing.assert_array_equal(arr[:10, 0, :5], expected)
    assert arr.traffic.nbytes == before


@pytest.mark.network
def test_https_b2z_slice():
    arr = blosc2.open("https://f001.backblazeb2.com/file/blosc2/hierarchy.b2z::/d0/a3", lazy=True)
    expected = np.arange(10)[:, None] * 1_000_000 + np.arange(5)
    np.testing.assert_array_equal(arr[:10, 0, :5], expected)
    before = arr.traffic.nbytes
    np.testing.assert_array_equal(arr[:10, 0, :5], expected)
    assert arr.traffic.nbytes == before
