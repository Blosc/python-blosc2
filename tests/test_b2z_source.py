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


def test_small_member_prefetch_carries_vlmeta(monkeypatch):
    data = np.random.default_rng(7).integers(0, 256, (200, 200), dtype="uint8")
    array = blosc2.asarray(data)
    array.vlmeta["greeting"] = "hello"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("d0/a.b2nd", array.to_cframe())
    fs = fsspec.filesystem("memory")
    fs.pipe_file("vlmeta_prefetch.b2z", buffer.getvalue())

    reads = []
    original = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return original(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    arr = blosc2.open("memory://vlmeta_prefetch.b2z", dataset="d0/a", lazy=True)
    # ZIP tail, then one whole-member request that also holds the frame trailer.
    assert len(reads) == 2
    assert dict(arr.vlmeta) == {"greeting": "hello"}
    assert len(reads) == 2  # the trailer came in the opening request


@pytest.mark.parametrize("suffix", ["supported", "ignored", "rejected", "malformed"])
@pytest.mark.parametrize("small", [False, True])
def test_http_tail_bootstrap(suffix, small):
    import http.server
    import threading

    from blosc2.b2z_source import B2ZArchive

    pytest.importorskip("aiohttp")
    url, data = memory_archive(np.zeros((40, 250), dtype="uint8") if small else None)
    body = fsspec.filesystem("memory").cat_file(url)
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            requests.append(("HEAD", None))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("ETag", '"fixture"')
            self.end_headers()

        def do_GET(self):
            span = self.headers.get("Range")
            requests.append(("GET", span))
            assert self.headers.get("X-Test") == "preserved"
            if span == "bytes=-8192" and suffix in {"ignored", "rejected"}:
                self.send_response(200 if suffix == "ignored" else 416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            first, last = span.removeprefix("bytes=").split("-")
            start = int(first) if first else max(0, len(body) - int(last))
            end = min(int(last), len(body) - 1) if first else len(body) - 1
            self.send_response(206)
            content_range = f"bytes {start}-{end}/{len(body)}"
            if span == "bytes=-8192" and suffix == "malformed":
                content_range = "invalid"
            self.send_header("Content-Range", content_range)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("ETag", '"fixture"')
            self.end_headers()
            self.wfile.write(body[start : end + 1])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/array.b2z"
        options = {"headers": {"X-Test": "preserved"}, "skip_instance_cache": True}
        with blosc2.open(url + "::/d0/a", storage_options=options) as arr:
            assert requests[0] == ("GET", "bytes=-8192")
            assert sum(method == "HEAD" for method, _ in requests) == (suffix != "supported")
            if suffix == "supported":
                assert len(requests) == (1 if small else 2)
                assert arr.traffic.nbytes == (len(body) if small else 8192 + 16384)
            np.testing.assert_array_equal(arr[:], data)

        # A persisted suffix bootstrap must have the same identity as HEAD.
        metadata = {}
        archive = B2ZArchive(url, storage_options=options, _metadata=metadata)
        archive.close()
        requests.clear()
        archive = B2ZArchive(url, storage_options=options, _metadata=metadata)
        archive.close()
        assert requests == [("HEAD", None)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("reopen", ["url", "carrier", "cframe"])
def test_disk_cache_reopen_replays_b2z_bootstrap(tmp_path, monkeypatch, reopen):
    data = np.random.default_rng(3).integers(0, 256, (1000, 1000), dtype="uint8")
    array = blosc2.asarray(data, chunks=(200, 250), blocks=(50, 50))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("d0/a.b2nd", array.to_cframe())
    fs = fsspec.filesystem("memory")
    fs.pipe_file("replay.b2z", buffer.getvalue())
    url = "memory://replay.b2z"

    reads = []
    original = type(fs).cat_file

    def counted(self, path, start=None, end=None, **kwargs):
        reads.append((start, end))
        return original(self, path, start=start, end=end, **kwargs)

    monkeypatch.setattr(type(fs), "cat_file", counted)
    first = blosc2.open(url, dataset="d0/a", lazy=True, cache_dir=tmp_path)
    first[:200, :250]
    assert reads  # the cold open bootstrapped the ZIP

    seed = first._carrier.schunk.vlmeta["b2z-frame"]
    assert isinstance(seed, dict)
    assert isinstance(seed["prefix"], bytes)

    def no_discovery(*args, **kwargs):
        pytest.fail("warm cache must not rediscover the archive")

    monkeypatch.setattr(type(fs), "info", no_discovery)
    reads.clear()
    if reopen == "url":
        second = blosc2.open(url, dataset="d0/a", lazy=True, cache_dir=tmp_path)
        assert second._cache_status == "reused"
    elif reopen == "carrier":
        second = blosc2.open(first.cache_path)
    else:
        second = blosc2.from_cframe(first.to_cframe())
    assert not reads  # the cached bootstrap replaced the remote ZIP bootstrap
    assert second.src._archive._fs is None
    np.testing.assert_array_equal(second[:200, :250], data[:200, :250])
    assert not reads  # the warm chunk needs no remote access either
    np.testing.assert_array_equal(second[:], data)
    assert reads
    lo, hi = second.src.member_offset, second.src.member_offset + second.src.member_length
    assert all(lo <= start < end <= hi for start, end in reads)
    assert second.traffic.nbytes == sum(end - start for start, end in reads)


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
    # A dataset path automatically selects lazy remote access.
    with blosc2.open(url, dataset="d0/a") as arr:
        assert isinstance(arr, blosc2.RemoteArray)
        assert arr.dataset == "d0/a"
    with pytest.raises(NotImplementedError, match="requires lazy=True"):
        blosc2.open(url, dataset="d0/a", lazy=False)
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


@pytest.mark.parametrize("separator", ["/", "::/"])
def test_parser_query_and_local_store(tmp_path, monkeypatch, separator):
    assert parse_container_url("zip://a.b2nd::memory://h.b2z")[2] is None
    assert parse_container_url("memory://h.b2z/group.h5/a") == ("memory://h.b2z", "group.h5/a", "b2z")
    assert parse_container_url("https://host/h.b2z/d0/a?x=1") == ("https://host/h.b2z?x=1", "d0/a", "b2z")
    assert parse_container_url("https://host/frame?x=h.b2z/d0/a")[2] is None
    path = tmp_path / "local.b2z"
    with blosc2.TreeStore(path, mode="w") as store:
        store["/a"] = np.arange(10)
    with blosc2.open(path) as store:
        np.testing.assert_array_equal(store["/a"][:], np.arange(10))
    monkeypatch.chdir(tmp_path)
    arr = blosc2.open(f"local.b2z{separator}a")
    np.testing.assert_array_equal(arr[:], np.arange(10))
    assert dict(arr.info_items)["source"]["urlpath"] == "local.b2z"
    assert "local.b2z" in repr(arr.info)
    assert "local.b2z" in arr.info._repr_html_()
    with pytest.raises(ValueError, match="remote URL"):
        arr.to_cframe()


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
    # Legacy carriers without a bootstrap still discover and validate remote geometry.
    del arr._carrier.schunk.vlmeta["b2z-frame"]
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


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
@pytest.mark.parametrize("limit", [1000, 100_000])
def test_whole_member_prefetch_counts_and_obeys_budget(tmp_path, policy, limit):
    data = np.random.default_rng(123).integers(0, 256, (200, 200), dtype="uint8")
    array = blosc2.asarray(data, chunks=(100, 100), blocks=(50, 50))
    frame = array.to_cframe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.b2nd", frame)
    fs = fsspec.filesystem("memory")
    fs.pipe_file("accounted.b2z", buffer.getvalue())
    kwargs = {} if policy is blosc2.CachePolicy.NONE else {"max_cache_bytes": limit}
    if policy is blosc2.CachePolicy.DISK:
        kwargs["cache_dir"] = tmp_path
    arr = blosc2.RemoteArray("memory://accounted.b2z", dataset="a", cache_policy=policy, **kwargs)
    expected = sum(len(array.schunk.get_chunk(i)) for i in range(array.schunk.nchunks))
    retained = expected if policy is not blosc2.CachePolicy.NONE and limit >= expected else 0
    assert arr.cache_bytes == retained
    assert arr.src._seed["prefix"] == arr.src._raw_header
    assert arr.src._head == arr.src._raw_header
    assert arr.src._prefetched_frame is None
    arr.traffic.reset()
    np.testing.assert_array_equal(arr[:], data)
    assert (arr.traffic.requests == 0) == bool(retained)
    assert arr.cache_bytes == retained  # Reading does not duplicate the prefetch.
    if policy is blosc2.CachePolicy.DISK:
        reopened = blosc2.open(arr.cache_path, mode="a")
        assert reopened.traffic.requests == 0
        assert reopened.cache_bytes == retained
        exported = blosc2.from_cframe(arr.to_cframe())
        assert exported.cache_bytes == retained
        cold = blosc2.from_cframe(arr.to_cframe(include_cache=False))
        assert cold.cache_bytes == 0
        reopened.trim_cache(0)
        assert reopened.cache_bytes == 0
        again = blosc2.open(arr.cache_path)
        assert again.cache_bytes == 0  # The bootstrap cannot resurrect evicted data.


@pytest.mark.parametrize("direct", [False, True])
def test_legacy_whole_member_bootstrap_moves_into_chunk_cache(tmp_path, direct):
    data = np.arange(10000, dtype="int32")
    array = blosc2.asarray(data, chunks=(2500,), blocks=(2500,))
    frame = array.to_cframe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.b2nd", frame)
    fsspec.filesystem("memory").pipe_file("legacy-prefetch.b2z", buffer.getvalue())
    url = "memory://legacy-prefetch.b2z"
    arr = blosc2.open(url, dataset="a", lazy=True, cache_dir=tmp_path)
    arr.trim_cache(0)
    # Reproduce the old carrier representation: all data hidden in b2z-frame.
    seed = dict(arr.src._seed, prefix=frame)
    arr._carrier.schunk.vlmeta["b2z-frame"] = seed
    reopened = (
        blosc2.open(arr.cache_path, mode="a")
        if direct
        else blosc2.open(url, dataset="a", lazy=True, cache_dir=tmp_path)
    )
    assert reopened.traffic.requests == 0
    assert reopened.cache_bytes == sum(len(array.schunk.get_chunk(i)) for i in range(array.schunk.nchunks))
    assert reopened._carrier.schunk.vlmeta["b2z-frame"]["prefix"] == reopened.src._raw_header
    np.testing.assert_array_equal(reopened[:], data)
    assert reopened.traffic.requests == 0


def test_read_only_portable_carrier_keeps_b2z_seed(tmp_path):
    """A stale legacy seed on a read-only carrier must not crash the sparse attach."""
    data = np.arange(1000, dtype="int32")
    array = blosc2.asarray(data, chunks=(250,), blocks=(250,))
    frame = array.to_cframe()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("a.b2nd", frame)
    fsspec.filesystem("memory").pipe_file("read-only-seed.b2z", buffer.getvalue())
    url = "memory://read-only-seed.b2z"

    live = blosc2.RemoteArray(
        url, dataset="a", cache_policy=blosc2.CachePolicy.DISK, cache_dir=tmp_path / "live"
    )
    live[:250]
    carrier_path = tmp_path / "portable.b2nd"
    live.save(carrier_path)

    writer = blosc2.blosc2_ext.open(str(carrier_path), "a", 0)
    seed = writer.schunk.vlmeta["b2z-frame"]
    writer.schunk.vlmeta["b2z-frame"] = dict(seed, prefix=frame)  # legacy whole-frame seed

    carrier = blosc2.blosc2_ext.open(str(carrier_path), "r", 0)
    src = blosc2.B2ZNDSource(url, dataset="a")
    descriptor = {"kind": "b2z", "version": 1, "urlpath": url, "dataset": "a", "assume_immutable": True}
    with blosc2.RemoteArray.with_sparse_cache(
        src, tmp_path / "sparse", carrier=carrier, source_descriptor=descriptor
    ) as runtime:
        np.testing.assert_array_equal(runtime[:], data)
    # The read-only carrier keeps its original seed instead of being rewritten.
    reopened = blosc2.blosc2_ext.open(str(carrier_path), "r", 0)
    assert reopened.schunk.vlmeta["b2z-frame"]["prefix"] == frame
