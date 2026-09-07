import builtins

import numpy as np
import pytest

import blosc2


@pytest.fixture(scope="module")
def zarr():
    return pytest.importorskip("zarr")


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("dtype", [np.bool_, np.int32, np.float64, np.complex64, ">i4"])
def test_zarr_source_through_proxy(tmp_path, zarr, zarr_format, dtype):
    path = tmp_path / f"array-{zarr_format}-{np.dtype(dtype).str}.zarr"
    data = np.arange(35).reshape(5, 7).astype(dtype)
    array = zarr.create_array(path, shape=data.shape, chunks=(3, 4), dtype=dtype, zarr_format=zarr_format)
    array[:] = data

    source = blosc2.ZarrNDSource(path)
    proxy = blosc2.Proxy(source)

    assert source.serves_blocks is False
    assert source.shape == data.shape
    assert source.chunks == (3, 4)
    np.testing.assert_array_equal(proxy[:], data)
    with pytest.raises(IndexError, match="nchunk"):
        source.get_chunk(4)


def test_zarr_source_preserves_fill_and_edge_chunks(tmp_path, zarr):
    path = tmp_path / "fill.zarr"
    array = zarr.create_array(
        path, shape=(5, 7), chunks=(3, 4), dtype=np.int16, fill_value=17, zarr_format=3
    )
    array[:2, :2] = 3

    proxy = blosc2.Proxy(blosc2.ZarrNDSource(path))
    expected = np.full((5, 7), 17, dtype=np.int16)
    expected[:2, :2] = 3
    np.testing.assert_array_equal(proxy[:], expected)


@pytest.mark.parametrize(
    "data",
    [
        np.array([[b"one", b"two"], [b"three", b""]], dtype="S6"),
        np.array([["one", "two"], ["three", ""]], dtype="U6"),
        np.array([["2024-01-01", "2024-01-02"], ["2024-01-03", "2024-01-04"]], dtype="M8[ns]"),
        np.array([[1, 2], [3, 4]], dtype="m8[ns]"),
    ],
)
@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning")
def test_zarr_source_supports_fixed_size_dtypes(tmp_path, zarr, data, zarr_format):
    path = tmp_path / "fixed-size.zarr"
    array = zarr.create_array(path, data=data, chunks=data.shape, zarr_format=zarr_format)

    source = blosc2.ZarrNDSource(path)
    proxy = blosc2.Proxy(source)

    assert source.dtype == data.dtype
    np.testing.assert_array_equal(proxy[:], data)


def test_zarr_source_supports_structured_dtype(tmp_path, zarr):
    data = np.array([(1, 1.5), (2, 2.5)], dtype=[("id", "i4"), ("value", "f4")])
    path = tmp_path / "structured.zarr"
    zarr.create_array(path, data=data, chunks=data.shape, zarr_format=2, fill_value=(0, 0))

    source = blosc2.ZarrNDSource(path)
    proxy = blosc2.Proxy(source)

    assert source.dtype == data.dtype
    np.testing.assert_array_equal(proxy[:], data)


def test_zarr_source_rejects_variable_string_dtype(tmp_path, zarr):
    path = tmp_path / "variable-string.zarr"
    zarr.create_array(path, shape=(2,), chunks=(2,), dtype="str")

    with pytest.raises(TypeError, match="fixed-size dtypes"):
        blosc2.ZarrNDSource(path)


def test_zarr_source_rejects_scalar_and_empty(tmp_path, zarr):
    scalar = tmp_path / "scalar.zarr"
    empty = tmp_path / "empty.zarr"
    zarr.create_array(scalar, shape=(), dtype="i4")
    zarr.create_array(empty, shape=(0,), chunks=(1,), dtype="i4")

    with pytest.raises(ValueError, match="scalar"):
        blosc2.ZarrNDSource(scalar)
    with pytest.raises(ValueError, match="zero-length"):
        blosc2.ZarrNDSource(empty)


def test_zarr_source_group_error(tmp_path, zarr):
    path = tmp_path / "group.zarr"
    zarr.create_group(path)

    with pytest.raises(ValueError, match="pass the path of an array"):
        blosc2.ZarrNDSource(path)


@pytest.mark.parametrize(
    ("url", "source_format"),
    [
        ("memory://zarr-tests/trailing.zarr/", None),
        ("memory://zarr-tests/hierarchy.zarr/d0/a", None),
        ("memory://zarr-tests/suffix-free-open", "zarr"),
    ],
)
def test_open_remote_zarr_as_remote_proxy(zarr, url, source_format):
    data = np.arange(35, dtype=np.int32).reshape(5, 7)
    array = zarr.create_array(url, shape=data.shape, chunks=(3, 4), dtype=data.dtype)
    array[:] = data

    kwargs = {} if source_format is None else {"source_format": source_format}
    proxy = blosc2.open(url, lazy=True, **kwargs)

    assert isinstance(proxy, blosc2.RemoteProxy)
    assert proxy.source == {
        "kind": "zarr",
        "version": 1,
        "urlpath": url,
        "assume_immutable": True,
    }
    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[1:5, 2:6], data[1:5, 2:6])
    assert proxy.traffic.nbytes > 0
    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[1:5, 2:6], data[1:5, 2:6])
    assert proxy.traffic.requests == 0


def test_remote_zarr_disk_carrier_reopens_warm(tmp_path, zarr):
    url = "memory://zarr-tests/persistent.zarr"
    data = np.arange(35, dtype=np.int32).reshape(5, 7)
    array = zarr.create_array(url, shape=data.shape, chunks=(3, 4), dtype=data.dtype)
    array[:] = data
    path = tmp_path / "zarr-proxy.b2nd"
    proxy = blosc2.RemoteProxy(
        url, cache_policy=blosc2.CachePolicy.DISK, cache_path=path, source_format="zarr"
    )
    np.testing.assert_array_equal(proxy[:3, :4], data[:3, :4])

    reopened = blosc2.open(path)
    assert isinstance(reopened, blosc2.RemoteProxy)
    reopened.src.get_chunk = lambda nchunk: (_ for _ in ()).throw(AssertionError("cache miss"))
    np.testing.assert_array_equal(reopened[:3, :4], data[:3, :4])


def test_zarr_requires_lazy_open():
    with pytest.raises(NotImplementedError, match="lazy=True"):
        blosc2.open("memory://zarr-tests/not-opened.zarr", source_format="zarr")


def test_zarr_rejects_mutable_source_mode():
    with pytest.raises(NotImplementedError, match="mutable Zarr"):
        blosc2.RemoteProxy(
            "memory://zarr-tests/not-opened.zarr",
            source_format="zarr",
            assume_immutable=False,
        )


def test_explicit_blosc2_format_overrides_zarr_suffix():
    fsspec = pytest.importorskip("fsspec")
    url = "memory://zarr-tests/blosc2-array.zarr"
    data = np.arange(8, dtype=np.int16)
    array = blosc2.asarray(data, chunks=(4,), blocks=(4,))
    fsspec.filesystem("memory").pipe_file("zarr-tests/blosc2-array.zarr", array.to_cframe())

    proxy = blosc2.open(url, lazy=True, source_format="blosc2")
    assert proxy.source["kind"] == "fsspec"
    np.testing.assert_array_equal(proxy[:], data)


def test_zarr_source_missing_dependency_error(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "zarr":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"blosc2\[zarr\]"):
        blosc2.ZarrNDSource("unused.zarr")


def test_direct_proxy_zarr_cache_reopens(tmp_path, zarr):
    source_path = tmp_path / "source.zarr"
    cache_path = tmp_path / "cache.b2nd"
    data = np.arange(35, dtype=np.int32).reshape(5, 7)
    array = zarr.create_array(source_path, shape=data.shape, chunks=(3, 4), dtype=data.dtype)
    array[:] = data
    proxy = blosc2.Proxy(blosc2.ZarrNDSource(source_path), urlpath=cache_path, mode="w")
    np.testing.assert_array_equal(proxy[:3, :4], data[:3, :4])

    reopened = blosc2.open(cache_path)
    assert isinstance(reopened.src, blosc2.ZarrNDSource)
    reopened.src.get_chunk = lambda nchunk: (_ for _ in ()).throw(AssertionError("cache miss"))
    np.testing.assert_array_equal(reopened[:3, :4], data[:3, :4])


def test_zarr_v3_shards_are_decoded_as_logical_chunks(tmp_path, zarr):
    path = tmp_path / "sharded.zarr"
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    array = zarr.create_array(path, shape=data.shape, chunks=(2, 2), shards=(4, 4), dtype=data.dtype)
    array[:] = data

    proxy = blosc2.Proxy(blosc2.ZarrNDSource(path))
    assert proxy.chunks == (2, 2)
    np.testing.assert_array_equal(proxy[1:7, 1:7], data[1:7, 1:7])


def test_zarr_ref_preserves_explicit_suffix_free_format(zarr):
    url = "memory://zarr-tests/suffix-free"
    data = np.arange(8, dtype=np.int16)
    array = zarr.create_array(url, shape=data.shape, chunks=(4,), dtype=data.dtype)
    array[:] = data
    proxy = blosc2.RemoteProxy(url, source_format="zarr")
    ref = blosc2.Ref.from_object(proxy)

    assert ref.kind == "zarr"
    np.testing.assert_array_equal(ref.open()[:], data)
    restored = blosc2.from_cframe(blosc2.lazyexpr("a + 1", operands={"a": proxy}).to_cframe())
    np.testing.assert_array_equal(restored[:], data + 1)


def test_authorized_zarr_store_is_retained_for_sparse_cache(tmp_path, monkeypatch, zarr):
    store = zarr.storage.MemoryStore()
    data = np.arange(8, dtype=np.int16)
    array = zarr.create_array(store, shape=data.shape, chunks=(4,), dtype=data.dtype)
    array[:] = data
    url = "https://example.org/data.zarr"
    source = blosc2.ZarrNDSource(store, _urlpath=url, _traffic=blosc2.Traffic())
    descriptor = {"kind": "zarr", "version": 1, "urlpath": url, "assume_immutable": True}

    monkeypatch.setattr(
        blosc2.RemoteProxy,
        "_open_source",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unrestricted reopen")),
    )
    proxy = blosc2.RemoteProxy.with_sparse_cache(
        source, tmp_path / "runtime-cache", source_descriptor=descriptor
    )
    np.testing.assert_array_equal(proxy[:], data)
