#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import asyncio

import fsspec
import numpy as np
import pytest

import blosc2
import blosc2.c2array as blosc2_c2array
from blosc2.b2objects import decode_b2object_payload


def _remote_array(name="remote-proxy.b2nd", *, nchunks=4, chunk_size=100_000):
    data = np.random.default_rng(1).integers(0, 256, nchunks * chunk_size, dtype=np.uint8)
    array = blosc2.asarray(data, chunks=(chunk_size,), blocks=(chunk_size,))
    url = f"memory://{name}"
    fsspec.filesystem("memory").pipe_file(name, array.to_cframe())
    return url, data


def test_cache_policy_validation(tmp_path):
    url, _ = _remote_array("policy.b2nd")

    none = blosc2.RemoteProxy(url)
    assert none.cache_policy is blosc2.CachePolicy.NONE
    assert none.max_cache_bytes is None

    disk = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "cache.b2nd",
    )
    assert disk.max_cache_bytes == 256 * 2**20
    assert disk.cache_path == str(tmp_path / "cache.b2nd")

    with pytest.raises(TypeError, match="CachePolicy"):
        blosc2.RemoteProxy(url, cache_policy="memory")
    with pytest.raises(ValueError, match="not applicable"):
        blosc2.RemoteProxy(url, max_cache_bytes=1)
    with pytest.raises(TypeError, match="positive integer"):
        blosc2.RemoteProxy(
            url,
            cache_policy=blosc2.CachePolicy.DISK,
            cache_path=tmp_path / "unlimited.b2nd",
            max_cache_bytes=None,
        )
    with pytest.raises(ValueError, match="requires cache_dir or cache_path"):
        blosc2.RemoteProxy(url, cache_policy=blosc2.CachePolicy.DISK)
    with pytest.raises(ValueError, match="max_concurrency"):
        blosc2.RemoteProxy(url, max_concurrency=0)


def test_remote_proxy_array_operand_interface():
    url, _ = _remote_array("operand-interface.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteProxy(url)

    assert proxy.ndim == 1
    assert len(proxy) == 100
    assert proxy.info is not None
    assert dict(proxy.info_items)["cache_policy"] == "NONE"

    expression = blosc2.lazyexpr("a + 1", operands={"a": proxy})
    assert url in dict(expression.info_items)["operands"].values()


def test_open_lazy_selects_remote_proxy(tmp_path):
    url, data = _remote_array("open-policy.b2nd")

    mem = blosc2.open(url, lazy=True)
    assert isinstance(mem, blosc2.RemoteProxy)
    assert mem.cache_policy is blosc2.CachePolicy.MEMORY
    assert mem.cache_path is None
    assert mem.max_cache_bytes == 256 * 2**20
    np.testing.assert_array_equal(mem[:100_000], data[:100_000])

    none = blosc2.open(url, lazy=True, cache_policy=blosc2.CachePolicy.NONE)
    assert isinstance(none, blosc2.RemoteProxy)
    assert none.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(none[:100_000], data[:100_000])

    mem_bounded = blosc2.open(url, lazy=True, max_cache_bytes=120_000)
    assert isinstance(mem_bounded, blosc2.RemoteProxy)
    assert mem_bounded.cache_policy is blosc2.CachePolicy.MEMORY
    assert mem_bounded.max_cache_bytes == 120_000

    disk = blosc2.open(
        url,
        lazy=True,
        cache_path=tmp_path / "open-cache.b2nd",
        max_cache_bytes=120_000,
    )
    assert isinstance(disk, blosc2.RemoteProxy)
    assert disk.cache_policy is blosc2.CachePolicy.DISK
    assert disk.max_cache_bytes == 120_000

    with pytest.raises(NotImplementedError, match="require lazy=True"):
        blosc2.open(url, lazy=False, max_cache_bytes=120_000)


def test_none_does_not_retain_remote_data():
    url, data = _remote_array("none.b2nd")
    proxy = blosc2.RemoteProxy(url)

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])
    assert proxy.traffic.requests > 0
    assert proxy.cache_bytes == 0

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])
    assert proxy.traffic.requests > 0
    assert proxy.cache_bytes == 0


def test_disk_bound_shrinks_self_caching_carrier(tmp_path):
    url, data = _remote_array("disk-bound.b2nd")
    cache_path = tmp_path / "bounded-cache.b2nd"
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=120_000,
    )

    for start in (0, 100_000, 200_000):
        np.testing.assert_array_equal(proxy[start : start + 100_000], data[start : start + 100_000])
    assert proxy.cache_bytes <= 120_000
    assert cache_path.stat().st_size < 140_000

    reopened = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=120_000,
    )
    assert reopened.cache_status == "reused"
    np.testing.assert_array_equal(reopened[300_000:400_000], data[300_000:400_000])
    assert reopened.cache_bytes <= 120_000


def test_interrupted_fetch_leaves_a_reusable_carrier(tmp_path):
    url, data = _remote_array("interrupted.b2nd", nchunks=3, chunk_size=100_000)
    cache_path = tmp_path / "interrupted-proxy.b2nd"
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=1_000_000,
    )
    get_chunk = proxy.src.get_chunk

    def interrupted(nchunk):
        if nchunk == 1:
            raise RuntimeError("simulated interruption")
        return get_chunk(nchunk)

    proxy.src.get_chunk = interrupted
    with pytest.raises(RuntimeError, match="simulated interruption"):
        proxy[:]

    carrier = blosc2.blosc2_ext.open(str(cache_path), "r", 0, dparams=blosc2.DParams(nthreads=1))
    assert carrier.schunk.vlmeta.get("proxy-fetched")
    proxy.src.get_chunk = get_chunk
    np.testing.assert_array_equal(proxy[:], data)

    reopened = blosc2.open(cache_path, mode="r")
    np.testing.assert_array_equal(reopened[:], data)


def test_disk_roundtrip_preserves_warm_cache_and_cold_escape_hatch(tmp_path):
    url, data = _remote_array("roundtrip.b2nd")
    original = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "live-proxy.b2nd",
        max_cache_bytes=120_000,
    )
    original[:100_000]

    carrier = blosc2.ndarray_from_cframe(original.to_cframe())
    assert carrier.schunk.meta["b2o"] == {"kind": "remote_proxy", "version": 1}
    assert carrier.schunk.vlmeta["b2o"] == {
        "kind": "remote_proxy",
        "version": 1,
        "source": {"kind": "fsspec", "version": 1, "urlpath": url},
        "cache_policy": "disk",
        "max_cache_bytes": 120_000,
    }
    assert carrier.schunk.vlmeta.get("proxy-cache-sizes")

    cold_carrier = blosc2.ndarray_from_cframe(original.to_cframe(include_cache=False))
    assert cold_carrier.schunk.vlmeta["b2o"] == carrier.schunk.vlmeta["b2o"]
    assert not cold_carrier.schunk.vlmeta.get("proxy-cache-sizes", {})
    assert original.cache_bytes > 0

    warm_path = tmp_path / "warm.b2nd"
    original.save(warm_path)
    restored = blosc2.open(warm_path, mode="r")
    assert isinstance(restored, blosc2.RemoteProxy)
    assert restored.cache_policy is blosc2.CachePolicy.DISK
    restored.traffic.reset()
    np.testing.assert_array_equal(restored[:100_000], data[:100_000])
    assert restored.traffic.requests == 0

    cold_path = tmp_path / "cold.b2nd"
    original.save(cold_path, include_cache=False)
    cold = blosc2.open(cold_path, mode="r")
    cold.traffic.reset()
    np.testing.assert_array_equal(cold[:100_000], data[:100_000])
    assert cold.traffic.requests > 0
    assert cold_path.stat().st_size < warm_path.stat().st_size

    mutable = blosc2.open(cold_path, mode="a")
    np.testing.assert_array_equal(mutable[:100_000], data[:100_000])
    reopened = blosc2.open(cold_path, mode="r")
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:100_000], data[:100_000])
    assert reopened.traffic.requests == 0


def test_reference_rejects_changed_source_geometry(tmp_path):
    url, _ = _remote_array("changed-geometry.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "changed-reference.b2nd"
    blosc2.RemoteProxy(url).save(path)

    replacement = blosc2.arange(200, dtype=np.uint8, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("changed-geometry.b2nd", replacement.to_cframe())
    with pytest.raises(ValueError, match="geometry no longer matches"):
        blosc2.open(path, mode="r")


def test_open_reference_rejects_geometry_changed_before_read(tmp_path):
    url, _ = _remote_array("changed-after-open.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "changed-after-open-reference.b2nd"
    blosc2.RemoteProxy(url).save(path)
    restored = blosc2.open(path, mode="r")

    replacement = blosc2.arange(200, dtype=np.uint8, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("changed-after-open.b2nd", replacement.to_cframe())

    with pytest.raises(ValueError, match="geometry no longer matches"):
        restored[:]


def test_runtime_cache_is_invalidated_after_same_geometry_replacement(tmp_path):
    url, data = _remote_array("same-geometry-disk.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "same-geometry-cache.b2nd",
    )
    traffic = proxy.traffic
    np.testing.assert_array_equal(proxy[:], data)

    replacement = np.arange(100, dtype=np.uint8)
    array = blosc2.asarray(replacement, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("same-geometry-disk.b2nd", array.to_cframe())

    np.testing.assert_array_equal(proxy[:], replacement)
    assert proxy.traffic is traffic


def test_reference_rejects_unknown_cache_policy_in_payload():
    url, _ = _remote_array("bad-persisted-policy.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteProxy(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "unknown_policy"

    with pytest.raises(ValueError, match="unsupported cache policy"):
        decode_b2object_payload(payload, carrier=carrier)


def test_reference_rejects_memory_policy_without_positive_limit_in_payload():
    url, _ = _remote_array("bad-memory-policy.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteProxy(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "memory"
    payload["max_cache_bytes"] = None

    with pytest.raises(ValueError, match="persisted MEMORY RemoteProxy requires positive max_cache_bytes"):
        decode_b2object_payload(payload, carrier=carrier)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("kind", "unknown", "unsupported RemoteProxy source kind"),
        ("version", 2, "unsupported RemoteProxy source descriptor"),
    ],
)
def test_reference_rejects_unknown_source_descriptor(field, value, error):
    url, _ = _remote_array(f"bad-source-{field}.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteProxy(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["source"] = dict(payload["source"], **{field: value})

    with pytest.raises(ValueError, match=error):
        decode_b2object_payload(payload, carrier=carrier)


@pytest.mark.parametrize("field", ["auth_token", "storage_options"])
def test_reference_rejects_secret_or_runtime_source_fields(field):
    url, _ = _remote_array(f"bad-source-{field}.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteProxy(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["source"] = dict(payload["source"], **{field: "secret"})

    with pytest.raises(ValueError, match="unsupported fields"):
        decode_b2object_payload(payload, carrier=carrier)


def test_caterva2_reference_does_not_persist_auth(monkeypatch):
    def fake_info(path, urlbase, params=None, headers=None, model=None, auth_token=None, traffic=None):
        return {
            "shape": [10],
            "chunks": [5],
            "blocks": [5],
            "dtype": np.dtype(np.int32).str,
            "schunk": {
                "cparams": dict(blosc2.cparams_dflts),
                "nbytes": 40,
                "cbytes": 40,
                "cratio": 1.0,
                "blocksize": 20,
                "vlmeta": {},
            },
        }

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    remote = blosc2.RemoteProxy(
        blosc2.URLPath(
            "@personal/private.b2nd",
            urlbase="https://example.org/caterva2",
            auth_token="secret",
        )
    )
    carrier = blosc2.ndarray_from_cframe(remote.to_cframe())
    assert carrier.schunk.vlmeta["b2o"]["source"] == {
        "kind": "caterva2",
        "version": 1,
        "path": "@personal/private.b2nd",
        "urlbase": "https://example.org/caterva2/",
    }

    restored = blosc2.from_cframe(remote.to_cframe())
    assert isinstance(restored, blosc2.RemoteProxy)
    assert isinstance(restored.src, blosc2.C2Array)
    assert restored.src.auth_token is None


def test_caterva2_no_cache_keeps_native_indexing(monkeypatch):
    def fake_info(path, urlbase, params=None, headers=None, model=None, auth_token=None, traffic=None):
        return {
            "shape": [10],
            "chunks": [5],
            "blocks": [5],
            "dtype": np.dtype(np.int32).str,
            "schunk": {"cparams": dict(blosc2.cparams_dflts)},
        }

    calls = []

    def fake_fetch_data(path, urlbase, params, auth_token=None, as_blosc2=False, traffic=None):
        calls.append(params)
        return np.arange(10, dtype=np.int32)[2:5]

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    monkeypatch.setattr(blosc2_c2array, "fetch_data", fake_fetch_data)
    remote = blosc2.RemoteProxy(
        blosc2.URLPath("@public/native-index.b2nd", urlbase="https://example.org/c2")
    )

    np.testing.assert_array_equal(remote[2:5], np.arange(10, dtype=np.int32)[2:5])
    assert calls == [{"slice_": "2:5"}]


def test_caterva2_disk_cache_reuses_chunks(monkeypatch, tmp_path):
    data = np.arange(10, dtype=np.int32)
    local = blosc2.asarray(data, chunks=(5,), blocks=(5,))
    compressed = [local.schunk.get_chunk(i) for i in range(2)]
    calls = []

    def fake_info(path, urlbase, params=None, headers=None, model=None, auth_token=None, traffic=None):
        return {
            "shape": [10],
            "chunks": [5],
            "blocks": [5],
            "dtype": data.dtype.str,
            "mtime": 1,
            "accept_ranges": "none",
            "schunk": {
                "cparams": dict(blosc2.cparams_dflts),
                "cbytes": sum(map(len, compressed)),
                "vlmeta": {},
            },
        }

    def fake_get_chunk(self, nchunk):
        calls.append(nchunk)
        return compressed[nchunk]

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    monkeypatch.setattr(blosc2.C2Array, "get_chunk", fake_get_chunk)
    remote = blosc2.RemoteProxy(
        blosc2.URLPath("@public/cache.b2nd", urlbase="https://example.org/c2"),
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "caterva2-cache.b2nd",
    )

    np.testing.assert_array_equal(remote[:5], data[:5])
    np.testing.assert_array_equal(remote[:5], data[:5])
    assert calls == [0]


def test_remote_proxy_is_a_persistable_lazyexpr_operand():
    url, data = _remote_array("operand.b2nd", nchunks=1, chunk_size=100)
    remote = blosc2.RemoteProxy(url)
    expression = blosc2.lazyexpr("a + 1", operands={"a": remote})

    restored = blosc2.from_cframe(expression.to_cframe())
    assert any(isinstance(operand, blosc2.RemoteProxy) for operand in restored.operands.values())
    np.testing.assert_array_equal(restored[:], data + 1)


def test_objectarray_msgpack_supports_remote_proxy():
    url, data = _remote_array("objectarray-remote-proxy.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteProxy(url)

    objects = blosc2.ObjectArray()
    objects.append(proxy)
    restored = objects[0]

    assert isinstance(restored, blosc2.RemoteProxy)
    assert restored.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(restored[:], data)


def test_batcharray_msgpack_supports_remote_proxy():
    url, data = _remote_array("batcharray-remote-proxy.b2nd", nchunks=1, chunk_size=100)
    batches = blosc2.BatchArray()
    batches.append([blosc2.RemoteProxy(url)])

    restored = batches[0][0]
    assert isinstance(restored, blosc2.RemoteProxy)
    assert restored.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(restored[:], data)


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_get_chunk_for_each_policy(tmp_path, policy):
    url, data = _remote_array(f"get-chunk-{policy.value}.b2nd", nchunks=2, chunk_size=100)
    kwargs = {"cache_path": tmp_path / "get-chunk-cache.b2nd"} if policy is blosc2.CachePolicy.DISK else {}
    proxy = blosc2.RemoteProxy(url, cache_policy=policy, **kwargs)

    chunk = proxy.get_chunk(1)
    np.testing.assert_array_equal(np.frombuffer(blosc2.decompress2(chunk), dtype=np.uint8), data[100:])


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_aget_chunk_for_each_policy(tmp_path, policy):
    url, data = _remote_array(f"aget-chunk-{policy.value}.b2nd", nchunks=2, chunk_size=100)
    kwargs = {"cache_path": tmp_path / "aget-chunk-cache.b2nd"} if policy is blosc2.CachePolicy.DISK else {}
    proxy = blosc2.RemoteProxy(url, cache_policy=policy, **kwargs)

    chunk = asyncio.run(proxy.aget_chunk(1))
    np.testing.assert_array_equal(np.frombuffer(blosc2.decompress2(chunk), dtype=np.uint8), data[100:])


def test_disk_cache_survives_reopen_without_remote_data_traffic(tmp_path):
    url, data = _remote_array("disk-reuse.b2nd", nchunks=2, chunk_size=100_000)
    cache_path = tmp_path / "disk-reuse-cache.b2nd"
    first = blosc2.RemoteProxy(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    np.testing.assert_array_equal(first[:100_000], data[:100_000])

    reopened = blosc2.RemoteProxy(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:100_000], data[:100_000])
    assert reopened.traffic.requests == 0


def test_save_after_disk_cache_use_preserves_or_strips_cache(tmp_path):
    url, data = _remote_array("save-after-disk.b2nd", nchunks=2, chunk_size=100_000)
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "runtime-cache.b2nd",
    )
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])

    warm_path = tmp_path / "saved-warm.b2nd"
    cold_path = tmp_path / "saved-cold.b2nd"
    proxy.save(warm_path)
    proxy.save(cold_path, include_cache=False)
    warm = blosc2.open(warm_path, mode="r")
    cold = blosc2.open(cold_path, mode="r")
    warm.traffic.reset()
    cold.traffic.reset()
    np.testing.assert_array_equal(warm[:100_000], data[:100_000])
    np.testing.assert_array_equal(cold[:100_000], data[:100_000])
    assert warm.traffic.requests == 0
    assert cold.traffic.requests > 0
    assert "proxy-source" not in warm._carrier.schunk.meta


def test_reference_size_is_independent_of_remote_payload(tmp_path):
    url, _ = _remote_array("metadata-sized.b2nd", nchunks=20, chunk_size=100_000)
    path = tmp_path / "metadata-sized-reference.b2nd"
    blosc2.RemoteProxy(url).save(path)

    assert path.stat().st_size < 10_000


@pytest.mark.parametrize(
    "url",
    [
        "https://user@example.org/data.b2nd",
        "https://example.org/data.b2nd?token=secret",
        "https://example.org/data.b2nd?sig=secret",
        "https://example.org/data.b2nd#credentials",
        "zip://data.b2nd::https://example.org/archive.zip",
        "file:///private/data.b2nd",
    ],
)
def test_persistence_rejects_credentials_and_chained_urls(url):
    with pytest.raises(ValueError):
        blosc2.RemoteProxy(url)


@pytest.mark.parametrize(
    "url", ["https://example.org/data.b2nd?token=secret", "https://user@example.org/data.b2nd"]
)
def test_fsspec_refs_reject_credentials(url):
    with pytest.raises(ValueError):
        blosc2.Ref.fsspec_ref(url)


def test_memory_cache_eviction_and_retention():
    url, data = _remote_array("mem-evict.b2nd", nchunks=5, chunk_size=20_000)
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=50_000,
    )
    assert proxy.cache_policy is blosc2.CachePolicy.MEMORY
    assert proxy.max_cache_bytes == 50_000
    assert proxy.cache_path is None
    assert proxy.cache is not None

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:20_000], data[:20_000])
    assert proxy.traffic.requests > 0

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:20_000], data[:20_000])
    assert proxy.traffic.requests == 0

    np.testing.assert_array_equal(proxy[:], data)
    assert proxy.cache_bytes <= 50_000


def test_memory_cache_fetch_and_afetch():
    url, _ = _remote_array("mem-fetch.b2nd", nchunks=3, chunk_size=10_000)
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
    )
    cached_container = proxy.fetch(slice(0, 10_000))
    assert cached_container is proxy.cache
    assert proxy.cache_bytes > 0

    async_container = asyncio.run(proxy.afetch(slice(10_000, 20_000)))
    assert async_container is proxy.cache


def test_memory_proxy_save_and_reopen(tmp_path):
    url, data = _remote_array("mem-save.b2nd", nchunks=2, chunk_size=10_000)
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=100_000,
    )
    save_path = tmp_path / "saved_memory_proxy.b2nd"
    proxy.save(save_path)

    reopened = blosc2.open(save_path, mode="r")
    assert isinstance(reopened, blosc2.RemoteProxy)
    assert reopened.cache_policy is blosc2.CachePolicy.MEMORY
    assert reopened.max_cache_bytes == 100_000
    assert reopened.cache_path is None
    np.testing.assert_array_equal(reopened[:], data)


def test_remote_proxy_fetch_rejects_none_policy():
    url, _ = _remote_array("none-fetch.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteProxy(url, cache_policy=blosc2.CachePolicy.NONE)
    with pytest.raises(
        NotImplementedError, match=r"fetch requires CachePolicy\.DISK or CachePolicy\.MEMORY"
    ):
        proxy.fetch()
    with pytest.raises(
        NotImplementedError, match=r"afetch requires CachePolicy\.DISK or CachePolicy\.MEMORY"
    ):
        asyncio.run(proxy.afetch())
