#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import hashlib

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

    memory = blosc2.RemoteProxy(url, cache_policy=blosc2.CachePolicy.MEMORY)
    assert memory.max_cache_bytes == 256 * 2**20

    disk = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "cache.b2nd",
    )
    assert disk.max_cache_bytes is None

    with pytest.raises(TypeError, match="CachePolicy"):
        blosc2.RemoteProxy(url, cache_policy="memory")
    with pytest.raises(ValueError, match="not applicable"):
        blosc2.RemoteProxy(url, max_cache_bytes=1)
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


def test_open_selects_remote_proxy_only_for_explicit_policy(tmp_path):
    url, data = _remote_array("open-policy.b2nd")

    legacy = blosc2.open(url, lazy=True)
    assert isinstance(legacy, blosc2.Proxy)
    assert not isinstance(legacy, blosc2.RemoteProxy)

    none = blosc2.open(url, lazy=True, cache_policy=blosc2.CachePolicy.NONE)
    assert isinstance(none, blosc2.RemoteProxy)
    assert none.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(none[:100_000], data[:100_000])

    memory = blosc2.open(url, lazy=True, max_cache_bytes=120_000)
    assert isinstance(memory, blosc2.RemoteProxy)
    assert memory.cache_policy is blosc2.CachePolicy.MEMORY
    assert memory.max_cache_bytes == 120_000

    disk = blosc2.open(
        url,
        lazy=True,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "open-cache.b2nd",
        max_cache_bytes=120_000,
    )
    assert isinstance(disk, blosc2.RemoteProxy)
    assert disk.cache_policy is blosc2.CachePolicy.DISK


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


def test_memory_bound_evicts_lru_chunk():
    url, data = _remote_array("memory-bound.b2nd")
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=120_000,
    )

    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])
    np.testing.assert_array_equal(proxy[100_000:200_000], data[100_000:200_000])
    assert proxy.cache_bytes <= 120_000

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])
    assert proxy.traffic.requests > 0
    assert proxy.cache_bytes <= 120_000


def test_memory_bound_refreshes_lru_on_cache_hit():
    url, data = _remote_array("memory-lru.b2nd")
    proxy = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=220_000,
    )

    proxy[:100_000]
    proxy[100_000:200_000]
    proxy[:100_000]  # chunk 0 is now newer than chunk 1
    proxy[200_000:300_000]

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])
    assert proxy.traffic.requests == 0

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[100_000:200_000], data[100_000:200_000])
    assert proxy.traffic.requests > 0


def test_disk_bound_is_optional_and_shrinks_cache(tmp_path):
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


def test_reference_roundtrip_is_none_and_does_not_mutate(tmp_path):
    url, data = _remote_array("roundtrip.b2nd")
    original = blosc2.RemoteProxy(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=120_000,
    )
    original[:100_000]

    carrier = blosc2.ndarray_from_cframe(original.to_cframe())
    assert carrier.schunk.meta["b2o"] == {"kind": "remote_proxy", "version": 1}
    assert carrier.schunk.vlmeta["b2o"] == {
        "kind": "remote_proxy",
        "version": 1,
        "source": {"kind": "fsspec", "version": 1, "urlpath": url},
        "cache_policy": "none",
    }

    path = tmp_path / "reference.b2nd"
    original.save(path)
    before = hashlib.sha256(path.read_bytes()).digest(), path.stat().st_size, path.stat().st_mtime_ns
    restored = blosc2.open(path, mode="r")
    assert isinstance(restored, blosc2.RemoteProxy)
    assert restored.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(restored[:100_000], data[:100_000])
    after = hashlib.sha256(path.read_bytes()).digest(), path.stat().st_size, path.stat().st_mtime_ns
    assert after == before


def test_reference_rejects_changed_source_geometry(tmp_path):
    url, _ = _remote_array("changed-geometry.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "changed-reference.b2nd"
    blosc2.RemoteProxy(url).save(path)

    replacement = blosc2.arange(200, dtype=np.uint8, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("changed-geometry.b2nd", replacement.to_cframe())
    with pytest.raises(ValueError, match="geometry no longer matches"):
        blosc2.open(path, mode="r")


def test_reference_rejects_runtime_cache_policy_in_payload():
    url, _ = _remote_array("bad-persisted-policy.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteProxy(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "memory"

    with pytest.raises(ValueError, match="must use cache policy 'none'"):
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
