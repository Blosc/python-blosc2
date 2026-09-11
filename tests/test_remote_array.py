#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import asyncio

import numpy as np
import pytest

import blosc2
import blosc2.c2array as blosc2_c2array
from blosc2.b2objects import decode_b2object_payload

fsspec = pytest.importorskip("fsspec")


def test_bounded_unbounded_cache_accounting_transition(tmp_path):
    url, data = _remote_array("accounting-transition.b2nd", nchunks=3, chunk_size=10_000)
    path = tmp_path / "transition.b2nd"
    creator = blosc2.RemoteArray(
        url, cache_policy=blosc2.CachePolicy.DISK, cache_path=path, max_cache_bytes=None
    )
    bounded = blosc2.Proxy(creator.src, _cache=creator._carrier, _max_cache_bytes=20_000)
    np.testing.assert_array_equal(bounded[:10_000], data[:10_000])
    assert creator.schunk.vlmeta.get("proxy-cache-sizes")
    unbounded = blosc2.open(path, mode="a")
    np.testing.assert_array_equal(unbounded[:], data)
    assert "proxy-cache-sizes" not in unbounded.schunk.vlmeta
    bounded = blosc2.Proxy(unbounded.src, _cache=unbounded._carrier, _max_cache_bytes=20_000)
    assert bounded._retained_cache_bytes() == unbounded.schunk.cbytes
    np.testing.assert_array_equal(bounded[:], data)
    assert bounded._retained_cache_bytes() <= 20_000


def _remote_array(name="remote-proxy.b2nd", *, nchunks=4, chunk_size=100_000):
    data = np.random.default_rng(1).integers(0, 256, nchunks * chunk_size, dtype=np.uint8)
    array = blosc2.asarray(data, chunks=(chunk_size,), blocks=(chunk_size,))
    url = f"memory://{name}"
    fsspec.filesystem("memory").pipe_file(name, array.to_cframe())
    return url, data


@pytest.mark.parametrize("carrier_kind", ["memory", "file-r", "file-a"])
def test_immutable_cache_chunk_reads_and_mutation_paths(tmp_path, carrier_kind):
    url, data = _remote_array("immutable-paths.b2nd", nchunks=3, chunk_size=1000)
    live = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_dir=tmp_path / "live")
    live[:1000]
    if carrier_kind == "memory":
        frozen = blosc2.from_cframe(live.to_cframe())
    else:
        path = tmp_path / "snapshot.b2nd"
        live.save(path)
        frozen = blosc2.open(path, mode=carrier_kind[-1])
    assert not frozen.is_cache_mutable
    before = frozen._carrier.to_cframe()
    retained = frozen.cache_bytes
    for nchunk in (0, 1, 1):
        chunk = frozen.get_chunk(nchunk)
        np.testing.assert_array_equal(
            np.frombuffer(blosc2.decompress(chunk), dtype=data.dtype),
            data[nchunk * 1000 : (nchunk + 1) * 1000],
        )
    chunk = asyncio.run(frozen.aget_chunk(2))
    np.testing.assert_array_equal(np.frombuffer(blosc2.decompress(chunk), dtype=data.dtype), data[2000:])
    assert frozen.read_cached(nchunk=0)[0]
    assert not frozen.cache_contains(nchunk=1)
    assert not frozen.cache_contains(nchunk=2)
    for operation in (
        lambda: frozen.fetch(slice(1000, 2000)),
        lambda: asyncio.run(frozen.afetch(slice(1000, 2000))),
        lambda: frozen.trim_cache(0),
    ):
        with pytest.raises(RuntimeError, match="immutable"):
            operation()
    assert frozen.cache_bytes == retained
    assert frozen._carrier.to_cframe() == before


@pytest.mark.parametrize("carrier_kind", ["memory", "file"])
def test_immutable_array_rejects_oversized_payload(tmp_path, carrier_kind):
    url, _ = _remote_array("immutable-budget.b2nd", nchunks=2, chunk_size=1000)
    live = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_dir=tmp_path / "live")
    live[:1000]
    carrier = blosc2.ndarray_from_cframe(live.to_cframe(), copy=True)
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["max_cache_bytes"] = 1
    carrier.schunk.vlmeta["b2o"] = payload
    if carrier_kind == "file":
        path = tmp_path / "oversized.b2nd"
        carrier.save(path)
        before = path.read_bytes()
        with pytest.raises(ValueError, match=r"immutable payload.*exceeds"):
            blosc2.open(path, mode="a")
        assert path.read_bytes() == before
    else:
        with pytest.raises(ValueError, match=r"immutable payload.*exceeds"):
            blosc2.from_cframe(carrier.to_cframe())


def test_cache_policy_validation(tmp_path):
    url, _ = _remote_array("policy.b2nd")

    none = blosc2.RemoteArray(url)
    assert none.cache_policy is blosc2.CachePolicy.NONE
    assert none.max_cache_bytes is None

    disk = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "cache.b2nd",
    )
    assert disk.max_cache_bytes == 256 * 2**20
    assert disk.cache_path == str(tmp_path / "cache.b2nd")

    with pytest.raises(TypeError, match="CachePolicy"):
        blosc2.RemoteArray(url, cache_policy="memory")
    with pytest.raises(ValueError, match="not applicable"):
        blosc2.RemoteArray(url, max_cache_bytes=1)
    disk_unlimited = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "unlimited.b2nd",
        max_cache_bytes=None,
    )
    assert disk_unlimited.max_cache_bytes is None
    with pytest.raises(TypeError, match="positive integer"):
        blosc2.RemoteArray(
            url,
            cache_policy=blosc2.CachePolicy.MEMORY,
            max_cache_bytes=None,
        )
    with pytest.raises(ValueError, match="positive integer"):
        blosc2.RemoteArray(
            url,
            cache_policy=blosc2.CachePolicy.DISK,
            cache_path=tmp_path / "neg.b2nd",
            max_cache_bytes=-1,
        )
    with pytest.raises(ValueError, match="requires cache_dir or cache_path"):
        blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK)
    with pytest.raises(ValueError, match="max_concurrency"):
        blosc2.RemoteArray(url, max_concurrency=0)
    with pytest.raises(TypeError, match="assume_immutable"):
        blosc2.RemoteArray(url, assume_immutable=1)


def test_assume_immutable_controls_identity_refresh(monkeypatch):
    url, _ = _remote_array("immutable-option.b2nd", nchunks=1, chunk_size=100)
    immutable = blosc2.RemoteArray(url)
    mutable = blosc2.open(url, lazy=True, assume_immutable=False)
    refreshed = []

    monkeypatch.setattr(
        immutable.src,
        "refresh_identity",
        lambda: (_ for _ in ()).throw(AssertionError("immutable source was refreshed")),
    )
    monkeypatch.setattr(mutable.src, "refresh_identity", lambda: refreshed.append(True))

    immutable._prepare_read()
    mutable._prepare_read()

    assert immutable.assume_immutable is True
    assert mutable.assume_immutable is False
    assert immutable.source["assume_immutable"] is True
    assert mutable.source["assume_immutable"] is False
    assert refreshed == [True]


def test_remote_array_operand_interface():
    url, _ = _remote_array("operand-interface.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteArray(url)

    assert type(proxy).__module__ == "blosc2.remote_array"
    assert type(proxy).__name__ == "RemoteArray"
    assert "RemoteArray" in blosc2.__all__
    assert proxy.ndim == 1
    assert len(proxy) == 100
    assert proxy.info is not None
    assert dict(proxy.info_items)["cache_policy"] == "NONE"

    expression = blosc2.lazyexpr("a + 1", operands={"a": proxy})
    assert url in dict(expression.info_items)["operands"].values()


def test_open_lazy_selects_remote_array(tmp_path):
    url, data = _remote_array("open-policy.b2nd")

    mem = blosc2.open(url, lazy=True)
    assert isinstance(mem, blosc2.RemoteArray)
    assert mem.cache_policy is blosc2.CachePolicy.MEMORY
    assert mem.cache_path is None
    assert mem.max_cache_bytes == 256 * 2**20
    np.testing.assert_array_equal(mem[:100_000], data[:100_000])

    none = blosc2.open(url, lazy=True, cache_policy=blosc2.CachePolicy.NONE)
    assert isinstance(none, blosc2.RemoteArray)
    assert none.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(none[:100_000], data[:100_000])

    mem_bounded = blosc2.open(url, lazy=True, max_cache_bytes=120_000)
    assert isinstance(mem_bounded, blosc2.RemoteArray)
    assert mem_bounded.cache_policy is blosc2.CachePolicy.MEMORY
    assert mem_bounded.max_cache_bytes == 120_000

    disk = blosc2.open(
        url,
        lazy=True,
        cache_path=tmp_path / "open-cache.b2nd",
        max_cache_bytes=120_000,
    )
    assert isinstance(disk, blosc2.RemoteArray)
    assert disk.cache_policy is blosc2.CachePolicy.DISK
    assert disk.max_cache_bytes == 120_000

    with pytest.raises(NotImplementedError, match="require lazy=True"):
        blosc2.open(url, lazy=False, max_cache_bytes=120_000)


def test_none_does_not_retain_remote_data():
    url, data = _remote_array("none.b2nd")
    proxy = blosc2.RemoteArray(url)

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
    proxy = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=120_000,
    )

    for start in (0, 100_000, 200_000):
        np.testing.assert_array_equal(proxy[start : start + 100_000], data[start : start + 100_000])
    assert proxy.cache_bytes <= 120_000
    assert cache_path.stat().st_size < 140_000

    reopened = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=120_000,
    )
    assert reopened.cache_status == "reused"
    np.testing.assert_array_equal(reopened[300_000:400_000], data[300_000:400_000])
    assert reopened.cache_bytes <= 120_000


def test_server_sparse_cache_reopens_and_exports_portable_carriers(tmp_path):
    url, data = _remote_array("server-sparse.b2nd", nchunks=3, chunk_size=100_000)
    runtime_path = tmp_path / "private-runtime"
    proxy = blosc2.RemoteArray.with_sparse_cache(url, runtime_path, max_cache_bytes=120_000)

    assert runtime_path.is_dir()
    assert proxy.runtime_cache_path == str(runtime_path)
    assert proxy.cache_path is None
    np.testing.assert_array_equal(proxy[:100_000], data[:100_000])

    reopened = blosc2.RemoteArray.with_sparse_cache(url, runtime_path, max_cache_bytes=120_000)
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:100_000], data[:100_000])
    assert reopened.traffic.requests == 0

    warm = blosc2.ndarray_from_cframe(reopened.to_cframe())
    cold = blosc2.ndarray_from_cframe(reopened.to_cframe(include_cache=False))
    assert warm.schunk.vlmeta.get("proxy-fetched")
    assert not cold.schunk.vlmeta.get("proxy-fetched")
    assert warm.schunk.vlmeta["b2o"] == cold.schunk.vlmeta["b2o"]


def test_server_sparse_handles_refresh_shared_fetched_state(tmp_path):
    url, data = _remote_array("server-shared.b2nd", nchunks=2, chunk_size=100)
    runtime_path = tmp_path / "shared-runtime"
    first = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)
    second = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)

    np.testing.assert_array_equal(first[:100], data[:100])
    second.traffic.reset()
    np.testing.assert_array_equal(second[:100], data[:100])
    assert second.traffic.requests == 0

    np.testing.assert_array_equal(second[100:], data[100:])
    first.traffic.reset()
    np.testing.assert_array_equal(first[100:], data[100:])
    assert first.traffic.requests == 0
    assert first.schunk.vlmeta["proxy-fetched"] == b"\x03"


def test_server_sparse_cache_recovers_a_dirty_generation(tmp_path):
    url, data = _remote_array("server-dirty.b2nd", nchunks=2, chunk_size=100)
    runtime_path = tmp_path / "dirty-runtime"
    proxy = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)
    np.testing.assert_array_equal(proxy[:100], data[:100])
    proxy.schunk.vlmeta["proxy-dirty"] = {"pid": -1, "version": 1}
    del proxy

    recovered = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)
    recovered.traffic.reset()
    np.testing.assert_array_equal(recovered[:100], data[:100])
    assert recovered.traffic.requests > 0
    assert "proxy-dirty" not in recovered.schunk.vlmeta


def test_server_sparse_cache_reuses_partial_blocks(tmp_path):
    data = np.random.default_rng(2).integers(0, 256, 200, dtype=np.uint8)
    array = blosc2.asarray(data, chunks=(100,), blocks=(10,))
    url = "memory://server-partial-blocks.b2nd"
    fsspec.filesystem("memory").pipe_file("server-partial-blocks.b2nd", array.to_cframe())
    runtime_path = tmp_path / "partial-runtime"

    proxy = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)
    np.testing.assert_array_equal(proxy[:10], data[:10])
    assert proxy.schunk.vlmeta.get("proxy-fetched-blocks")
    assert proxy.schunk.vlmeta["proxy-fetched-bpc"] == 10

    reopened = blosc2.RemoteArray.with_sparse_cache(url, runtime_path)
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:10], data[:10])
    assert reopened.traffic.requests == 0


def test_server_sparse_cache_invalidates_same_geometry_replacement(tmp_path):
    url, data = _remote_array("server-replaced.b2nd", nchunks=2, chunk_size=100)
    runtime_path = tmp_path / "replaced-runtime"
    proxy = blosc2.RemoteArray.with_sparse_cache(url, runtime_path, assume_immutable=False)
    np.testing.assert_array_equal(proxy[:100], data[:100])

    replacement = np.arange(200, dtype=np.uint8)
    array = blosc2.asarray(replacement, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("server-replaced.b2nd", array.to_cframe())

    proxy.traffic.reset()
    np.testing.assert_array_equal(proxy[:100], replacement[:100])
    assert proxy.traffic.requests > 0


def test_server_sparse_warm_seed_is_migrated_only_once(tmp_path):
    url, data = _remote_array("server-seed.b2nd", nchunks=2, chunk_size=100)
    seed = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "seed.b2nd",
        max_cache_bytes=None,
    )
    np.testing.assert_array_equal(seed[:100], data[:100])

    runtime_path = tmp_path / "seed-runtime"
    runtime = blosc2.RemoteArray.with_sparse_cache(
        url, runtime_path, carrier=seed._carrier, max_cache_bytes=1
    )
    runtime.traffic.reset()
    np.testing.assert_array_equal(runtime[:100], data[:100])
    assert runtime.traffic.requests == 0
    assert runtime.cache_bytes == 0
    del runtime

    reopened = blosc2.RemoteArray.with_sparse_cache(
        url, runtime_path, carrier=seed._carrier, max_cache_bytes=1
    )
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:100], data[:100])
    assert reopened.traffic.requests > 0


def test_server_sparse_rejects_a_seed_from_another_source(tmp_path):
    first_url, _ = _remote_array("server-first-seed.b2nd", nchunks=1, chunk_size=100)
    second_url, _ = _remote_array("server-second-seed.b2nd", nchunks=1, chunk_size=100)
    seed = blosc2.RemoteArray(
        first_url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "other-seed.b2nd",
    )

    runtime_path = tmp_path / "wrong-seed-runtime"
    with pytest.raises(ValueError, match="different remote source"):
        blosc2.RemoteArray.with_sparse_cache(second_url, runtime_path, carrier=seed._carrier)
    assert not runtime_path.exists()


def test_interrupted_fetch_leaves_a_reusable_carrier(tmp_path):
    url, data = _remote_array("interrupted.b2nd", nchunks=3, chunk_size=100_000)
    cache_path = tmp_path / "interrupted-proxy.b2nd"
    proxy = blosc2.RemoteArray(
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
    original = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "live-proxy.b2nd",
        max_cache_bytes=120_000,
    )
    original[:100_000]

    carrier = blosc2.ndarray_from_cframe(original.to_cframe())
    assert carrier.schunk.meta["b2o"] == {"kind": "remote_array", "version": 1}
    assert carrier.schunk.vlmeta["b2o"] == {
        "kind": "remote_array",
        "version": 1,
        "source": {
            "kind": "fsspec",
            "version": 1,
            "urlpath": url,
            "assume_immutable": True,
        },
        "cache_policy": "disk",
        "max_cache_bytes": 120_000,
        "mutable": False,
    }
    assert carrier.schunk.vlmeta.get("proxy-cache-sizes")

    cold_carrier = blosc2.ndarray_from_cframe(original.to_cframe(include_cache=False))
    assert cold_carrier.schunk.vlmeta["b2o"] == carrier.schunk.vlmeta["b2o"]
    assert not cold_carrier.schunk.vlmeta.get("proxy-cache-sizes", {})
    assert original.cache_bytes > 0

    warm_path = tmp_path / "warm.b2nd"
    original.save(warm_path)
    restored = blosc2.open(warm_path, mode="r")
    assert isinstance(restored, blosc2.RemoteArray)
    assert restored.cache_policy is blosc2.CachePolicy.DISK
    restored.traffic.reset()
    np.testing.assert_array_equal(restored[:100_000], data[:100_000])
    assert restored.traffic.requests == 0

    cold_path = tmp_path / "cold.b2nd"
    original.save(cold_path, include_cache=False, mutable=True)
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

    immutable_path = tmp_path / "immutable.b2nd"
    original.save(immutable_path, include_cache=False)
    imm = blosc2.open(immutable_path, mode="a")
    imm.traffic.reset()
    np.testing.assert_array_equal(imm[:100_000], data[:100_000])
    assert imm.traffic.requests > 0
    imm_reopened = blosc2.open(immutable_path, mode="r")
    imm_reopened.traffic.reset()
    np.testing.assert_array_equal(imm_reopened[:100_000], data[:100_000])
    assert imm_reopened.traffic.requests > 0


def test_save_returns_written_path(tmp_path):
    url, _ = _remote_array("save-return.b2nd")
    original = blosc2.RemoteArray(
        url, cache_policy=blosc2.CachePolicy.DISK, cache_path=tmp_path / "carrier.b2nd"
    )
    original[:]
    destination = tmp_path / "out.b2nd"
    assert original.save(destination) == str(destination)


def test_dict_store_externalizes_disk_remote_array(tmp_path):
    from blosc2.dict_store import DictStore

    url, data = _remote_array("dictstore-external.b2nd")
    array = blosc2.RemoteArray(
        url, cache_policy=blosc2.CachePolicy.DISK, cache_path=tmp_path / "carrier.b2nd"
    )
    array[:]

    assert DictStore._is_external_value(array)
    assert DictStore._external_ext(array) == ".b2nd"

    store_path = tmp_path / "store.b2d"
    with blosc2.DictStore(store_path, mode="w") as store:
        store["/leaf"] = array
    # The local DISK carrier, not the remote URL, is what gets externalized.
    assert (store_path / "leaf.b2nd").is_file()

    with blosc2.DictStore(store_path, mode="r") as store:
        np.testing.assert_array_equal(store["/leaf"][:], data)


def test_reference_rejects_changed_source_geometry(tmp_path):
    url, _ = _remote_array("changed-geometry.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "changed-reference.b2nd"
    blosc2.RemoteArray(url, assume_immutable=False).save(path)

    replacement = blosc2.arange(200, dtype=np.uint8, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("changed-geometry.b2nd", replacement.to_cframe())
    with pytest.raises(ValueError, match="geometry no longer matches"):
        blosc2.open(path, mode="r")


def test_open_reference_rejects_geometry_changed_before_read(tmp_path):
    url, _ = _remote_array("changed-after-open.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "changed-after-open-reference.b2nd"
    blosc2.RemoteArray(url, assume_immutable=False).save(path)
    restored = blosc2.open(path, mode="r")

    replacement = blosc2.arange(200, dtype=np.uint8, chunks=(100,), blocks=(100,))
    fsspec.filesystem("memory").pipe_file("changed-after-open.b2nd", replacement.to_cframe())

    with pytest.raises(ValueError, match="geometry no longer matches"):
        restored[:]


def test_runtime_cache_is_invalidated_after_same_geometry_replacement(tmp_path):
    url, data = _remote_array("same-geometry-disk.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "same-geometry-cache.b2nd",
        assume_immutable=False,
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
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteArray(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "unknown_policy"

    with pytest.raises(ValueError, match="unsupported cache policy"):
        decode_b2object_payload(payload, carrier=carrier)


def test_reference_rejects_memory_policy_without_positive_limit_in_payload():
    url, _ = _remote_array("bad-memory-policy.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteArray(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "memory"
    payload["max_cache_bytes"] = None

    with pytest.raises(ValueError, match="persisted MEMORY RemoteArray requires positive max_cache_bytes"):
        decode_b2object_payload(payload, carrier=carrier)


def test_reference_accepts_disk_policy_with_none_limit_in_payload(tmp_path):
    url, data = _remote_array("unlimited-payload.b2nd", nchunks=2, chunk_size=50)
    cache_path = tmp_path / "unlimited-carrier.b2nd"
    proxy = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=cache_path,
        max_cache_bytes=None,
    )
    assert proxy.max_cache_bytes is None
    carrier_raw = blosc2.ndarray_from_cframe(proxy.to_cframe())
    payload = dict(carrier_raw.schunk.vlmeta["b2o"])
    decoded_carrier = decode_b2object_payload(payload, carrier=carrier_raw)
    assert payload["max_cache_bytes"] is None
    assert decoded_carrier.max_cache_bytes is None

    reopened = blosc2.open(cache_path)
    assert isinstance(reopened, blosc2.RemoteArray)
    assert reopened.cache_policy is blosc2.CachePolicy.DISK
    assert reopened.max_cache_bytes is None
    np.testing.assert_array_equal(reopened[:], data)


@pytest.mark.parametrize("invalid_limit", [False, True, 0, -10, "1000"])
def test_reference_rejects_invalid_disk_limit_in_payload(invalid_limit):
    url, _ = _remote_array("bad-disk-limit.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteArray(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["cache_policy"] = "disk"
    payload["max_cache_bytes"] = invalid_limit

    with pytest.raises(
        ValueError, match="persisted DISK RemoteArray requires positive max_cache_bytes or None"
    ):
        decode_b2object_payload(payload, carrier=carrier)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("kind", "unknown", "unsupported RemoteArray source kind"),
        ("version", 2, "unsupported RemoteArray source descriptor"),
    ],
)
def test_reference_rejects_unknown_source_descriptor(field, value, error):
    url, _ = _remote_array(f"bad-source-{field}.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteArray(url).to_cframe())
    payload = dict(carrier.schunk.vlmeta["b2o"])
    payload["source"] = dict(payload["source"], **{field: value})

    with pytest.raises(ValueError, match=error):
        decode_b2object_payload(payload, carrier=carrier)


@pytest.mark.parametrize("field", ["auth_token", "storage_options"])
def test_reference_rejects_secret_or_runtime_source_fields(field):
    url, _ = _remote_array(f"bad-source-{field}.b2nd", nchunks=1, chunk_size=100)
    carrier = blosc2.ndarray_from_cframe(blosc2.RemoteArray(url).to_cframe())
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
    remote = blosc2.RemoteArray(
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
        "assume_immutable": True,
    }

    restored = blosc2.from_cframe(remote.to_cframe())
    assert isinstance(restored, blosc2.RemoteArray)
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
    remote = blosc2.RemoteArray(
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
    remote = blosc2.RemoteArray(
        blosc2.URLPath("@public/cache.b2nd", urlbase="https://example.org/c2"),
        cache_policy=blosc2.CachePolicy.DISK,
        cache_path=tmp_path / "caterva2-cache.b2nd",
    )

    np.testing.assert_array_equal(remote[:5], data[:5])
    np.testing.assert_array_equal(remote[:5], data[:5])
    assert calls == [0]


def test_remote_array_is_a_persistable_lazyexpr_operand():
    url, data = _remote_array("operand.b2nd", nchunks=1, chunk_size=100)
    remote = blosc2.RemoteArray(url)
    expression = blosc2.lazyexpr("a + 1", operands={"a": remote})

    restored = blosc2.from_cframe(expression.to_cframe())
    assert any(isinstance(operand, blosc2.RemoteArray) for operand in restored.operands.values())
    np.testing.assert_array_equal(restored[:], data + 1)


def test_objectarray_msgpack_supports_remote_array():
    url, data = _remote_array("objectarray-remote-proxy.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteArray(url)

    objects = blosc2.ObjectArray()
    objects.append(proxy)
    restored = objects[0]

    assert isinstance(restored, blosc2.RemoteArray)
    assert restored.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(restored[:], data)


def test_batcharray_msgpack_supports_remote_array():
    url, data = _remote_array("batcharray-remote-proxy.b2nd", nchunks=1, chunk_size=100)
    batches = blosc2.BatchArray()
    batches.append([blosc2.RemoteArray(url)])

    restored = batches[0][0]
    assert isinstance(restored, blosc2.RemoteArray)
    assert restored.cache_policy is blosc2.CachePolicy.NONE
    np.testing.assert_array_equal(restored[:], data)


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_get_chunk_for_each_policy(tmp_path, policy):
    url, data = _remote_array(f"get-chunk-{policy.value}.b2nd", nchunks=2, chunk_size=100)
    kwargs = {"cache_path": tmp_path / "get-chunk-cache.b2nd"} if policy is blosc2.CachePolicy.DISK else {}
    proxy = blosc2.RemoteArray(url, cache_policy=policy, **kwargs)

    chunk = proxy.get_chunk(1)
    np.testing.assert_array_equal(np.frombuffer(blosc2.decompress2(chunk), dtype=np.uint8), data[100:])


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_aget_chunk_for_each_policy(tmp_path, policy):
    url, data = _remote_array(f"aget-chunk-{policy.value}.b2nd", nchunks=2, chunk_size=100)
    kwargs = {"cache_path": tmp_path / "aget-chunk-cache.b2nd"} if policy is blosc2.CachePolicy.DISK else {}
    proxy = blosc2.RemoteArray(url, cache_policy=policy, **kwargs)

    chunk = asyncio.run(proxy.aget_chunk(1))
    np.testing.assert_array_equal(np.frombuffer(blosc2.decompress2(chunk), dtype=np.uint8), data[100:])


def test_disk_cache_survives_reopen_without_remote_data_traffic(tmp_path):
    url, data = _remote_array("disk-reuse.b2nd", nchunks=2, chunk_size=100_000)
    cache_path = tmp_path / "disk-reuse-cache.b2nd"
    first = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    np.testing.assert_array_equal(first[:100_000], data[:100_000])

    reopened = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    reopened.traffic.reset()
    np.testing.assert_array_equal(reopened[:100_000], data[:100_000])
    assert reopened.traffic.requests == 0


def test_save_after_disk_cache_use_preserves_or_strips_cache(tmp_path):
    url, data = _remote_array("save-after-disk.b2nd", nchunks=2, chunk_size=100_000)
    proxy = blosc2.RemoteArray(
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
    blosc2.RemoteArray(url).save(path)

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
        blosc2.RemoteArray(url)


@pytest.mark.parametrize(
    "url", ["https://example.org/data.b2nd?token=secret", "https://user@example.org/data.b2nd"]
)
def test_fsspec_refs_reject_credentials(url):
    with pytest.raises(ValueError):
        blosc2.Ref.fsspec_ref(url)


def test_memory_cache_eviction_and_retention():
    url, data = _remote_array("mem-evict.b2nd", nchunks=5, chunk_size=20_000)
    proxy = blosc2.RemoteArray(
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
    proxy = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
    )
    cached_container = proxy.fetch(slice(0, 10_000))
    assert cached_container is proxy
    assert proxy.cache_bytes > 0

    async_container = asyncio.run(proxy.afetch(slice(10_000, 20_000)))
    assert async_container is proxy


def test_memory_proxy_save_and_reopen(tmp_path):
    url, data = _remote_array("mem-save.b2nd", nchunks=2, chunk_size=10_000)
    proxy = blosc2.RemoteArray(
        url,
        cache_policy=blosc2.CachePolicy.MEMORY,
        max_cache_bytes=100_000,
    )
    save_path = tmp_path / "saved_memory_proxy.b2nd"
    proxy.save(save_path)

    reopened = blosc2.open(save_path, mode="r")
    assert isinstance(reopened, blosc2.RemoteArray)
    assert reopened.cache_policy is blosc2.CachePolicy.MEMORY
    assert reopened.max_cache_bytes == 100_000
    assert reopened.cache_path is None
    np.testing.assert_array_equal(reopened[:], data)


def test_remote_array_fetch_rejects_none_policy():
    url, _ = _remote_array("none-fetch.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.NONE)
    with pytest.raises(
        NotImplementedError, match=r"fetch requires CachePolicy\.DISK or CachePolicy\.MEMORY"
    ):
        proxy.fetch()
    with pytest.raises(
        NotImplementedError, match=r"afetch requires CachePolicy\.DISK or CachePolicy\.MEMORY"
    ):
        asyncio.run(proxy.afetch())


def test_prefetch_over_limit_and_materialize():
    url, data = _remote_array("prefetch-limit.b2nd", nchunks=3, chunk_size=10_000)
    proxy = blosc2.open(url, lazy=True, max_cache_bytes=12_000)
    assert proxy.fetch() is proxy
    assert proxy.cache_bytes <= 12_000
    assert asyncio.run(proxy.afetch()) is proxy
    np.testing.assert_array_equal(proxy.materialize()[:], data)
    assert proxy.cache_bytes <= 12_000


def test_open_error_preserves_existing_file(tmp_path, monkeypatch):
    url, _ = _remote_array("open-error.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "existing.b2nd"
    blosc2.arange(100).save(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("transient open failure")

    monkeypatch.setattr(blosc2.blosc2_ext, "open", fail)
    with pytest.raises(OSError, match="transient"):
        blosc2.open(url, lazy=True, cache_path=path)
    assert path.read_bytes() == before


def test_completed_caterva2_reference_refreshes_identity(monkeypatch):
    state = {"nonce": "old"}

    def info(*args, **kwargs):
        return {
            "shape": [10],
            "chunks": [5],
            "blocks": [5],
            "dtype": "<i4",
            "schunk": {
                "cparams": dict(blosc2.cparams_dflts),
                "cbytes": 40,
                "vlmeta": {"fill_nonce": state["nonce"], "fill_state": "complete"},
            },
        }

    monkeypatch.setattr(blosc2_c2array, "info", info)
    proxy = blosc2.RemoteArray(
        blosc2.URLPath("@public/a.b2nd", urlbase="https://example.org"), assume_immutable=False
    )
    old = proxy.src.stamp
    state["nonce"] = "replacement"
    proxy._prepare_read()
    assert proxy.src.stamp != old


@pytest.mark.parametrize("policy", list(blosc2.CachePolicy))
def test_memory_export_policy(tmp_path, policy):
    url, data = _remote_array("export-policy.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.open(url, lazy=True)
    proxy[:]
    frame = proxy.to_cframe(cache_policy=policy)
    carrier = blosc2.ndarray_from_cframe(frame)
    assert not carrier.schunk.vlmeta.get("proxy-fetched")
    assert not carrier.schunk.vlmeta.get("proxy-cache-sizes")
    path = tmp_path / "export.b2nd"
    proxy.save(path, cache_policy=policy)
    reopened = blosc2.open(path)
    assert reopened.cache_policy is policy
    np.testing.assert_array_equal(reopened[:], data)
    assert proxy.cache_policy is blosc2.CachePolicy.MEMORY
    assert proxy.cache_bytes > 0


def test_runtime_signed_url_is_not_exportable():
    url, data = _remote_array("signed.b2nd?token=secret", nchunks=1, chunk_size=100)
    proxy = blosc2.open(url, lazy=True)
    np.testing.assert_array_equal(proxy[:], data)
    with pytest.raises(ValueError, match="credential-like"):
        proxy.to_cframe()
    with pytest.raises(ValueError, match="credential-like"):
        _ = proxy.source


def test_interrupted_memory_fetch_enforces_limit(monkeypatch):
    url, _ = _remote_array("interrupted-memory.b2nd", nchunks=3, chunk_size=10_000)
    proxy = blosc2.open(url, lazy=True, max_cache_bytes=100)
    original = proxy.src.get_chunk

    def interrupted(nchunk):
        if nchunk == 1:
            raise RuntimeError("interrupted")
        return original(nchunk)

    monkeypatch.setattr(proxy.src, "get_chunk", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        proxy.fetch(max_concurrency=1)
    assert proxy.cache_bytes <= 100


def test_concurrent_reads_with_eviction():
    from concurrent.futures import ThreadPoolExecutor

    url, data = _remote_array("concurrent-memory.b2nd", nchunks=3, chunk_size=10_000)
    proxy = blosc2.open(url, lazy=True, max_cache_bytes=100)
    slices = [slice(0, 20_000), slice(10_000, 30_000)] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(proxy.__getitem__, slices))
    for item, result in zip(slices, results, strict=True):
        np.testing.assert_array_equal(result, data[item])
    assert proxy.cache_bytes <= 100


def test_legacy_cache_url_open_preserves_file(tmp_path):
    url, data = _remote_array("legacy-preserve.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "legacy.b2nd"
    legacy = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=path, mode="a")
    np.testing.assert_array_equal(legacy[:], data)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="open legacy Proxy caches directly"):
        blosc2.open(url, lazy=True, cache_path=path)
    assert path.read_bytes() == before
    np.testing.assert_array_equal(blosc2.open(path)[:], data)


def test_memory_warm_export_is_cold():
    url, _ = _remote_array("warm-memory.b2nd", nchunks=1, chunk_size=100)
    proxy = blosc2.open(url, lazy=True)
    proxy[:]
    carrier = blosc2.ndarray_from_cframe(proxy.to_cframe())
    assert carrier.schunk.vlmeta["b2o"]["cache_policy"] == "memory"
    assert not carrier.schunk.vlmeta.get("proxy-fetched")
    assert proxy.cache_bytes > 0


def test_cold_export_cannot_overwrite_live_carrier(tmp_path):
    url, _ = _remote_array("same-export.b2nd", nchunks=1, chunk_size=100)
    path = tmp_path / "live.b2nd"
    proxy = blosc2.open(url, lazy=True, cache_path=path)
    proxy[:]
    before = path.read_bytes()
    for kwargs in ({"include_cache": False}, {"cache_policy": blosc2.CachePolicy.NONE}):
        with pytest.raises(ValueError, match="different destination"):
            proxy.save(path, **kwargs)
        assert path.read_bytes() == before


def test_unlimited_disk_cache_does_not_evict(tmp_path):
    url, data = _remote_array("unlimited-disk.b2nd", nchunks=5, chunk_size=20)
    carrier_path = tmp_path / "unlimited.b2nd"
    proxy = blosc2.open(url, lazy=True, cache_path=carrier_path, max_cache_bytes=None)
    assert proxy.max_cache_bytes is None

    # Read all chunks
    np.testing.assert_array_equal(proxy[:], data)
    assert proxy.cache_bytes > 0
    initial_cache_bytes = proxy.cache_bytes

    # Access individual chunks again, verify no eviction occurred
    for i in range(5):
        np.testing.assert_array_equal(proxy[i * 20 : (i + 1) * 20], data[i * 20 : (i + 1) * 20])
    assert proxy.cache_bytes == initial_cache_bytes

    # Carrier has all chunks warm
    carrier = blosc2.ndarray_from_cframe(proxy.to_cframe())
    assert carrier.schunk.cbytes == initial_cache_bytes
    assert carrier.schunk.vlmeta.get("proxy-fetched") == b"\x1f"
    assert carrier.schunk.vlmeta["b2o"]["max_cache_bytes"] is None

    # Export tests
    # 1. Exporting with cache_policy=DISK preserves None limit
    disk_export = blosc2.ndarray_from_cframe(proxy.to_cframe(cache_policy=blosc2.CachePolicy.DISK))
    assert disk_export.schunk.vlmeta["b2o"]["cache_policy"] == "disk"
    assert disk_export.schunk.vlmeta["b2o"]["max_cache_bytes"] is None

    # 2. Exporting with cache_policy=MEMORY falls back to default limit
    mem_export = blosc2.ndarray_from_cframe(proxy.to_cframe(cache_policy=blosc2.CachePolicy.MEMORY))
    assert mem_export.schunk.vlmeta["b2o"]["cache_policy"] == "memory"
    assert mem_export.schunk.vlmeta["b2o"]["max_cache_bytes"] == blosc2.remote_array.DEFAULT_DISK_CACHE_BYTES

    # 3. Exporting with cache_policy=NONE sets limit to None
    none_export = blosc2.ndarray_from_cframe(proxy.to_cframe(cache_policy=blosc2.CachePolicy.NONE))
    assert none_export.schunk.vlmeta["b2o"]["cache_policy"] == "none"
    assert none_export.schunk.vlmeta["b2o"]["max_cache_bytes"] is None


def test_authorized_sparse_snapshot_never_reopens(tmp_path, monkeypatch):
    url, data = _remote_array("authorized-sparse.b2nd", nchunks=3, chunk_size=10000)
    source = blosc2.FsspecNDSource(url)
    descriptor = {"kind": "fsspec", "version": 1, "urlpath": url, "assume_immutable": True}

    def forbidden(*args, **kwargs):
        raise AssertionError("authorized transport was reopened or refreshed")

    monkeypatch.setattr(blosc2.RemoteArray, "_open_source", forbidden)
    monkeypatch.setattr(source, "refresh_stamp", forbidden, raising=False)
    monkeypatch.setattr(source, "refresh_identity", forbidden, raising=False)
    path = tmp_path / "authorized"
    proxy = blosc2.RemoteArray.with_sparse_cache(
        source, path, source_descriptor=descriptor, max_cache_bytes=None
    )
    assert proxy.src is source
    assert proxy.read_cached(slice(0, 10000)) == (False, None)
    np.testing.assert_array_equal(proxy[:10000], data[:10000])
    hit, result = proxy.read_cached(slice(0, 10000))
    assert hit
    np.testing.assert_array_equal(result, data[:10000])
    assert proxy.cached_payload_bytes >= 10000
    assert proxy.trim_cache(0, max_chunks=1) == (0,)
    assert not proxy.cache_contains(nchunk=0)
    np.testing.assert_array_equal(proxy[:], data)
    del proxy
    evicted, remaining = blosc2.RemoteArray.trim_sparse_cache(path, 0, max_chunks=1)
    assert len(evicted) == 1
    assert remaining >= 20000
    with pytest.raises(ValueError, match="does not match"):
        blosc2.RemoteArray.with_sparse_cache(
            source, path, source_descriptor=dict(descriptor, urlpath="memory://other")
        )


def test_sparse_seed_with_dirty_marker_is_not_imported(tmp_path):
    url, data = _remote_array("dirty-seed.b2nd", nchunks=2, chunk_size=10000)
    seed = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=tmp_path / "seed.b2nd")
    np.testing.assert_array_equal(seed[:10000], data[:10000])
    seed.schunk.vlmeta["proxy-dirty"] = {"version": 1}
    runtime = blosc2.RemoteArray.with_sparse_cache(url, tmp_path / "runtime", carrier=seed.cache)
    assert not runtime.cache_contains(nchunk=0)
    np.testing.assert_array_equal(runtime[:], data)


def test_remote_array_metadata_access_and_caching():
    data = np.arange(20_000, dtype=np.int32)
    array = blosc2.asarray(
        data,
        chunks=(10_000,),
        blocks=(5_000,),
        meta={"experiment": {"id": 42, "user": "alice"}},
    )
    array.vlmeta["notes"] = {"status": "calibrated", "tags": ["optical", "v2"]}
    url = "memory://metadata-test.b2nd"
    fsspec.filesystem("memory").pipe_file("metadata-test.b2nd", array.to_cframe())

    proxy = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.MEMORY)

    # Fixed-length metadata
    meta = proxy.meta
    assert isinstance(meta, blosc2.RemoteMetadataMapping)
    assert "b2nd" in meta
    assert "experiment" in meta
    assert meta["experiment"] == {"id": 42, "user": "alice"}
    assert meta.get("experiment") == {"id": 42, "user": "alice"}
    assert meta.get("missing", 999) == 999
    assert len(meta) >= 2
    assert meta[:] == meta.getall()
    assert meta.copy() == meta.getall()
    assert meta == meta.getall()
    assert repr(meta) == repr(meta.getall())
    assert str(meta) == str(meta.getall())
    with pytest.raises(TypeError):
        meta["new_key"] = 1
    with pytest.raises(TypeError):
        del meta["experiment"]
    with pytest.raises(NotImplementedError, match="Slicing is not supported"):
        _ = meta[0:1]

    # Variable-length metadata
    vlmeta = proxy.vlmeta
    assert proxy.attrs is vlmeta
    assert isinstance(vlmeta, blosc2.RemoteMetadataMapping)
    assert "notes" in vlmeta
    assert vlmeta["notes"] == {"status": "calibrated", "tags": ["optical", "v2"]}
    assert vlmeta.get("notes") == {"status": "calibrated", "tags": ["optical", "v2"]}
    assert len(vlmeta) == 1
    assert vlmeta[:] == {"notes": {"status": "calibrated", "tags": ["optical", "v2"]}}
    with pytest.raises(TypeError):
        vlmeta["new_key"] = 1
    with pytest.raises(TypeError):
        del vlmeta["notes"]

    # In-memory caching: subsequent accesses issue 0 network traffic
    proxy.traffic.reset()
    _ = proxy.meta["experiment"]
    _ = proxy.vlmeta["notes"]
    _ = proxy.meta[:]
    _ = proxy.vlmeta[:]
    _ = proxy.attrs["notes"]
    assert proxy.traffic.requests == 0


def test_remote_array_disk_cache_persists_metadata_and_offline_reopen(tmp_path):
    data = np.arange(20_000, dtype=np.int32)
    array = blosc2.asarray(
        data,
        chunks=(10_000,),
        blocks=(5_000,),
        meta={"exp_header": "test_disk_meta"},
    )
    array.vlmeta["exp_trailer"] = [1, 2, 3]
    url = "memory://disk-meta-test.b2nd"
    fsspec.filesystem("memory").pipe_file("disk-meta-test.b2nd", array.to_cframe())

    cache_path = tmp_path / "disk-meta-cache.b2nd"
    first = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    assert first.meta["exp_header"] == "test_disk_meta"
    assert first.vlmeta["exp_trailer"] == [1, 2, 3]

    # Reopen existing DISK cache
    reopened = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.DISK, cache_path=cache_path)
    reopened.traffic.reset()
    assert reopened.meta["exp_header"] == "test_disk_meta"
    assert reopened.vlmeta["exp_trailer"] == [1, 2, 3]
    # Reading metadata from existing carrier issues 0 remote requests!
    assert reopened.traffic.requests == 0

    # Open carrier directly using blosc2.open()
    opened = blosc2.open(cache_path)
    assert isinstance(opened, blosc2.RemoteArray)
    opened.traffic.reset()
    assert opened.meta["exp_header"] == "test_disk_meta"
    assert opened.vlmeta["exp_trailer"] == [1, 2, 3]
    assert opened.traffic.requests == 0


def test_remote_array_export_preserves_metadata(tmp_path):
    data = np.arange(10_000, dtype=np.int32)
    array = blosc2.asarray(
        data,
        chunks=(5_000,),
        blocks=(2_500,),
        meta={"export_meta": {"model": "sensor_a"}},
    )
    array.vlmeta["export_vlmeta"] = {"quality": "high"}
    url = "memory://export-meta-test.b2nd"
    fsspec.filesystem("memory").pipe_file("export-meta-test.b2nd", array.to_cframe())

    proxy = blosc2.RemoteArray(url, cache_policy=blosc2.CachePolicy.MEMORY)

    # 1. Export via save()
    save_path = tmp_path / "saved_carrier.b2nd"
    proxy.save(save_path)
    saved_proxy = blosc2.open(save_path)
    assert isinstance(saved_proxy, blosc2.RemoteArray)
    assert saved_proxy.meta["export_meta"] == {"model": "sensor_a"}
    assert saved_proxy.vlmeta["export_vlmeta"] == {"quality": "high"}

    # 2. Export via to_cframe()
    cframe = proxy.to_cframe()
    restored_proxy = blosc2.from_cframe(cframe)
    assert isinstance(restored_proxy, blosc2.RemoteArray)
    assert restored_proxy.meta["export_meta"] == {"model": "sensor_a"}
    assert restored_proxy.vlmeta["export_vlmeta"] == {"quality": "high"}


def test_remote_array_invalidation_refetches_metadata():
    data1 = np.arange(10_000, dtype=np.int32)
    array1 = blosc2.asarray(data1, chunks=(5_000,), blocks=(2_500,), meta={"v": 1})
    array1.vlmeta["note"] = "version 1"
    url = "memory://mutable-meta.b2nd"
    fsspec.filesystem("memory").pipe_file("mutable-meta.b2nd", array1.to_cframe())

    proxy = blosc2.RemoteArray(url, assume_immutable=False)
    assert proxy.meta["v"] == 1
    assert proxy.vlmeta["note"] == "version 1"

    # Replace remote array with new version
    data2 = np.arange(10_000, dtype=np.int32) + 100
    array2 = blosc2.asarray(data2, chunks=(5_000,), blocks=(2_500,), meta={"v": 2})
    array2.vlmeta["note"] = "version 2"
    fsspec.filesystem("memory").pipe_file("mutable-meta.b2nd", array2.to_cframe())

    # proxy detects the change and returns updated metadata
    assert proxy.meta["v"] == 2
    assert proxy.vlmeta["note"] == "version 2"
    assert proxy.attrs["note"] == "version 2"


def test_zarr_source_vlmeta():
    zarr = pytest.importorskip("zarr")
    zarr_url = "memory://test_attrs.zarr"
    mapper = fsspec.get_mapper(zarr_url)
    z_arr = zarr.open_array(
        store=mapper,
        mode="w",
        shape=(100,),
        chunks=(50,),
        dtype="i4",
    )
    z_arr[:] = np.arange(100, dtype=np.int32)
    z_arr.attrs["author"] = "researcher"
    z_arr.attrs["dataset_id"] = 12345

    src = blosc2.ZarrNDSource(zarr_url)
    assert src.vlmeta["author"] == "researcher"
    assert src.vlmeta["dataset_id"] == 12345

    proxy = blosc2.RemoteArray(zarr_url, source_format="zarr")
    assert proxy.vlmeta["author"] == "researcher"
    assert proxy.vlmeta["dataset_id"] == 12345
    assert proxy.attrs is proxy.vlmeta


@pytest.mark.parametrize(
    "attrs", [None, {}, {"user_tag": {"nested": [1, True]}, "fill_state": "user value"}]
)
def test_caterva2_vlmeta_filters_internal_keys(monkeypatch, attrs):
    requests = 0

    def fake_info(path, urlbase, params=None, headers=None, model=None, auth_token=None, traffic=None):
        nonlocal requests
        requests += 1
        result = {
            "shape": [10],
            "chunks": [5],
            "blocks": [5],
            "dtype": np.dtype(np.int32).str,
            "schunk": {
                "cparams": dict(blosc2.cparams_dflts),
                "vlmeta": {
                    "fill_nonce": "secret_nonce_123",
                    "fill_state": "complete",
                    "user_tag": "public_data",
                },
            },
        }
        if attrs is not None:
            result["attrs"] = attrs
        return result

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    remote = blosc2.RemoteArray(blosc2.URLPath("@public/test-vlmeta.b2nd", urlbase="https://example.org/c2"))

    # Caterva2 fixed metalayers are not supported by api/info, returns empty
    assert remote.meta == {}
    # Only legacy responses need client-side filtering of internal keys.
    expected = {"user_tag": "public_data"} if attrs is None else attrs
    assert remote.vlmeta == expected
    assert remote.attrs is remote.vlmeta
    assert remote.src.attrs == (remote.src.vlmeta if attrs is None else attrs)
    assert remote.src.vlmeta["fill_nonce"] == "secret_nonce_123"

    # Export to carrier and verify roundtrip doesn't persist internal keys
    cframe = remote.to_cframe()
    restored = blosc2.from_cframe(cframe)
    assert restored.vlmeta == expected

    previous_requests = requests
    _ = remote.src.attrs
    assert requests == previous_requests
    attrs = {"refreshed": True}
    remote.src._forget_index()
    assert remote.src.attrs == attrs
    assert requests == previous_requests + 1


def test_remote_array_filters_carrier_internal_metalayers():
    data = np.arange(10, dtype=np.int32)
    meta = {
        "user_key": "val1",
        "b2o": {"kind": "carrier"},
        "proxy": {"foo": "bar"},
        "proxy-source": {"bar": "baz"},
    }
    carrier = blosc2.asarray(data, chunks=(5,), blocks=(5,), meta=meta)

    url = "memory://carrier-leak-test.b2nd"
    fsspec.filesystem("memory").pipe_file("carrier-leak-test.b2nd", carrier.to_cframe())

    proxy = blosc2.RemoteArray(url)
    assert "user_key" in proxy.meta
    assert "b2nd" in proxy.meta
    assert "b2o" not in proxy.meta
    assert "proxy" not in proxy.meta
    assert "proxy-source" not in proxy.meta


def test_remote_array_metadata_slicing_and_caching():
    data = np.arange(10, dtype=np.int32)
    arr = blosc2.asarray(data, chunks=(5,), blocks=(5,), meta={"foo": "bar"})
    arr.vlmeta["desc"] = "test"
    url = "memory://slice-test.b2nd"
    fsspec.filesystem("memory").pipe_file("slice-test.b2nd", arr.to_cframe())

    proxy = blosc2.RemoteArray(url)
    # [:] full slice returns copy of dict
    assert proxy.meta[:] == dict(proxy.meta)
    assert proxy.vlmeta[:] == dict(proxy.vlmeta)

    # [::2] stepped slice raises NotImplementedError
    with pytest.raises(NotImplementedError, match="Slicing is not supported, unless"):
        _ = proxy.meta[::2]
    with pytest.raises(NotImplementedError, match="Slicing is not supported, unless"):
        _ = proxy.vlmeta[::2]

    # Repeated access returns the cached mapping instance without re-wrapping
    meta1 = proxy.meta
    meta2 = proxy.meta
    assert meta1 is meta2

    vlmeta1 = proxy.vlmeta
    vlmeta2 = proxy.vlmeta
    assert vlmeta1 is vlmeta2


def test_remote_array_metadata_complex_and_containers():
    from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb

    payload = {
        "attr0": True,
        "attr1": 11,
        "attr2": 3.5,
        "attr3": 1.0 + 2.0j,
        "attr4": "str_4",
        "attr5": b"bytes_5",
        "attr6": [1, 2, 3],
        "attr7": (4, 5, 6),
        "attr8": {"key": "val_8", "index": 8},
        "attr9": {7, 8, 9},
        "np_int": np.int64(42),
        "np_float": np.float32(3.14),
        "np_complex": np.complex128(2.0 + 3.0j),
        "np_bool": np.bool_(True),
    }
    unpacked = msgpack_unpackb(msgpack_packb(payload))
    assert unpacked["attr0"] is True
    assert unpacked["attr1"] == 11
    assert unpacked["attr2"] == 3.5
    assert unpacked["attr3"] == 1.0 + 2.0j
    assert isinstance(unpacked["attr3"], complex)
    assert unpacked["attr4"] == "str_4"
    assert unpacked["attr5"] == b"bytes_5"
    assert unpacked["attr6"] == [1, 2, 3]
    assert unpacked["attr7"] == (4, 5, 6)
    assert isinstance(unpacked["attr7"], tuple)
    assert unpacked["attr8"] == {"key": "val_8", "index": 8}
    assert unpacked["attr9"] == {7, 8, 9}
    assert isinstance(unpacked["attr9"], set)
    assert unpacked["np_int"] == 42
    assert unpacked["np_bool"] is True

    # Test via RemoteArray and trailer vlmeta
    data = np.arange(10, dtype=np.int32)
    arr = blosc2.asarray(data, chunks=(5,), blocks=(5,))
    for k, v in [
        ("attr0", True),
        ("attr1", 11),
        ("attr2", 3.5),
        ("attr3", 1.0 + 2.0j),
        ("attr4", "str_4"),
        ("attr5", b"bytes_5"),
        ("attr6", [1, 2, 3]),
        ("attr7", (4, 5, 6)),
        ("attr8", {"key": "val_8", "index": 8}),
        ("attr9", {7, 8, 9}),
    ]:
        arr.vlmeta[k] = v

    url = "memory://complex-attrs-test.b2nd"
    fsspec.filesystem("memory").pipe_file("complex-attrs-test.b2nd", arr.to_cframe())

    proxy = blosc2.RemoteArray(url)
    assert proxy.vlmeta["attr0"] is True
    assert proxy.vlmeta["attr1"] == 11
    assert proxy.vlmeta["attr2"] == 3.5
    assert proxy.vlmeta["attr3"] == 1.0 + 2.0j
    assert isinstance(proxy.vlmeta["attr3"], complex)
    assert proxy.vlmeta["attr4"] == "str_4"
    assert proxy.vlmeta["attr5"] == b"bytes_5"
    assert proxy.vlmeta["attr6"] == [1, 2, 3]
    assert proxy.vlmeta["attr7"] == (4, 5, 6)
    assert isinstance(proxy.vlmeta["attr7"], tuple)
    assert proxy.vlmeta["attr8"] == {"key": "val_8", "index": 8}
    assert proxy.vlmeta["attr9"] == {7, 8, 9}
    assert isinstance(proxy.vlmeta["attr9"], set)


def test_disk_cache_dir_includes_storage_options(tmp_path):
    url, data = _remote_array("storage-options-identity.b2nd", nchunks=2, chunk_size=1000)
    cache = tmp_path / "cache"

    for endpoint in ("one", "two"):
        array = blosc2.RemoteArray(
            url,
            cache_policy=blosc2.CachePolicy.DISK,
            cache_dir=cache,
            storage_options={"endpoint": endpoint},
        )
        np.testing.assert_array_equal(array[:], data)

    # Different backends are different sources, so they get different carriers.
    assert len(list(cache.glob("*.b2nd"))) == 2


def test_disk_cache_dir_reuses_same_storage_options(tmp_path):
    url, data = _remote_array("storage-options-reuse.b2nd", nchunks=2, chunk_size=1000)
    cache = tmp_path / "cache"

    for _ in range(2):
        array = blosc2.RemoteArray(
            url,
            cache_policy=blosc2.CachePolicy.DISK,
            cache_dir=cache,
            storage_options={"endpoint": "same"},
        )
        np.testing.assert_array_equal(array[:], data)

    assert len(list(cache.glob("*.b2nd"))) == 1
