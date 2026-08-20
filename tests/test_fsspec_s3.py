#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""fsspec reads against a real S3 endpoint, served locally by moto.

Everything else about the fsspec support is tested over ``memory://``, which is
protocol-generic and needs no dependencies.  Two things it structurally cannot
cover, both of which have already hidden a bug:

- it is not an *async* backend, so ``aget_chunk`` always took its blocking
  fallback there, while against s3fs it raised "got Future attached to a
  different loop" on every chunk;
- it is poorer in metadata than any real store (no mtime), which let a
  size-only cache stamp serve a stale chunk cache.

These run offline -- moto is a local server, no credentials, no network -- so
they are not marked ``network``.
"""

import asyncio

import numpy as np
import pytest

import blosc2

pytest.importorskip("s3fs")
pytest.importorskip("moto")
fsspec = pytest.importorskip("fsspec")

BUCKET = "blosc2-test"


@pytest.fixture(scope="module")
def s3_endpoint():
    """A local S3 server, and fsspec configured to reach it."""
    import fsspec.config
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    endpoint = f"http://{host}:{port}"

    # blosc2.open() has no storage_options passthrough, so the endpoint and the
    # dummy credentials go through fsspec's own per-protocol defaults
    previous = fsspec.config.conf.get("s3")
    fsspec.config.conf["s3"] = {
        "endpoint_url": endpoint,
        "key": "testing",
        "secret": "testing",
        # Not us-east-1: creating a bucket there must carry no location
        # constraint, and s3fs sends one whenever it knows the region
        "client_kwargs": {"region_name": "eu-west-1"},
    }
    fsspec.filesystem("s3", **fsspec.config.conf["s3"]).mkdir(BUCKET)
    yield endpoint

    fsspec.config.conf.pop("s3", None)
    if previous is not None:
        fsspec.config.conf["s3"] = previous
    server.stop()


@pytest.fixture(scope="module")
def stored(s3_endpoint):
    """A 10-chunk array in the bucket, plus the array it was made from."""
    a = blosc2.arange(0, 1000, dtype=np.int32, chunks=(100,))
    urlpath = f"s3://{BUCKET}/ds.b2nd"
    a.save(urlpath)
    return urlpath, a


def test_save_and_open_whole(stored):
    urlpath, a = stored
    assert np.array_equal(blosc2.open(urlpath)[:], a[:])


def test_cache_storage(stored, tmp_path):
    urlpath, a = stored
    b = blosc2.open(urlpath, cache_storage=tmp_path, mmap_mode="r")
    assert np.array_equal(b[:], a[:])


def test_lazy_range_reads(stored):
    urlpath, a = stored
    p = blosc2.open(urlpath, lazy=True)
    assert np.array_equal(p[150:250], a[150:250])
    assert np.array_equal(p[:], a[:])


@pytest.mark.parametrize("max_concurrency", [1, 8])
def test_lazy_concurrency(stored, max_concurrency):
    urlpath, a = stored
    p = blosc2.open(urlpath, lazy=True, max_concurrency=max_concurrency)
    assert np.array_equal(p[:], a[:])


@pytest.mark.parametrize("max_concurrency", [1, 8])
def test_afetch_on_an_async_backend(stored, max_concurrency):
    # The regression this file exists for: s3fs runs its coroutines on a private
    # event loop, so awaiting one from the caller's loop fails outright
    urlpath, a = stored
    p = blosc2.open(urlpath, lazy=True)
    cache = asyncio.run(p.afetch(slice(150, 250), max_concurrency=max_concurrency))
    assert np.array_equal(cache[150:250], a[150:250])


def test_lazy_expression(stored):
    urlpath, a = stored
    p = blosc2.open(urlpath, lazy=True)
    assert np.array_equal((p * 2)[150:250], a[150:250] * 2)


@pytest.fixture(scope="module")
def blocky(s3_endpoint):
    """An array whose chunks are big enough to be worth reading block by block."""
    data = np.random.default_rng(0).random((600, 600))
    a = blosc2.asarray(data, chunks=(300, 600), blocks=(30, 600))
    assert a.schunk.cbytes / a.schunk.nchunks > blosc2.proxy_source.BLOCK_MIN_CBYTES
    urlpath = f"s3://{BUCKET}/blocky.b2nd"
    a.save(urlpath, mode="w")
    return urlpath, data


@pytest.mark.parametrize("max_concurrency", [1, 8])
def test_lazy_block_reads(blocky, max_concurrency):
    urlpath, data = blocky
    p = blosc2.open(urlpath, lazy=True, max_concurrency=max_concurrency)
    traffic = []
    original = p.src.read_range
    p.src.read_range = lambda *args: (out := original(*args), traffic.append(len(out)))[0]

    assert np.array_equal(p[10:12, 30:40], data[10:12, 30:40])
    # Where the chunks are, where the blocks of the one wanted are, and one
    # block, against 1.4 MB of chunk.  The frame's header is not in here: the
    # open read that, before the hook went on
    assert len(traffic) == 3
    assert sum(traffic) < 200_000
    # And the array still reads back whole, over the blocks already cached
    assert np.array_equal(p[...], data)


def test_a_kept_index_is_refused_when_the_object_was_replaced(s3_endpoint, tmp_path):
    """The cache keeps *positions* now, which a replaced object invalidates.

    A stale chunk cache serves old data; a stale index is worse, since blocks
    fetched at the old frame's offsets and spliced into a chunk decode to
    nonsense.  s3fs is where this can be checked honestly: its `ukey` is a real
    object identity, where `memory://` has almost no metadata to build one from.
    """
    data = np.random.default_rng(0).random((600, 600))
    url = f"s3://{BUCKET}/replaced.b2nd"
    blosc2.asarray(data, chunks=(300, 600), blocks=(30, 600)).save(url, mode="w")
    cache = str(tmp_path / "replaced-cache.b2nd")

    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")
    assert np.array_equal(p[10:12, 30:40], data[10:12, 30:40])
    del p
    holder = blosc2.open(cache)
    assert "proxy-index" in holder.schunk.vlmeta  # where the chunks and blocks are
    del holder

    # Same shape, same partitioning, other bytes: every offset in there now
    # points somewhere else in a frame of the same size
    other = np.random.default_rng(1).random((600, 600))
    blosc2.asarray(other, chunks=(300, 600), blocks=(30, 600)).save(url, mode="w")
    with pytest.raises(ValueError, match="different remote bytes"):
        blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a")

    # ... and starting afresh reads the new bytes, index and all
    p = blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="w")
    assert np.array_equal(p[10:12, 30:40], other[10:12, 30:40])
    assert np.array_equal(p[...], other)


def test_a_kept_index_spares_a_later_run_the_reads(s3_endpoint, tmp_path):
    # The same slice geometry as test_lazy_block_reads, one run later: the
    # offsets and the layout of the half-held chunk both come out of the cache,
    # so only the blocks that are missing travel
    data = np.random.default_rng(0).random((600, 600))
    url = f"s3://{BUCKET}/kept.b2nd"
    blosc2.asarray(data, chunks=(300, 600), blocks=(30, 600)).save(url, mode="w")
    cache = str(tmp_path / "kept-cache.b2nd")
    blosc2.Proxy(blosc2.FsspecNDSource(url), urlpath=cache, mode="a").fetch((slice(10, 12), slice(30, 40)))

    src = blosc2.FsspecNDSource(url)
    traffic = []
    original = src.read_range
    src.read_range = lambda *args: (out := original(*args), traffic.append(len(out)))[0]
    p = blosc2.Proxy(src, urlpath=cache, mode="a")
    assert np.array_equal(p[60:62, 30:40], data[60:62, 30:40])
    assert len(traffic) == 1  # one block, and nothing to say where it was
    assert np.array_equal(p[...], data)
