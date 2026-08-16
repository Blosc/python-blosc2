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
    assert a.schunk.cbytes / a.schunk.nchunks > blosc2.proxy.BLOCK_MIN_CBYTES
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
    # The offsets and one block, against 1.4 MB of chunk
    assert len(traffic) == 2
    assert sum(traffic) < 200_000
    # And the array still reads back whole, over the blocks already cached
    assert np.array_equal(p[...], data)
