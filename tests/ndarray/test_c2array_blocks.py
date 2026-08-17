#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Block-granular reads of a C2Array, against a stand-in for a subscriber.

The server here answers the two endpoints the block path uses -- `api/info` for
the geometry and `api/fetch` for the bytes -- the way Caterva2 does: a stored
dataset comes back through a file response that honours `Range`, and one the
subscriber would compute comes back as a stream that ignores it.  Which is the
distinction the whole arrangement rests on, and the one thing a test against a
live subscriber could not switch off at will.
"""

import contextlib
import json
import math
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import blosc2


class _Subscriber:
    """A Caterva2-shaped server over one .b2nd file."""

    def __init__(self, path, ranges=True, cookie=None):
        self.path = str(path)
        self.frame = pathlib.Path(self.path).read_bytes()
        self.array = blosc2.open(self.path)
        self.ranges = ranges  # False: stream the body and ignore Range, as a
        self.cookie = cookie  # computed dataset does
        self.log = []  # (endpoint, status, bytes served)

    @property
    def meta(self):
        schunk = self.array.schunk
        return {
            "shape": list(self.array.shape),
            "chunks": list(self.array.chunks),
            "blocks": list(self.array.blocks),
            "dtype": str(self.array.dtype),
            "mtime": None,
            "schunk": {
                "cparams": {"typesize": self.array.dtype.itemsize},
                "nbytes": schunk.nbytes,
                "cbytes": schunk.cbytes,
                "cratio": schunk.cratio,
                "blocksize": schunk.blocksize,
                "vlmeta": {},
            },
        }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass  # no stderr noise per request

    def handle(self):
        # A client that hangs up on a body it refused, which is the point of the
        # probe, leaves the write half of this raising
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            super().handle()

    def _send(self, status, body, headers=(), endpoint=""):
        self.server.subscriber.log.append((endpoint, status, len(body)))
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        sub = self.server.subscriber
        if sub.cookie and self.headers.get("Cookie") != sub.cookie:
            self._send(401, b"unauthorized", endpoint="auth")
            return
        endpoint = self.path.split("/")[2]
        if endpoint == "info":
            self._send(200, json.dumps(sub.meta).encode(), endpoint="info")
        elif endpoint == "chunk":
            nchunk = int(self.path.split("nchunk=")[1])
            self._send(200, sub.array.schunk.get_chunk(nchunk), endpoint="chunk")
        elif endpoint == "fetch":
            self._fetch(sub)
        else:
            self._send(404, b"", endpoint=endpoint)

    def _fetch(self, sub):
        wanted = self.headers.get("Range")
        if not wanted or not sub.ranges:
            # What a StreamingResponse does with a Range header: nothing at all
            self._send(200, sub.frame, endpoint="fetch")
            return
        start, end = (int(n) for n in wanted.removeprefix("bytes=").split("-"))
        end = min(end, len(sub.frame) - 1)
        self._send(
            206,
            sub.frame[start : end + 1],
            [("Content-Range", f"bytes {start}-{end}/{len(sub.frame)}"), ("Accept-Ranges", "bytes")],
            endpoint="fetch",
        )


def _serve(tmp_path, data, chunks, blocks, ranges=True, cookie=None, name="ds.b2nd"):
    """A C2Array over *data*, served by a subscriber stand-in on localhost."""
    urlpath = str(tmp_path / name)
    blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.subscriber = _Subscriber(urlpath, ranges=ranges, cookie=cookie)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    urlbase = f"http://127.0.0.1:{server.server_address[1]}/"
    array = blosc2.C2Array(f"@public/{name}", urlbase=urlbase, auth_token=cookie)
    return array, server.subscriber, server


@pytest.fixture
def subscriber(tmp_path):
    """Serve one array; the test parametrizes with `_serve`'s arguments."""
    servers = []

    def build(*args, **kwargs):
        array, sub, server = _serve(tmp_path, *args, **kwargs)
        servers.append(server)
        return array, sub

    yield build
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def any_chunk_wants_blocks(monkeypatch):
    """Take the size threshold out of the way, so small test arrays use blocks."""
    monkeypatch.setattr(blosc2.proxy, "BLOCK_MIN_CBYTES", 0)


def _incompressible(shape, seed=0):
    return np.random.default_rng(seed).random(shape)


def _bytes(sub, endpoint):
    return sum(n for kind, _, n in sub.log if kind == endpoint)


def test_blocks_are_read_over_ranges(subscriber, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    sub.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    # The frame index, then one read for the block offsets and one for the block
    assert [kind for kind, _, _ in sub.log] == ["fetch"] * 6
    assert {status for _, status, _ in sub.log} == {206}
    assert not _bytes(sub, "chunk")
    # A block of a chunk, not the chunk: an eighth of it here, and never the frame
    assert _bytes(sub, "fetch") < sub.array.schunk.cbytes / 8
    # ... and the rest of the array still arrives correctly afterwards
    assert np.array_equal(p[...], data)


@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "item"),
    [
        ((1000,), (500,), (50,), slice(120, 140)),
        ((200, 200), (100, 200), (10, 20), (slice(5, 7), slice(30, 90))),
        ((20, 60, 60), (10, 30, 60), (5, 10, 20), (3, slice(10, 20), slice(10, 20))),
        ((200, 200), (100, 200), (10, 20), (5, 5)),
    ],
    ids=["1d", "2d", "3d", "point"],
)
def test_block_reads_are_correct(subscriber, any_chunk_wants_blocks, shape, chunks, blocks, item):
    data = _incompressible(shape)
    array, _ = subscriber(data, chunks=chunks, blocks=blocks)
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[item], data[item])
    assert np.array_equal(p[...], data)


def test_blocks_carry_the_auth_cookie(subscriber, any_chunk_wants_blocks):
    # fsspec's HTTP filesystem cannot carry this, which is why the block reads of
    # a C2Array are its own rather than an fsspec URL pointed at the subscriber
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), cookie="token=sikrit")
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is not None
    assert not any(status == 401 for _, status, _ in sub.log)


def test_a_streamed_dataset_falls_back_to_chunks(subscriber, any_chunk_wants_blocks):
    # A lazy expression, an HDF5 leaf or a .b2z member is built rather than
    # stored, and the response that carries it ignores Range
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), ranges=False)
    p = blosc2.Proxy(array, mode="w")
    sub.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is None
    assert [kind for kind, _, _ in sub.log] == ["fetch", "chunk"]
    # The probe must not read the body it refused: the whole dataset is what it
    # would have downloaded to find out that ranges are not served
    assert _bytes(sub, "fetch") == len(sub.frame)  # served, but never read

    # And it is never probed again, whatever else is asked for
    assert np.array_equal(p[...], data)
    assert sum(1 for kind, _, _ in sub.log if kind == "fetch") == 1


def test_a_computed_dataset_is_ruled_out_without_a_request(subscriber, any_chunk_wants_blocks):
    # api/info tells a stored dataset from one the subscriber computes: the
    # latter reports `expression` and `operands` where this reports a geometry
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    del array.meta["chunks"]
    sub.log.clear()

    assert array.block_source() is None
    assert not sub.log


def test_small_chunks_are_fetched_whole(subscriber):
    # Below the threshold a chunk is one cheap request, so blocks would only add
    # a round trip: nothing goes looking for the frame index, let alone a block
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    sub.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is None
    assert [kind for kind, _, _ in sub.log] == ["chunk"]


def test_blocks_accumulate_in_a_chunk(subscriber, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    served = len(sub.log)
    # A different block of the same chunk: what is already cached stays cached
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(sub.log) > served
    served = len(sub.log)
    # Both are now in the same cached chunk, and both are still right
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(sub.log) == served


def test_blocks_survive_a_reopened_cache(tmp_path, subscriber, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "c2-cache.b2nd")

    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    # A partly filled chunk survives, so the blocks in it do not travel again
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    served = len(sub.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(sub.log) == served
    # ... and the ones missing from it still do
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(sub.log) > served
    assert np.array_equal(p[...], data)


def test_a_whole_chunk_cache_is_adopted(tmp_path, subscriber, any_chunk_wants_blocks):
    # A cache left by a run that fetched whole chunks (which is every run before
    # this existed) holds complete chunks, so nothing in it is fetched again
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "chunkwise.b2nd")

    array._block_source = None  # as if the subscriber served no ranges
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p, array

    array, _ = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    served = len(sub.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(sub.log) == served
    assert np.array_equal(p[...], data)


def test_blocks_per_chunk_costs_no_request(subscriber):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    sub.log.clear()

    assert array.blocks_per_chunk == math.prod((100 // 10, 200 // 20))
    assert not sub.log


def test_read_range_says_so_when_there_are_no_ranges(subscriber, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, _ = subscriber(data, chunks=(100, 200), blocks=(10, 20), ranges=False)

    with pytest.raises(ValueError, match="not served in byte ranges"):
        array.read_range(0, 32)
