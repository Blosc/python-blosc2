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
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import blosc2

# The stand-in subscriber binds a real socket, and Pyodide has no listen(2):
# node asks for the `ws` module that is not there, and takes the runtime down
# with it rather than raising
pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no listening sockets on wasm32")


class _Subscriber:
    """A Caterva2-shaped server over one .b2nd file."""

    def __init__(
        self,
        path,
        key=None,
        ranges=True,
        cookie=None,
        multipart=True,
        merge_ranges=True,
        fetch_failures=0,
    ):
        self.fetch_failures = fetch_failures  # answer this many fetches 503 first
        self.path = str(path)
        self.key = key  # a leaf inside a .b2z container, rather than a file of its own
        self.ranges = ranges  # False: stream the body and ignore Range, as a
        self.cookie = cookie  # computed dataset does
        self.multipart = multipart  # False: answer only the first range asked for
        self.merge_ranges = merge_ranges  # as Starlette does with ranges that touch
        self.log = []  # (endpoint, status, bytes served)
        self.reload()

    def reload(self):
        """Pick up the file as it is now, as a subscriber would on the next request.

        A leaf is served out of its window in the container, which is what makes
        it look to a client exactly like a dataset of its own: byte 0 of what it
        asks for is the frame's first byte.
        """
        raw = pathlib.Path(self.path).read_bytes()
        self.mtime = pathlib.Path(self.path).stat().st_mtime
        if self.key is None:
            self.frame, self.array = raw, blosc2.open(self.path)
            return
        store = blosc2.open(self.path)
        offset, nbytes = store.member_window(self.key)
        self.frame, self.array = raw[offset : offset + nbytes], store[self.key]

    @property
    def meta(self):
        schunk = self.array.schunk
        return {
            "shape": list(self.array.shape),
            "chunks": list(self.array.chunks),
            "blocks": list(self.array.blocks),
            "dtype": str(self.array.dtype),
            "mtime": self.mtime,
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
        if sub.fetch_failures:
            # A subscriber too busy to answer says nothing about how it serves
            sub.fetch_failures -= 1
            self._send(503, b"busy", endpoint="fetch")
            return
        wanted = self.headers.get("Range")
        if not wanted or not sub.ranges:
            # What a StreamingResponse does with a Range header: nothing at all
            self._send(200, sub.frame, endpoint="fetch")
            return
        spans = []
        for span in wanted.removeprefix("bytes=").split(","):
            start, end = (int(n) for n in span.split("-"))
            spans.append((start, min(end, len(sub.frame) - 1)))
        # Starlette sorts the spans and merges the ones that touch, so a client
        # cannot count on getting a part per span, nor on the order it asked in
        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1] + 1 and sub.merge_ranges:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if len(merged) == 1 or not sub.multipart:
            start, end = merged[0]  # ... and answers a plain 206 when one is left
            self._send(
                206,
                sub.frame[start : end + 1],
                [("Content-Range", f"bytes {start}-{end}/{len(sub.frame)}"), ("Accept-Ranges", "bytes")],
                endpoint="fetch",
            )
            return
        boundary = "c2boundary"
        body = b""
        for start, end in merged:
            body += (
                f"--{boundary}\r\nContent-Type: application/octet-stream\r\n"
                f"Content-Range: bytes {start}-{end}/{len(sub.frame)}\r\n\r\n"
            ).encode()
            body += sub.frame[start : end + 1] + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        self._send(
            206,
            body,
            [
                ("Content-Type", f"multipart/byteranges; boundary={boundary}"),
                ("Accept-Ranges", "bytes"),
            ],
            endpoint="fetch",
        )


def _serve(tmp_path, data, chunks, blocks, name="ds.b2nd", key=None, **kwargs):
    """A C2Array over *data*, served by a subscriber stand-in on localhost.

    With *key*, the array is a leaf of a TreeStore container instead of a file
    of its own, and the subscriber serves it from its window -- which is what
    Caterva2 does, and what the client is meant not to notice.
    """
    urlpath = str(tmp_path / name)
    if key is None:
        blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
    else:
        with blosc2.TreeStore(urlpath, mode="w") as tstore:
            tstore[key] = blosc2.asarray(data, chunks=chunks, blocks=blocks)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.subscriber = _Subscriber(urlpath, key=key, **kwargs)
    # A short poll interval, because `shutdown()` waits for one to elapse before
    # the serve loop notices: at the default 0.5 s that is half a second of doing
    # nothing per test, and this file has enough of them for that to be most of
    # what it costs
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    urlbase = f"http://127.0.0.1:{server.server_address[1]}/"
    path = f"@public/{name}{key or ''}"
    array = blosc2.C2Array(path, urlbase=urlbase, auth_token=kwargs.get("cookie"))
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
    monkeypatch.setattr(blosc2.proxy_source, "BLOCK_MIN_CBYTES", 0)


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
    # The frame index (header, then offsets), one read for the chunk's block
    # offsets, and one for the block the slice lands in
    assert [kind for kind, _, _ in sub.log] == ["fetch"] * 4
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


def test_a_cache_that_holds_the_slice_costs_no_request(tmp_path, subscriber, any_chunk_wants_blocks):
    # Re-running a script over a cache that already covers the slice: `api/info`
    # is all it takes.  Nothing opens the frame, because opening it is what
    # `block_source` puts off until a fetch actually wants a chunk -- and this
    # fetch wants none.
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "held.b2nd")
    item = (slice(0, 5), slice(0, 10))
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch(item)

    # A later run: its own array over the same subscriber, its own source
    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    sub.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    p.fetch(item)
    assert np.array_equal(p[item], data[item])
    assert not sub.log

    # ... and a slice the cache does not hold opens the frame then: the header,
    # the layout of the chunk it lands in, and the blocks.  Not where the chunks
    # are -- the earlier run left that in the cache
    assert np.array_equal(p[100:105, 0:10], data[100:105, 0:10])
    assert [kind for kind, _, _ in sub.log] == ["fetch"] * 3


def test_a_kept_index_halves_a_warm_fetch(tmp_path, subscriber, any_chunk_wants_blocks):
    # A later run wanting different blocks of chunks a previous one half filled:
    # where the chunks are and where those blocks are both came out of the cache,
    # so what travels is the header and the blocks, and nothing between
    data = _incompressible((400, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "kept.b2nd")
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch((slice(None), slice(0, 10)))

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    sub.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    assert np.array_equal(p[:, 100:110], data[:, 100:110])
    assert [kind for kind, _, _ in sub.log] == ["fetch", "fetch"]  # the header, the blocks
    assert np.array_equal(p[...], data)  # ... and the rest still reads right


def test_a_kept_index_does_not_open_the_frame_to_be_taken_up(tmp_path, subscriber, any_chunk_wants_blocks):
    # Handing the index to the source would build the source to receive it, and
    # building it reads the header -- a request, at the very moment of a run that
    # may go on to fetch nothing.  It waits with the array until there is a source
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "unopened.b2nd")
    item = (slice(0, 5), slice(0, 10))
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch(item)

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    sub.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    assert again._pending_index is not None  # taken out of the cache, not yet used
    assert not sub.log
    p.fetch(item)
    assert not sub.log


def test_a_whole_chunk_cache_is_adopted(tmp_path, subscriber, any_chunk_wants_blocks):
    # A cache left by a run that fetched whole chunks (which is every run before
    # this existed) holds complete chunks, so nothing in it is fetched again
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "chunkwise.b2nd")

    array._block_source = None  # as if the subscriber served no ranges
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    # The same dataset, opened afresh: this one takes the blocks path
    array = blosc2.C2Array(array.path, urlbase=array.urlbase)
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


def test_a_whole_wave_travels_in_one_request(subscriber, any_chunk_wants_blocks):
    # A column through every chunk: each one wants a handful of blocks that lie
    # apart in the file, which without batching is a request each
    data = _incompressible((400, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    assert array.block_source() is not None  # the header, read once per array
    sub.log.clear()

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    # One request for where the four chunks are, one for the layouts of all of
    # them, one for all their blocks -- three waves, not three per chunk
    assert [kind for kind, _, _ in sub.log] == ["fetch", "fetch", "fetch"]
    assert {status for _, status, _ in sub.log} == {206}
    assert _bytes(sub, "fetch") < sub.array.schunk.cbytes / 4

    # Which is the whole of the difference: one request per range otherwise
    other, sub2 = subscriber(data, chunks=(100, 200), blocks=(10, 20), name="unbatched.b2nd")
    other.block_source().max_ranges = 1
    q = blosc2.Proxy(other, mode="w")
    sub2.log.clear()
    assert np.array_equal(q[:, 0:10], data[:, 0:10])
    assert len(sub2.log) > 4 * len(sub.log)


def test_merged_and_reordered_parts_are_read_correctly(subscriber, any_chunk_wants_blocks):
    # The server sorts the spans and merges the ones that touch, so the answer
    # carries fewer parts than were asked for and in an order of its own
    data = _incompressible((400, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    assert np.array_equal(p[...], data)
    assert array.max_ranges > 1  # ... and it never had to stop batching


def test_a_server_that_answers_one_range_stops_being_batched(subscriber, any_chunk_wants_blocks):
    # A subscriber that takes the first span of a multi-range request and ignores
    # the rest: the answer does not cover what was asked for, which is noticed
    # and never repeated
    data = _incompressible((400, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), multipart=False)
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    assert array.max_ranges == 1
    served = len(sub.log)
    assert np.array_equal(p[:, 100:110], data[:, 100:110])
    # One request per range from here on, and no second attempt at batching
    assert len(sub.log) > served + 2
    assert np.array_equal(p[...], data)


# --- a cache is checked against the bytes it was filled from ----------------


def _replace(sub, data, chunks, blocks):
    """Rewrite the served dataset, as an upload of new data would."""
    blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=sub.path, mode="w")
    stat = pathlib.Path(sub.path).stat()
    os.utime(sub.path, (stat.st_atime, stat.st_mtime + 10))  # a tick the clock cannot swallow
    sub.reload()


def test_a_cache_is_stamped_with_the_remote_mtime(subscriber):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert array.stamp == f"{sub.mtime}:{sub.array.schunk.cbytes}"
    assert p.schunk.vlmeta["proxy-stamp"] == array.stamp


def test_a_cache_from_other_bytes_is_refused(tmp_path, subscriber, any_chunk_wants_blocks):
    # Same shape, same partitioning, different data: geometry cannot tell, and
    # every cached chunk (and the offsets it was fetched by) is stale
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "stamped.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(sub, other, chunks=(100, 200), blocks=(10, 20))
    replaced = blosc2.C2Array(array.path, urlbase=array.urlbase)
    assert replaced.stamp != array.stamp

    with pytest.raises(ValueError, match="different remote bytes"):
        blosc2.Proxy(replaced, urlpath=cache, mode="a")

    # ... and mode="w" is the way through, with the new data behind it
    p = blosc2.Proxy(replaced, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], other[0:5, 0:10])
    assert np.array_equal(p[...], other)


def test_a_cache_of_bytes_that_were_replaced_is_emptied(tmp_path, subscriber, any_chunk_wants_blocks):
    # `blosc2.open` rebuilds the proxy over the cache as it stands, with no
    # `mode="a"` to refuse it by: the chunks in there were fetched from a frame
    # that is gone, so what the cache says it holds is dropped rather than served
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "emptied.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(sub, other, chunks=(100, 200), blocks=(10, 20))

    reopened = blosc2.open(cache, mode="a")
    assert np.array_equal(reopened[0:5, 0:10], other[0:5, 0:10])
    assert np.array_equal(reopened[...], other)


def test_a_cache_emptied_of_replaced_bytes_stays_emptied(tmp_path, subscriber, any_chunk_wants_blocks):
    # The stamp is written on the way in, so the run after this one finds a cache
    # whose stamp fits and believes what it says it holds: emptying it has to
    # reach the file, not just the proxy that noticed
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "stillemptied.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[...], data)  # every chunk of it, fetched and recorded
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(sub, other, chunks=(100, 200), blocks=(10, 20))
    noticed = blosc2.open(cache, mode="a")  # opened over the new bytes, dropped unread
    del noticed

    again = blosc2.open(cache, mode="a")
    assert again.schunk.vlmeta["proxy-stamp"] == blosc2.C2Array(array.path, urlbase=array.urlbase).stamp
    assert np.array_equal(again[...], other)


def test_a_read_only_cache_of_replaced_bytes_reads_past_it(tmp_path, subscriber, any_chunk_wants_blocks):
    # Nothing may be written to a cache opened read-only, so it cannot be emptied
    # either -- but nothing of it is believed either, and the read falls through
    # to the source rather than coming back with the bytes that are gone
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "readonlyreplaced.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[...], data)
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(sub, other, chunks=(100, 200), blocks=(10, 20))

    reopened = blosc2.open(cache, mode="r")
    assert np.array_equal(reopened[:], other)  # read past the cache, off the source
    # ... and the cache is left exactly as it was, stamp and chunks alike
    assert reopened.schunk.vlmeta["proxy-stamp"] != blosc2.C2Array(array.path, urlbase=array.urlbase).stamp


def test_a_cache_from_the_same_bytes_is_adopted(tmp_path, subscriber, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "unchanged.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    assert again.stamp == array.stamp
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    served = len(sub.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(sub.log) == served  # what it holds was not fetched again


def test_no_stamp_when_the_subscriber_reports_no_mtime(tmp_path, subscriber):
    # Then the cache is checked on geometry alone, as every unstamped source is
    data = _incompressible((200, 200))
    array, _sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    del array.meta["mtime"]
    assert array.stamp is None

    cache = str(tmp_path / "unstamped.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert "proxy-stamp" not in p.schunk.vlmeta.getall()
    del p
    assert blosc2.Proxy(array, urlpath=cache, mode="a") is not None


def test_a_read_only_cache_is_not_stamped(tmp_path, subscriber):
    # `blosc2.open(path, mode="r")` rebuilds the proxy over a cache that may not
    # be written to; recording the stamp there raised instead of opening it
    data = _incompressible((200, 200))
    array, _sub = subscriber(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "readonly.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    reopened = blosc2.open(cache, mode="r")
    assert np.array_equal(reopened[0:5, 0:10], data[0:5, 0:10])


def test_blocks_of_a_container_leaf(subscriber, any_chunk_wants_blocks):
    """A leaf of a .b2z is a whole frame inside the container, and a subscriber
    serves it from that window -- so the client reads its blocks knowing nothing
    about containers, which is the whole of what it takes."""
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), name="tree.b2z", key="/g/leaf")
    p = blosc2.Proxy(array, mode="w")
    assert array.block_source() is not None
    sub.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert not _bytes(sub, "chunk")
    assert _bytes(sub, "fetch") < sub.array.schunk.cbytes / 8
    assert np.array_equal(p[...], data)


def test_a_container_leaf_is_stamped_like_any_other(subscriber):
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), name="tree.b2z", key="/g/leaf")
    # A leaf has no mtime of its own: the container's is what says it changed
    assert array.stamp == f"{sub.mtime}:{sub.array.schunk.cbytes}"


def test_a_busy_subscriber_is_asked_again(subscriber, any_chunk_wants_blocks):
    # A 503 to the probe says nothing about whether the dataset is served from a
    # file, and cost no download to find out -- unlike the streamed 200 the
    # permanent fallback exists for, so this one is asked again on the next fetch
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), fetch_failures=1)
    p = blosc2.Proxy(array, mode="w")

    assert array.block_source() is None  # the probe was refused ...
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])  # ... and whole chunks served it
    assert array.block_source() is not None  # ... but the next ask gets an answer
    assert np.array_equal(p[...], data)


def test_a_streamed_dataset_is_not_asked_again(subscriber, any_chunk_wants_blocks):
    # The other half of the same rule: a 200 is the dataset itself, and asking
    # again would pay for the whole of it to be told the same thing
    data = _incompressible((200, 200))
    array, sub = subscriber(data, chunks=(100, 200), blocks=(10, 20), ranges=False)
    p = blosc2.Proxy(array, mode="w")

    assert array.block_source() is None
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is None
    assert not any(status == 206 for _, status, _ in sub.log)


def test_a_part_that_ends_early_is_refused(subscriber, any_chunk_wants_blocks):
    # A part that starts inside a span and stops short of its end would slice to
    # a short payload, which is spliced against a `bstarts` promising the whole
    # of it -- so it is refused, and the fetch falls back to a range per request
    parts = [(100, b"12345")]
    assert blosc2.c2array._span_of(parts, 100, 5, "url") == b"12345"
    with pytest.raises(blosc2.c2array._PartsMissing):
        blosc2.c2array._span_of(parts, 100, 6, "url")
    with pytest.raises(blosc2.c2array._PartsMissing):
        blosc2.c2array._span_of(parts, 103, 4, "url")


def test_the_shared_client_keeps_no_cookies():
    # One client serves every C2Array, so a `Set-Cookie` kept from any of them
    # would start authorizing requests that asked for none
    import httpx

    client = blosc2.c2array._sync_client()
    request = httpx.Request("GET", "http://127.0.0.1/api/fetch/x")
    response = httpx.Response(200, headers={"set-cookie": "session=secret; Path=/"}, request=request)
    client.cookies.extract_cookies(response)
    assert not dict(client.cookies)
