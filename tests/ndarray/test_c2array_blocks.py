#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Block-granular reads of a C2Array, against a stand-in for a server.

The server here answers the two endpoints the block path uses -- `api/info` for
the geometry and `api/fetch` for the bytes -- the way Caterva2 does: a stored
dataset comes back through a file response that honours `Range`, and one the
server would compute comes back as a stream that ignores it.  Which is the
distinction the whole arrangement rests on, and the one thing a test against a
live server could not switch off at will.
"""

import contextlib
import json
import math
import os
import pathlib
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

import blosc2

# The stand-in server binds a real socket, and Pyodide has no listen(2):
# node asks for the `ws` module that is not there, and takes the runtime down
# with it rather than raising
pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no listening sockets on wasm32")


class _Cat2Server:
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
        bad_parts=0,
        geometry=True,
        accept_ranges=None,
        post_fetch=True,
    ):
        # False: a server that answers 405 to `POST api/fetch`, as one from
        # before the route existed does
        self.post_fetch = post_fetch
        # What `api/info` reports for byte ranges: "bytes", "none", or None for a
        # server old enough to report nothing, which is what the client must survive
        self.accept_ranges = accept_ranges
        self.fetch_failures = fetch_failures  # answer this many fetches 503 first
        self.path = str(path)
        self.key = key  # a leaf inside a .b2z container, rather than a file of its own
        self.ranges = ranges  # False: stream the body and ignore Range, as a
        self.cookie = cookie  # computed dataset does
        self.multipart = multipart  # False: answer only the first range asked for
        self.bad_parts = bad_parts  # ... and this many multi-range answers do too
        self.merge_ranges = merge_ranges  # as Starlette does with ranges that touch
        self.geometry = geometry  # False: report a dataset the server computes,
        # which has no chunks or blocks of its own to report
        self.log = []  # (endpoint, status, bytes served)
        self.reload()

    def reload(self):
        """Pick up the file as it is now, as a server would on the next request.

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
        if not self.geometry:
            # What a lazy expression looks like: an expression and its operands
            # where a stored dataset has a partitioning
            return {
                "shape": list(self.array.shape),
                "dtype": str(self.array.dtype),
                "expression": "a + 1",
                "operands": {"a": "@public/other.b2nd"},
                "schunk": {"cparams": {"typesize": self.array.dtype.itemsize}},
            }
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
            **({} if self.accept_ranges is None else {"accept_ranges": self.accept_ranges}),
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
        self.server.cat2.log.append((endpoint, status, len(body)))
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        srv = self.server.cat2
        if srv.cookie and self.headers.get("Cookie") != srv.cookie:
            self._send(401, b"unauthorized", endpoint="auth")
            return
        endpoint = self.path.split("/")[2]
        if endpoint == "info":
            self._send(200, json.dumps(srv.meta).encode(), endpoint="info")
        elif endpoint == "chunk":
            nchunk = int(self.path.split("nchunk=")[1])
            self._send(200, srv.array.schunk.get_chunk(nchunk), endpoint="chunk")
        elif endpoint == "fetch":
            self._fetch(srv)
        else:
            self._send(404, b"", endpoint=endpoint)

    def do_POST(self):
        srv = self.server.cat2
        if srv.cookie and self.headers.get("Cookie") != srv.cookie:
            self._send(401, b"unauthorized", endpoint="auth")
            return
        raw = self.rfile.read(int(self.headers["Content-Length"]))  # drained either way
        if not srv.post_fetch:  # a server old enough not to know the route
            # Answering without reading the body would leave it on the connection
            # for the next request to read as a request line, which is a thing
            # this stand-in does and a real server does not
            self._send(405, b"method not allowed", endpoint="fetch")
            return
        body = json.loads(raw)
        self._gather(srv, body["indices"])

    @staticmethod
    def _entry(e):
        """One dimension of an `indices` key, as the thing numpy indexes with."""
        if e is None:
            return slice(None)
        if isinstance(e, list):
            return np.array(e)
        if isinstance(e, str):  # a bounded slice travels as "start:stop"
            first, _, last = e.partition(":")
            return slice(int(first) if first else None, int(last) if last else None)
        return e

    def _gather(self, srv, raw):
        # What Caterva2 does with a fancy key: gather the points and send those,
        # since there is no file to seek into for coordinates
        key = tuple(self._entry(e) for e in json.loads(raw))
        data = blosc2.asarray(np.ascontiguousarray(srv.array[key])).to_cframe()
        self._send(200, data, endpoint="fetch")

    @staticmethod
    def _spelled(raw):
        """A `slice_` string, read back the way `slice_to_string` wrote it."""
        key = []
        for part in (p.strip() for p in raw.split(",")):
            first, colon, last = part.partition(":")
            if colon:
                key.append(slice(int(first) if first else None, int(last) if last else None))
            else:
                key.append(int(part))
        return tuple(key)

    def _fetch(self, srv):
        if "indices=" in self.path:
            self._gather(srv, urllib.parse.unquote(self.path.split("indices=")[1].split("&")[0]))
            return
        if "slice_=" in self.path and not self.headers.get("Range"):
            # A box, which the server reads out of the array as any slice is read
            raw = urllib.parse.unquote_plus(self.path.split("slice_=")[1].split("&")[0])
            if raw:
                data = srv.array[self._spelled(raw)]
                self._send(200, blosc2.asarray(np.ascontiguousarray(data)).to_cframe(), endpoint="fetch")
                return
        if srv.fetch_failures:
            # A server too busy to answer says nothing about how it serves
            srv.fetch_failures -= 1
            self._send(503, b"busy", endpoint="fetch")
            return
        wanted = self.headers.get("Range")
        if not wanted or not srv.ranges:
            # What a StreamingResponse does with a Range header: nothing at all
            self._send(200, srv.frame, endpoint="fetch")
            return
        spans = []
        for span in wanted.removeprefix("bytes=").split(","):
            start, end = (int(n) for n in span.split("-"))
            spans.append((start, min(end, len(srv.frame) - 1)))
        # Starlette sorts the spans and merges the ones that touch, so a client
        # cannot count on getting a part per span, nor on the order it asked in
        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1] + 1 and srv.merge_ranges:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        partial = not srv.multipart
        if len(merged) > 1 and srv.bad_parts:
            srv.bad_parts -= 1  # an answer that carries only the first part, once
            partial = True
        if len(merged) == 1 or partial:
            start, end = merged[0]  # ... and answers a plain 206 when one is left
            self._send(
                206,
                srv.frame[start : end + 1],
                [("Content-Range", f"bytes {start}-{end}/{len(srv.frame)}"), ("Accept-Ranges", "bytes")],
                endpoint="fetch",
            )
            return
        boundary = "c2boundary"
        body = b""
        for start, end in merged:
            body += (
                f"--{boundary}\r\nContent-Type: application/octet-stream\r\n"
                f"Content-Range: bytes {start}-{end}/{len(srv.frame)}\r\n\r\n"
            ).encode()
            body += srv.frame[start : end + 1] + b"\r\n"
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
    """A C2Array over *data*, served by a server stand-in on localhost.

    With *key*, the array is a leaf of a TreeStore container instead of a file
    of its own, and the server serves it from its window -- which is what
    Caterva2 does, and what the client is meant not to notice.
    """
    urlpath = str(tmp_path / name)
    if key is None:
        blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=urlpath, mode="w")
    else:
        with blosc2.TreeStore(urlpath, mode="w") as tstore:
            tstore[key] = blosc2.asarray(data, chunks=chunks, blocks=blocks)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.cat2 = _Cat2Server(urlpath, key=key, **kwargs)
    # A short poll interval, because `shutdown()` waits for one to elapse before
    # the serve loop notices: at the default 0.5 s that is half a second of doing
    # nothing per test, and this file has enough of them for that to be most of
    # what it costs
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    urlbase = f"http://127.0.0.1:{httpd.server_address[1]}/"
    path = f"@public/{name}{key or ''}"
    array = blosc2.C2Array(path, urlbase=urlbase, auth_token=kwargs.get("cookie"))
    return array, httpd.cat2, httpd


@pytest.fixture
def server(tmp_path):
    """Serve one array; the test parametrizes with `_serve`'s arguments."""
    servers = []

    def build(*args, **kwargs):
        array, srv, httpd = _serve(tmp_path, *args, **kwargs)
        servers.append(httpd)
        return array, srv

    yield build
    for httpd in servers:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def any_chunk_wants_blocks(monkeypatch):
    """Take the size threshold out of the way, so small test arrays use blocks."""
    monkeypatch.setattr(blosc2.proxy_source, "BLOCK_MIN_CBYTES", 0)


def _incompressible(shape, seed=0):
    return np.random.default_rng(seed).random(shape)


def _bytes(srv, endpoint):
    return sum(n for kind, _, n in srv.log if kind == endpoint)


def test_open_urlpath_lazy_memory_cache(server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    urlpath = blosc2.URLPath(array.path, urlbase=array.urlbase)

    srv.log.clear()
    proxy = blosc2.open(urlpath, lazy=True, max_concurrency=3)

    assert [endpoint for endpoint, _, _ in srv.log] == ["info"]
    assert isinstance(proxy, blosc2.Proxy)
    assert isinstance(proxy.src, blosc2.C2Array)
    assert proxy.src.max_concurrency == 3
    assert proxy.urlpath is None

    result = proxy[0:5, 0:10]
    served = len(srv.log)
    assert np.array_equal(result, data[0:5, 0:10])
    assert np.array_equal(proxy[0:5, 0:10], result)
    assert len(srv.log) == served


def test_open_urlpath_lazy_persistent_cache(tmp_path, server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    urlpath = blosc2.URLPath(array.path, urlbase=array.urlbase)
    cache_dir = tmp_path / "cache"

    srv.log.clear()
    proxy = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    assert [endpoint for endpoint, _, _ in srv.log] == ["info"]
    assert np.array_equal(proxy[0:5, 0:10], data[0:5, 0:10])
    del proxy

    srv.log.clear()
    proxy = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    assert [endpoint for endpoint, _, _ in srv.log] == ["info"]
    assert np.array_equal(proxy[0:5, 0:10], data[0:5, 0:10])
    assert [endpoint for endpoint, _, _ in srv.log] == ["info"]
    assert len(list(cache_dir.glob("*.b2nd"))) == 1


def test_open_urlpath_lazy_exact_cache_path(tmp_path, server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    urlpath = blosc2.URLPath(array.path, urlbase=array.urlbase)
    cache_path = tmp_path / "chosen.b2nd"

    proxy = blosc2.open(urlpath, lazy=True, cache_path=cache_path)
    assert np.array_equal(proxy[0:5, 0:10], data[0:5, 0:10])
    assert proxy.urlpath == str(cache_path)
    assert proxy.schunk.meta["proxy-source"]["source_kind"] == "caterva2"
    del proxy

    srv.log.clear()
    proxy = blosc2.open(cache_path, mode="a")
    assert isinstance(proxy, blosc2.Proxy)
    assert isinstance(proxy.src, blosc2.C2Array)
    assert np.array_equal(proxy[0:5, 0:10], data[0:5, 0:10])
    assert [endpoint for endpoint, _, _ in srv.log] == ["info"]
    assert np.array_equal(proxy[100:105, 0:10], data[100:105, 0:10])
    assert len(srv.log) > 1


def test_open_urlpath_lazy_uses_c2context_without_persisting_token(tmp_path, server):
    token = "session=secret"
    data = _incompressible((20, 20))
    array, _ = server(data, chunks=(10, 20), blocks=(5, 10), cookie=token)
    urlpath = blosc2.URLPath(array.path)
    cache_dir = tmp_path / "cache"

    with blosc2.c2context(urlbase=array.urlbase, auth_token=token):
        proxy = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
        assert np.array_equal(proxy[0:5, 0:5], data[0:5, 0:5])
        assert proxy.schunk.meta["proxy-source"]["urlpath"][2] is None

        cache = next(cache_dir.glob("*.b2nd"))
        reopened = blosc2.open(cache, mode="a")
        assert np.array_equal(reopened[0:5, 0:5], data[0:5, 0:5])


def test_open_urlpath_lazy_rebuilds_stale_cache(tmp_path, server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    urlpath = blosc2.URLPath(array.path, urlbase=array.urlbase)
    cache_dir = tmp_path / "cache"

    proxy = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    assert np.array_equal(proxy[0:5, 0:10], data[0:5, 0:10])
    del proxy

    other = _incompressible((200, 200), seed=1)
    _replace(srv, other, chunks=(100, 200), blocks=(10, 20))

    proxy = blosc2.open(urlpath, lazy=True, cache_dir=cache_dir)
    assert np.array_equal(proxy[0:5, 0:10], other[0:5, 0:10])


def test_open_urlpath_cache_options_need_lazy(tmp_path, server):
    data = _incompressible((20, 20))
    array, _ = server(data, chunks=(10, 20), blocks=(5, 10))
    urlpath = blosc2.URLPath(array.path, urlbase=array.urlbase)

    assert isinstance(blosc2.open(urlpath), blosc2.C2Array)
    with pytest.raises(NotImplementedError, match=r"cache_dir.*lazy=True"):
        blosc2.open(urlpath, cache_dir=tmp_path)
    with pytest.raises(NotImplementedError, match=r"max_concurrency.*lazy=True"):
        blosc2.open(urlpath, max_concurrency=2)


def test_blocks_are_read_over_ranges(server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    srv.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    # The frame index (header, then offsets), one read for the chunk's block
    # offsets, and one for the block the slice lands in
    assert [kind for kind, _, _ in srv.log] == ["fetch"] * 4
    assert {status for _, status, _ in srv.log} == {206}
    assert not _bytes(srv, "chunk")
    # A block of a chunk, not the chunk: an eighth of it here, and never the frame
    assert _bytes(srv, "fetch") < srv.array.schunk.cbytes / 8
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
def test_block_reads_are_correct(server, any_chunk_wants_blocks, shape, chunks, blocks, item):
    data = _incompressible(shape)
    array, _ = server(data, chunks=chunks, blocks=blocks)
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[item], data[item])
    assert np.array_equal(p[...], data)


def test_blocks_carry_the_auth_cookie(server, any_chunk_wants_blocks):
    # fsspec's HTTP filesystem cannot carry this, which is why the block reads of
    # a C2Array are its own rather than an fsspec URL pointed at the server
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), cookie="token=sikrit")
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is not None
    assert not any(status == 401 for _, status, _ in srv.log)


def test_a_streamed_dataset_falls_back_to_chunks(server, any_chunk_wants_blocks):
    # A lazy expression, an HDF5 leaf or a .b2z member is built rather than
    # stored, and the response that carries it ignores Range
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), ranges=False)
    p = blosc2.Proxy(array, mode="w")
    srv.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is None
    assert [kind for kind, _, _ in srv.log] == ["fetch", "chunk"]
    # The probe must not read the body it refused: the whole dataset is what it
    # would have downloaded to find out that ranges are not served
    assert _bytes(srv, "fetch") == len(srv.frame)  # served, but never read

    # And it is never probed again, whatever else is asked for
    assert np.array_equal(p[...], data)
    assert sum(1 for kind, _, _ in srv.log if kind == "fetch") == 1


def test_a_computed_dataset_is_ruled_out_without_a_request(server, any_chunk_wants_blocks):
    # api/info tells a stored dataset from one the server computes: the
    # latter reports `expression` and `operands` where this reports a geometry
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    del array.meta["chunks"]
    srv.log.clear()

    assert array.block_source() is None
    assert not srv.log


def test_small_chunks_are_fetched_whole(server):
    # Read whole, this dataset is 320 KB: no slice of it could save the 1 MiB a
    # round trip is budgeted against, so blocks can never pay anywhere in it.
    # `api/info` says that much, so nothing goes looking for the frame's header,
    # let alone its index or a block -- the read costs the one request it always did
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    srv.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert not array.serves_blocks
    assert array.block_source() is None
    assert [kind for kind, _, _ in srv.log] == ["chunk"]


def test_small_chunks_are_still_split_where_there_are_enough_of_them(server):
    """The bound is over the frame, not over one chunk of it.

    `serves_blocks` used to weigh a single chunk against the budget, and so ruled
    out every dataset of small chunks however many it held -- though a slice wide
    enough saves the budget out of chunks of any size.  This frame has chunks of
    the same size as the one above and thirty-two times as many, and is asked.
    """
    data = _incompressible((3200, 400))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    assert array.serves_blocks
    p = blosc2.Proxy(array, mode="w")
    srv.log.clear()
    np.testing.assert_array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert "fetch" in [kind for kind, _, _ in srv.log]  # read in ranges, not whole


def test_a_c2array_gathers_its_points_at_the_server(server):
    """`C2Array` sends the coordinates and is sent the points, and nothing else.

    A `Proxy` over a source that cannot gather fetches the blocks holding the
    points instead; a Caterva2 server can gather, and a block is nearly all
    waste for a single coordinate.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    array.traffic.reset()

    pts = [1, 30, 59]
    np.testing.assert_array_equal(array[pts, 7], data[pts, 7])
    assert array.traffic.requests == 1  # one gather, not one fetch per point
    gathered = array.traffic.nbytes

    array.traffic.reset()
    array[:]  # what the same points used to cost, at their coarsest
    assert gathered < array.traffic.nbytes


def test_a_c2array_reads_the_coordinates_numpy_reads(server):
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    mask = np.zeros(60, dtype=bool)
    mask[[4, 41]] = True
    for key in ([1, 5, 59], [0, -1], np.array([2, 4]), (([1, 5]), 7), (slice(None), [1, 2]), mask):
        np.testing.assert_array_equal(array[key], data[key])


def test_a_c2array_reads_a_key_that_mixes_points_and_a_bounded_slice(server):
    """The commonest mixed key: coordinates on one axis, a real slice on the next.

    Both bounds travel in the same string, so a bound of 0 has to be told from no
    bound at all -- an empty selection asked for as an open one comes back the
    full width of the axis.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    for key in (
        ([1, 2], slice(3, 5)),
        ([1, 2], slice(None, 0)),
        ([1, 2], slice(2, 0)),
        ([1, 2], slice(0, 4)),
        (slice(10, 12), [1, 2]),
    ):
        np.testing.assert_array_equal(array[key], data[key])


def test_a_c2array_reads_an_ellipsis_and_a_numpy_integer(server):
    # An ellipsis is the run of full slices it abbreviates, which a fetch request
    # can say; refusing it made `array[...]` an error where it used to be the
    # whole dataset
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    for key in (Ellipsis, (slice(0, 5), Ellipsis), (Ellipsis, 3), np.int64(3), (np.int64(3), [1, 2])):
        np.testing.assert_array_equal(array[key], data[key])


def test_a_c2array_gathers_a_two_dimensional_index_array(server):
    """The points come back in the shape the index array asked them in.

    Flattening it here would gather the right points and hand them back in the
    wrong shape, which no error anywhere would catch.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    key = np.array([[0, 1], [2, 3]])
    np.testing.assert_array_equal(array[key], data[key])

    mask = np.zeros((60, 70), dtype=bool)  # one laid over both dimensions
    mask[[4, 41], [5, 60]] = True
    np.testing.assert_array_equal(array[mask], data[mask])


def test_a_key_a_fetch_request_cannot_spell_is_refused(server):
    # It used to be dropped instead, and a dropped index asks for the whole
    # dataset and hands back all of it -- neither what was asked for nor smaller
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    with pytest.raises(IndexError, match="step=1"):
        array[::2]


def test_a_key_too_long_for_an_url_goes_in_a_body(server):
    """Past what a query carries the parameters move to a POST, and nothing else."""
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    key = list(range(60)) * 400  # far more coordinates than an URL holds
    np.testing.assert_array_equal(array[key], data[key])


def test_a_key_an_url_only_holds_once_encoded_goes_in_a_body(server):
    """The encoded length is what the client caps, and it is half again the raw one.

    A key measured as written slips under the limit and is sent as a GET that the
    client then refuses to build -- the very failure the POST route exists for.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    key = list(range(60)) * 16  # 2,723 chars as written, 4,656 encoded
    assert len(blosc2.c2array.key_to_indices(key)) < blosc2.c2array._MAX_QUERY_CHARS
    np.testing.assert_array_equal(array[key], data[key])


def test_a_server_without_the_post_route_says_so(server):
    # 405 says which method, not which key, so it is turned into the sentence a
    # caller can act on -- batch the coordinates, or upgrade the server
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9), post_fetch=False)
    with pytest.raises(IndexError, match="request body"):
        array[list(range(60)) * 400]


def test_a_mid_sized_key_falls_back_to_a_get_where_there_is_no_post_route(server):
    """Below what a client can build, a 405 is worth one GET rather than an error.

    The threshold is set by what a *server* will carry in a request line -- 8 KB
    on nginx -- not by what httpx will build, which is far more.  So keys now
    take the POST route that a GET would have carried, and an older server
    answering 405 must not turn those into a failure.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9), post_fetch=False)
    key = list(range(60)) * 16  # over the query threshold, under what httpx builds
    assert len(blosc2.c2array.key_to_indices(key)) < blosc2.c2array._MAX_URL_CHARS
    np.testing.assert_array_equal(array[key], data[key])


def test_a_reversed_slice_is_placed_on_the_span_it_covers(tmp_path, server, any_chunk_wants_blocks):
    """A fancy key next to a reversed slice fetches the blocks the slice names.

    Read as an empty selection it fetched nothing at all, and a block that was
    never fetched reads as zeros -- which nothing downstream can tell from data.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "reversed.b2nd"), mode="w")
    for key in ((np.array([1, 2]), slice(None, None, -1)), (np.array([1, 105]), slice(150, 10, -1))):
        np.testing.assert_array_equal(p[key], data[key])


def test_a_stepped_slice_reads_the_run_it_lies_in(tmp_path, server, any_chunk_wants_blocks):
    """A step is covered by its run, not refused and not fetched as every chunk.

    `_fancy_cells` calls a key with no advanced index a box, and the box path
    used to hand a step to `get_slice_nchunks`, which raised `IndexError: Step
    parameter is not supported yet` -- so `p[::2]` did not work at all.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "stepped.b2nd"), mode="w")
    for key in (np.s_[::2], np.s_[::-1], np.s_[0:10, ::-1], np.s_[5:150:7, ::3], np.s_[199:0:-2]):
        np.testing.assert_array_equal(p[key], data[key])


def test_a_step_is_placed_on_the_blocks_it_selects(tmp_path, server, any_chunk_wants_blocks):
    """A step reads the blocks holding its coordinates, not the run they lie in.

    With blocks of extent 1 along the stepped axis -- which is how an image stack
    or a tomography volume is chunked, `kevlar-tomo.b2nd` included -- the run a
    step lies in is the whole array, and covering it over-fetches by the step.
    """
    data = _incompressible((60, 200))
    array, srv = server(data, chunks=(1, 200), blocks=(1, 20))
    blocks_per_chunk = 10
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "stepped-exact.b2nd"), mode="w")
    for step in (2, 3, 5):
        key = np.s_[::step]
        wanted = p._wanted_blocks(key)
        assert sum(len(bs) for bs in wanted.values()) == len(range(0, 60, step)) * blocks_per_chunk
        assert sorted(wanted) == list(range(0, 60, step))  # and nothing in between
        np.testing.assert_array_equal(p[key], data[key])


def test_an_unstepped_box_still_says_every_rather_than_counting(tmp_path, server):
    """The exact path is for steps only; a plain slice keeps the cheaper one.

    Naming a covered chunk's blocks one by one is the expensive way to say
    `every`, and that shortcut is what keeps a large box cheap to plan.
    """
    data = _incompressible((60, 200))
    array, srv = server(data, chunks=(1, 200), blocks=(1, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "unstepped.b2nd"), mode="w")
    assert p._plan(np.s_[0:10])[0] == "box"
    assert isinstance(next(iter(p._wanted_blocks(np.s_[0:10]).values())), range)


def test_a_step_costs_its_run_and_not_the_whole_array(tmp_path, server, any_chunk_wants_blocks):
    """A stepped slice bounded to one corner reads that corner, not every chunk."""
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "corner.b2nd"), mode="w")
    p.traffic.reset()
    np.testing.assert_array_equal(p[0:20:2, 0:20:2], data[0:20:2, 0:20:2])
    corner = p.traffic.nbytes

    whole = blosc2.Proxy(
        blosc2.C2Array(array.path, urlbase=array.urlbase),
        urlpath=str(tmp_path / "corner-whole.b2nd"),
        mode="w",
    )
    whole.traffic.reset()
    whole.fetch(())
    assert corner < whole.traffic.nbytes


def test_a_mask_over_a_whole_array_is_a_key_and_not_a_comparison(tmp_path, server):
    """`item == ()` asks a numpy key whether it equals a tuple, which raises.

    A one-dimensional boolean mask is exactly such a key, so the whole-array
    shortcut has to recognise the empty tuple without comparing against it.
    """
    data = _incompressible((60, 40))
    array, srv = server(data, chunks=(30, 40), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "mask.b2nd"), mode="w")
    mask = np.zeros(60, dtype=bool)
    mask[[3, 44]] = True
    np.testing.assert_array_equal(p[mask], data[mask])


def test_a_two_argument_wants_blocks_is_never_handed_the_wave():
    """`max_ranges` and `wants_wave` are opt-ins of their own.

    A source that batches ranges but was written to the two-argument protocol
    used to be called with more, and raised `TypeError` on its first fetch.
    """

    class TwoArg:
        max_ranges = 8

        def wants_blocks(self, nchunk, nwanted):
            return True

    class ThreeArg(TwoArg):
        wants_wave = True

        def wants_blocks(self, nchunk, nwanted, wave=None, nruns=None):
            return wave is not None

    def asking(src, wave):
        proxy = blosc2.Proxy.__new__(blosc2.Proxy)
        proxy.src = src
        return proxy._asking_blocks({0: [1, 2, 3]}, wave)

    wave = {0: 3}
    assert asking(TwoArg(), wave)  # never handed the wave, and says yes anyway
    assert asking(ThreeArg(), wave)
    assert not asking(ThreeArg(), None)  # no wave to weigh, so nothing to say yes to


def test_scattered_points_cost_blocks_and_not_chunks(tmp_path, server, any_chunk_wants_blocks):
    """An integer-array key is placed on the block grid, one block per point.

    Falling back to whole chunks for it -- which is what anything that cannot be
    reduced to a box got -- fetches the chunks the points live in, and a chunk is
    what blocks exist to avoid fetching.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "points.b2nd"), mode="w")
    p.traffic.reset()

    pts = [5, 105]  # one in each chunk, and one block of ten in each
    np.testing.assert_array_equal(p[pts, 7], data[pts, 7])
    blocks = p.traffic.nbytes

    whole = blosc2.Proxy(
        blosc2.C2Array(array.path, urlbase=array.urlbase),
        urlpath=str(tmp_path / "whole.b2nd"),
        mode="w",
    )
    whole.traffic.reset()
    whole.fetch((slice(0, 200), slice(0, 200)))
    assert blocks < whole.traffic.nbytes  # the point of the exercise


def test_a_fancy_key_reads_what_numpy_reads(server, any_chunk_wants_blocks):
    """Whatever is fetched, the answer is the array's own -- the cache decides it.

    Which is why the mapping may only ever be too generous: a block that should
    have been fetched and was not reads as zeros, and nothing downstream could
    tell those from data.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    p = blosc2.Proxy(array, mode="w")
    for key in ([1, 5, 59], [0, -1], ([1, 5], [2, 7]), (np.array([[1, 2], [3, 4]]),), [5]):
        np.testing.assert_array_equal(p[key], data[key])


def test_a_boolean_mask_is_placed_as_exactly_as_a_list(tmp_path, server, any_chunk_wants_blocks):
    """A mask reaches the grid as the coordinates it selects, not as a rule.

    `process_key` turns one into an integer array per dimension it spanned, so
    nothing in the mapping has to know which it was given -- and a mask picking
    two rows of an array costs those rows' blocks, not every chunk they lie in.
    """
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    p = blosc2.Proxy(array, urlpath=str(tmp_path / "mask.b2nd"), mode="w")
    mask = np.zeros(60, dtype=bool)
    mask[[3, 40]] = True
    p.traffic.reset()
    np.testing.assert_array_equal(p[mask], data[mask])
    masked = p.traffic.nbytes

    whole = blosc2.Proxy(
        blosc2.C2Array(array.path, urlbase=array.urlbase),
        urlpath=str(tmp_path / "maskwhole.b2nd"),
        mode="w",
    )
    whole.traffic.reset()
    whole.fetch(())
    assert masked < whole.traffic.nbytes


def test_masks_of_every_shape_read_what_numpy_reads(server, any_chunk_wants_blocks):
    # A mask may span one dimension or several, and may sit anywhere in the key;
    # each spelling pairs its coordinates differently, and none may lose a block
    data = _incompressible((60, 70))
    array, srv = server(data, chunks=(20, 25), blocks=(7, 9))
    p = blosc2.Proxy(array, mode="w")
    rows = np.zeros(60, dtype=bool)
    rows[[1, 59]] = True
    cols = np.zeros(70, dtype=bool)
    cols[[0, 44, 69]] = True
    both = np.zeros((60, 70), dtype=bool)
    both[[2, 50], [3, 60]] = True
    for key in (rows, both, (rows, 7), (slice(None), cols), np.zeros(60, dtype=bool)):
        np.testing.assert_array_equal(p[key], data[key])


def test_a_key_the_cache_cannot_index_is_refused_before_it_is_fetched(server, any_chunk_wants_blocks):
    # Arrays separated by a slice are not supported by the layer below, so the
    # answer was never going to be returned: raise it here rather than after
    # fetching an array's worth of blocks to build it from
    data = _incompressible((30, 40, 50))
    array, srv = server(data, chunks=(10, 20, 25), blocks=(5, 7, 9))
    p = blosc2.Proxy(array, mode="w")
    srv.log.clear()
    with pytest.raises(NotImplementedError):
        p[[1, 5], slice(0, 3), 2]  # two arrays with a slice between them
    assert not srv.log


def test_a_peer_dataset_is_ruled_out_by_what_api_info_says(server, any_chunk_wants_blocks):
    """`Accept-Ranges: none` spares the request that would have found it out.

    A dataset the server mounts from a peer reports the peer's geometry, being
    stored *there*, but is re-serialized here and so refuses a range.  Nothing in
    the payload tells it from a local one; the header does.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), accept_ranges="none")
    srv.log.clear()

    assert not array.serves_blocks
    assert array.block_source() is None
    assert not srv.log  # ... and no request was spent learning it


def test_a_peer_dataset_spares_the_index_read_too(server, any_chunk_wants_blocks):
    """The shortcut belongs where every path passes, not only in `serves_blocks`.

    Reading the frame's index is a different question from reading blocks of its
    chunks, but it goes over the same refused range: `written_chunks` was still
    paying the probe the header had already answered.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), accept_ranges="none")
    srv.log.clear()

    with pytest.raises(blosc2.proxy_source.NotRanged):
        array.written_chunks()
    assert not srv.log  # the same answer the probe would reach, for nothing


def test_a_server_that_names_nothing_is_asked_as_before(server, any_chunk_wants_blocks):
    # An older server names no Accept-Ranges at all, and then the probe is the
    # only way to know -- which is what was always done, and must keep working
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), accept_ranges=None)
    srv.log.clear()

    assert array.serves_blocks
    assert array.block_source() is not None
    assert [kind for kind, _, _ in srv.log] == ["fetch"]  # the probe, and only it


def test_a_server_that_says_bytes_is_believed(server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), accept_ranges="bytes")
    assert array.serves_blocks
    assert array.block_source() is not None


def test_traffic_counts_what_crossed_the_wire(server, any_chunk_wants_blocks):
    """Bytes and requests, counted at the transport and not inferred.

    The point of counting them is that blocks and chunks cost about the same
    wall time on a fast link and differ by the compression ratio in traffic, so
    traffic is the only thing that shows the block path working at all.
    """
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    assert p.traffic is array.traffic  # the array's tally, not a second one

    p.traffic.reset()
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    blocks = (p.traffic.requests, p.traffic.nbytes)
    assert blocks[0] > 0
    assert blocks[1] > 0
    # What the server logged for the data endpoints is what was counted; the
    # `api/info` that opened the handle is metadata and is deliberately not
    served = [(kind, nbytes) for kind, _, nbytes in srv.log if kind != "info"]
    assert blocks[0] == len(served)
    assert blocks[1] <= sum(nbytes for _, nbytes in served)

    # The same slice again is served from the cache and costs nothing more
    before = (p.traffic.requests, p.traffic.nbytes)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert (p.traffic.requests, p.traffic.nbytes) == before


def test_traffic_shows_blocks_costing_fewer_bytes_than_chunks(tmp_path, server, monkeypatch):
    """The comparison the counter exists to make, on one array and one slice."""
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))

    monkeypatch.setattr(blosc2.proxy_source, "BLOCK_MIN_CBYTES", 1 << 62)  # chunks
    whole = blosc2.Proxy(array, urlpath=str(tmp_path / "whole.b2nd"), mode="w")
    whole.traffic.reset()
    assert np.array_equal(whole[0:5, 0:10], data[0:5, 0:10])
    by_chunk = whole.traffic.nbytes

    monkeypatch.setattr(blosc2.proxy_source, "BLOCK_MIN_CBYTES", 0)  # blocks
    split = blosc2.Proxy(
        blosc2.C2Array(array.path, urlbase=array.urlbase),
        urlpath=str(tmp_path / "split.b2nd"),
        mode="w",
    )
    split.traffic.reset()
    assert np.array_equal(split[0:5, 0:10], data[0:5, 0:10])
    assert split.traffic.nbytes < by_chunk  # which is the whole point


def test_a_dataset_that_serves_no_blocks_keeps_the_chunkwise_bitmap(tmp_path, server, monkeypatch):
    # A source that says it serves no blocks -- which a dataset the server
    # computes says, having no frame to read ranges of -- gets a cache that
    # records chunks: the bitmap an older blosc2 also reads, and none of the
    # per-block bookkeeping that would be kept only to say `all of them` every
    # time.  Said here rather than served that way, since a computed dataset
    # reports no partitioning at all and a proxy cannot be laid out over one.
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    monkeypatch.setattr(type(array), "serves_blocks", property(lambda self: False))
    assert not array.serves_blocks  # decided from api/info, without a request
    cache = str(tmp_path / "chunkwise-cache.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert p._blocks_per_chunk == 1

    p.fetch((slice(0, 5), slice(0, 10)))
    kept = p.schunk.vlmeta.getall()
    assert "proxy-fetched" in kept
    assert "proxy-fetched-blocks" not in kept


def test_blocks_accumulate_in_a_chunk(server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    served = len(srv.log)
    # A different block of the same chunk: what is already cached stays cached
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(srv.log) > served
    served = len(srv.log)
    # Both are now in the same cached chunk, and both are still right
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(srv.log) == served


def test_blocks_survive_a_reopened_cache(tmp_path, server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "c2-cache.b2nd")

    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    # A partly filled chunk survives, so the blocks in it do not travel again
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    served = len(srv.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(srv.log) == served
    # ... and the ones missing from it still do
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert len(srv.log) > served
    assert np.array_equal(p[...], data)


def test_a_cache_that_holds_the_slice_costs_no_request(tmp_path, server, any_chunk_wants_blocks):
    # Re-running a script over a cache that already covers the slice: `api/info`
    # is all it takes -- the one the proxy spends looking again at an array that
    # could have been written to since (see `refresh_stamp`).  Nothing opens the
    # frame, because opening it is what `block_source` puts off until a fetch
    # actually wants a chunk -- and this fetch wants none.
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "held.b2nd")
    item = (slice(0, 5), slice(0, 10))
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch(item)

    # A later run: its own array over the same server, its own source
    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    srv.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    p.fetch(item)
    assert np.array_equal(p[item], data[item])
    assert [kind for kind, _, _ in srv.log] == ["info"]

    # ... and a slice the cache does not hold opens the frame then: the header,
    # the layout of the chunk it lands in, and the blocks.  Not where the chunks
    # are -- the earlier run left that in the cache
    assert np.array_equal(p[100:105, 0:10], data[100:105, 0:10])
    assert [kind for kind, _, _ in srv.log] == ["info"] + ["fetch"] * 3


def test_a_kept_index_halves_a_warm_fetch(tmp_path, server, any_chunk_wants_blocks):
    # A later run wanting different blocks of chunks a previous one half filled:
    # where the chunks are and where those blocks are both came out of the cache,
    # so what travels is the header and the blocks, and nothing between
    data = _incompressible((400, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "kept.b2nd")
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch((slice(None), slice(0, 10)))

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    srv.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    assert np.array_equal(p[:, 100:110], data[:, 100:110])
    # The proxy's look at the array, then the header and the blocks -- nothing between
    assert [kind for kind, _, _ in srv.log] == ["info", "fetch", "fetch"]
    assert np.array_equal(p[...], data)  # ... and the rest still reads right


def test_a_kept_index_does_not_open_the_frame_to_be_taken_up(tmp_path, server, any_chunk_wants_blocks):
    # Handing the index to the source would build the source to receive it, and
    # building it reads the header -- a request, at the very moment of a run that
    # may go on to fetch nothing.  It waits with the array until there is a source
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "unopened.b2nd")
    item = (slice(0, 5), slice(0, 10))
    blosc2.Proxy(array, urlpath=cache, mode="a").fetch(item)

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    srv.log.clear()
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    assert again._pending_index is not None  # taken out of the cache, not yet used
    assert [kind for kind, _, _ in srv.log] == ["info"]  # the proxy's look, and no frame read
    p.fetch(item)
    assert [kind for kind, _, _ in srv.log] == ["info"]


def test_a_whole_chunk_cache_is_adopted(tmp_path, server, any_chunk_wants_blocks):
    # A cache left by a run that fetched whole chunks (which is every run before
    # this existed) holds complete chunks, so nothing in it is fetched again
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "chunkwise.b2nd")

    array._block_source = None  # as if the server served no ranges
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    # The same dataset, opened afresh: this one takes the blocks path
    array = blosc2.C2Array(array.path, urlbase=array.urlbase)
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    served = len(srv.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(srv.log) == served
    assert np.array_equal(p[...], data)


def test_blocks_per_chunk_costs_no_request(server):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    srv.log.clear()

    assert array.blocks_per_chunk == math.prod((100 // 10, 200 // 20))
    assert not srv.log


def test_read_range_says_so_when_there_are_no_ranges(server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, _ = server(data, chunks=(100, 200), blocks=(10, 20), ranges=False)

    with pytest.raises(ValueError, match="not served in byte ranges"):
        array.read_range(0, 32)


def test_a_whole_wave_travels_in_one_request(server, any_chunk_wants_blocks):
    # A column through every chunk: each one wants a handful of blocks that lie
    # apart in the file, which without batching is a request each
    data = _incompressible((400, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    assert array.block_source() is not None  # the header, read once per array
    srv.log.clear()

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    # One request for where the four chunks are, one for the layouts of all of
    # them, one for all their blocks -- three waves, not three per chunk
    assert [kind for kind, _, _ in srv.log] == ["fetch", "fetch", "fetch"]
    assert {status for _, status, _ in srv.log} == {206}
    assert _bytes(srv, "fetch") < srv.array.schunk.cbytes / 4

    # Which is the whole of the difference: one request per range otherwise
    other, sub2 = server(data, chunks=(100, 200), blocks=(10, 20), name="unbatched.b2nd")
    other.block_source().max_ranges = 1
    q = blosc2.Proxy(other, mode="w")
    sub2.log.clear()
    assert np.array_equal(q[:, 0:10], data[:, 0:10])
    assert len(sub2.log) > 4 * len(srv.log)


def test_merged_and_reordered_parts_are_read_correctly(server, any_chunk_wants_blocks):
    # The server sorts the spans and merges the ones that touch, so the answer
    # carries fewer parts than were asked for and in an order of its own
    data = _incompressible((400, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    assert np.array_equal(p[...], data)
    assert array.max_ranges > 1  # ... and it never had to stop batching


def test_a_server_that_answers_one_range_stops_being_batched(server, any_chunk_wants_blocks):
    # A server that takes the first span of a multi-range request and ignores
    # the rest: the answer does not cover what was asked for, which is noticed
    # and never repeated
    data = _incompressible((400, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), multipart=False)
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[:, 0:10], data[:, 0:10])
    assert array.max_ranges == 1
    served = len(srv.log)
    assert np.array_equal(p[:, 100:110], data[:, 100:110])
    # One request per range from here on, and no second attempt at batching
    assert len(srv.log) > served + 2
    assert np.array_equal(p[...], data)


# --- a cache is checked against the bytes it was filled from ----------------


def _replace(srv, data, chunks, blocks):
    """Rewrite the served dataset, as an upload of new data would."""
    blosc2.asarray(data, chunks=chunks, blocks=blocks, urlpath=srv.path, mode="w")
    stat = pathlib.Path(srv.path).stat()
    os.utime(srv.path, (stat.st_atime, stat.st_mtime + 10))  # a tick the clock cannot swallow
    srv.reload()


def test_a_cache_is_stamped_with_the_remote_mtime(server):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")

    assert array.stamp == f"{srv.mtime}:{srv.array.schunk.cbytes}"
    assert p.schunk.vlmeta["proxy-stamp"] == array.stamp


def test_a_cache_from_other_bytes_is_refused(tmp_path, server, any_chunk_wants_blocks):
    # Same shape, same partitioning, different data: geometry cannot tell, and
    # every cached chunk (and the offsets it was fetched by) is stale
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "stamped.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(srv, other, chunks=(100, 200), blocks=(10, 20))
    replaced = blosc2.C2Array(array.path, urlbase=array.urlbase)
    assert replaced.stamp != array.stamp

    with pytest.raises(ValueError, match="different remote bytes"):
        blosc2.Proxy(replaced, urlpath=cache, mode="a")

    # ... and mode="w" is the way through, with the new data behind it
    p = blosc2.Proxy(replaced, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], other[0:5, 0:10])
    assert np.array_equal(p[...], other)


def test_a_cache_of_bytes_that_were_replaced_is_emptied(tmp_path, server, any_chunk_wants_blocks):
    # `blosc2.open` rebuilds the proxy over the cache as it stands, with no
    # `mode="a"` to refuse it by: the chunks in there were fetched from a frame
    # that is gone, so what the cache says it holds is dropped rather than served
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "emptied.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(srv, other, chunks=(100, 200), blocks=(10, 20))

    reopened = blosc2.open(cache, mode="a")
    assert np.array_equal(reopened[0:5, 0:10], other[0:5, 0:10])
    assert np.array_equal(reopened[...], other)


def test_a_cache_emptied_of_replaced_bytes_stays_emptied(tmp_path, server, any_chunk_wants_blocks):
    # The stamp is written on the way in, so the run after this one finds a cache
    # whose stamp fits and believes what it says it holds: emptying it has to
    # reach the file, not just the proxy that noticed
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "stillemptied.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[...], data)  # every chunk of it, fetched and recorded
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(srv, other, chunks=(100, 200), blocks=(10, 20))
    noticed = blosc2.open(cache, mode="a")  # opened over the new bytes, dropped unread
    del noticed

    again = blosc2.open(cache, mode="a")
    assert again.schunk.vlmeta["proxy-stamp"] == blosc2.C2Array(array.path, urlbase=array.urlbase).stamp
    assert np.array_equal(again[...], other)


def test_a_read_only_cache_of_replaced_bytes_reads_past_it(tmp_path, server, any_chunk_wants_blocks):
    # Nothing may be written to a cache opened read-only, so it cannot be emptied
    # either -- but nothing of it is believed either, and the read falls through
    # to the source rather than coming back with the bytes that are gone
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "readonlyreplaced.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[...], data)
    del p

    other = _incompressible((200, 200), seed=1)
    _replace(srv, other, chunks=(100, 200), blocks=(10, 20))

    reopened = blosc2.open(cache, mode="r")
    assert np.array_equal(reopened[:], other)  # read past the cache, off the source
    # ... and the cache is left exactly as it was, stamp and chunks alike
    assert reopened.schunk.vlmeta["proxy-stamp"] != blosc2.C2Array(array.path, urlbase=array.urlbase).stamp


def test_a_cache_from_the_same_bytes_is_adopted(tmp_path, server, any_chunk_wants_blocks):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "unchanged.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    again = blosc2.C2Array(array.path, urlbase=array.urlbase)
    assert again.stamp == array.stamp
    p = blosc2.Proxy(again, urlpath=cache, mode="a")
    served = len(srv.log)
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert len(srv.log) == served  # what it holds was not fetched again


def test_no_stamp_when_the_server_reports_no_mtime(tmp_path, server):
    # Then the cache is checked on geometry alone, as every unstamped source is
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    srv.mtime = None  # the server itself reports none, and goes on doing so
    array = blosc2.C2Array(array.path, urlbase=array.urlbase)
    assert array.stamp is None

    cache = str(tmp_path / "unstamped.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="a")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert "proxy-stamp" not in p.schunk.vlmeta.getall()
    del p
    assert blosc2.Proxy(array, urlpath=cache, mode="a") is not None


def test_a_read_only_cache_is_not_stamped(tmp_path, server):
    # `blosc2.open(path, mode="r")` rebuilds the proxy over a cache that may not
    # be written to; recording the stamp there raised instead of opening it
    data = _incompressible((200, 200))
    array, _sub = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "readonly.b2nd")
    p = blosc2.Proxy(array, urlpath=cache, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    del p

    reopened = blosc2.open(cache, mode="r")
    assert np.array_equal(reopened[0:5, 0:10], data[0:5, 0:10])


def test_blocks_of_a_container_leaf(server, any_chunk_wants_blocks):
    """A leaf of a .b2z is a whole frame inside the container, and a server
    serves it from that window -- so the client reads its blocks knowing nothing
    about containers, which is the whole of what it takes."""
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), name="tree.b2z", key="/g/leaf")
    p = blosc2.Proxy(array, mode="w")
    assert array.block_source() is not None
    srv.log.clear()

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert not _bytes(srv, "chunk")
    assert _bytes(srv, "fetch") < srv.array.schunk.cbytes / 8
    assert np.array_equal(p[...], data)


def test_a_container_leaf_is_stamped_like_any_other(server):
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), name="tree.b2z", key="/g/leaf")
    # A leaf has no mtime of its own: the container's is what says it changed
    assert array.stamp == f"{srv.mtime}:{srv.array.schunk.cbytes}"


def test_a_busy_server_is_asked_again(server, any_chunk_wants_blocks):
    # A 503 to the probe says nothing about whether the dataset is served from a
    # file, and cost no download to find out -- unlike the streamed 200 the
    # permanent fallback exists for, so this one is asked again on the next fetch
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), fetch_failures=1)
    p = blosc2.Proxy(array, mode="w")

    assert array.block_source() is None  # the probe was refused ...
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])  # ... and whole chunks served it
    assert array.block_source() is not None  # ... but the next ask gets an answer
    assert np.array_equal(p[...], data)


def test_a_streamed_dataset_is_not_asked_again(server, any_chunk_wants_blocks):
    # The other half of the same rule: a 200 is the dataset itself, and asking
    # again would pay for the whole of it to be told the same thing
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20), ranges=False)
    p = blosc2.Proxy(array, mode="w")

    assert array.block_source() is None
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is None
    assert not any(status == 206 for _, status, _ in srv.log)


def test_a_dataset_that_stops_being_stored_falls_back_to_chunks(server, any_chunk_wants_blocks):
    # A server can stop serving a dataset from a file between one fetch and
    # the next -- replaced by a lazy expression, moved into a container it
    # streams out of.  The fetch that runs into it reads the chunks it was after
    # whole rather than failing, and nothing asks for a range again
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is not None

    srv.ranges = False
    srv.log.clear()
    assert np.array_equal(p[50:55, 0:10], data[50:55, 0:10])
    assert array.block_source() is None  # retired: whole chunks read every dataset
    assert _bytes(srv, "chunk")
    assert np.array_equal(p[...], data)
    # The one refused request is the whole of what the change cost: the body it
    # answered with was never read, and nothing asked for a range again
    assert [status for kind, status, _ in srv.log if kind == "fetch"] == [200]


def test_a_server_too_busy_for_a_range_keeps_its_source(server, any_chunk_wants_blocks):
    # The other half of the rule that governs the probe: a 503 says nothing about
    # how the dataset is served, so the fetch falls back for now and the next one
    # asks for blocks again
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    p = blosc2.Proxy(array, mode="w")
    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    source = array.block_source()

    srv.fetch_failures = 1
    srv.log.clear()
    assert np.array_equal(p[50:55, 0:10], data[50:55, 0:10])
    assert _bytes(srv, "chunk")  # served whole, since the range was refused
    assert array.block_source() is source

    srv.log.clear()
    assert np.array_equal(p[0:5, 100:110], data[0:5, 100:110])
    assert not _bytes(srv, "chunk")  # ... and blocks are asked for again


def test_a_proxy_over_a_cache_survives_a_dataset_that_became_computed(
    tmp_path, server, any_chunk_wants_blocks
):
    # `blosc2.open` rebuilds the proxy over its own cache, and the source it
    # rebuilds may by then be a dataset the server computes -- which reports
    # no partitioning at all, so nothing may go asking one for it
    data = _incompressible((200, 200))
    array, srv = server(data, chunks=(100, 200), blocks=(10, 20))
    cache = str(tmp_path / "computed-cache.b2nd")
    blosc2.Proxy(array, urlpath=cache, mode="w").fetch((slice(0, 5), slice(0, 10)))

    srv.geometry = False
    assert not blosc2.C2Array(array.path, urlbase=array.urlbase).serves_blocks
    reopened = blosc2.open(cache)
    assert isinstance(reopened, blosc2.Proxy)
    assert np.array_equal(reopened[0:5, 0:10], data[0:5, 0:10])  # out of the cache


def test_one_answer_that_misses_its_parts_does_not_end_batching(server, any_chunk_wants_blocks):
    # Batching is worth an order of magnitude, so a single truncated answer is
    # worth retrying a range at a time rather than giving up the whole of it
    data = _incompressible((400, 400))
    array, srv = server(data, chunks=(200, 200), blocks=(10, 20), bad_parts=1)
    p = blosc2.Proxy(array, mode="w")

    item = (slice(190, 210), slice(190, 210))  # a corner of each of the four chunks
    assert np.array_equal(p[item], data[item])
    assert not srv.bad_parts  # the answer that carried one part was asked for ...
    assert array.max_ranges > 1  # ... and cost the batching nothing
    assert np.array_equal(p[...], data)


def test_a_part_that_ends_early_is_refused():
    # A part that starts inside a span and stops short of its end would slice to
    # a short payload, which is spliced against a `bstarts` promising the whole
    # of it -- so it is refused, and the fetch falls back to a range per request
    parts = [(100, b"12345", 200)]  # (where it starts, its bytes, the frame's length)
    assert blosc2.c2array._span_of(parts, 100, 5, "url") == b"12345"
    with pytest.raises(blosc2.proxy_source.PartsMissing):
        blosc2.c2array._span_of(parts, 100, 6, "url")
    with pytest.raises(blosc2.proxy_source.PartsMissing):
        blosc2.c2array._span_of(parts, 103, 4, "url")


class _Answer:
    """The little of a response that taking a multipart body apart reads."""

    def __init__(self, content, **headers):
        self.content = content
        self.headers = {name.replace("_", "-"): value for name, value in headers.items()}


def test_a_payload_that_spells_the_boundary_is_read_whole():
    # Compressed bytes hold whatever they hold, the boundary and a trailing CRLF
    # among them: each part is cut to the length its own Content-Range gives, so
    # what the data happens to spell decides nothing
    payload = b"--c2boundary\r\n\r\nnot a boundary at all\r\ntail"
    body = (
        b"--c2boundary\r\nContent-Type: application/octet-stream\r\n"
        + f"Content-Range: bytes 64-{64 + len(payload) - 1}/1000\r\n\r\n".encode()
        + payload
        + b"\r\n--c2boundary--\r\n"
    )
    answer = _Answer(body, content_type="multipart/byteranges; boundary=c2boundary")
    assert blosc2.c2array._byteranges(answer) == [(64, payload, 1000)]
    assert blosc2.c2array._span_of([(64, payload, 1000)], 64, len(payload), "url") == payload


def test_a_multipart_answer_is_taken_apart_in_order():
    parts = [(0, b"first-part-bytes"), (500, b"second\r\npart")]
    body = b""
    for start, data in parts:
        body += b"--sep\r\n" + f"Content-Range: bytes {start}-{start + len(data) - 1}/1000\r\n\r\n".encode()
        body += data + b"\r\n"
    body += b"--sep--\r\n"
    answer = _Answer(body, content_type='multipart/byteranges; boundary="sep"')
    assert blosc2.c2array._byteranges(answer) == [(0, parts[0][1], 1000), (500, parts[1][1], 1000)]


def test_a_part_without_a_content_range_is_refused():
    body = b"--sep\r\nContent-Type: application/octet-stream\r\n\r\nbytes\r\n--sep--\r\n"
    answer = _Answer(body, content_type="multipart/byteranges; boundary=sep")
    with pytest.raises(blosc2.proxy_source.NotRanged, match="Content-Range"):
        blosc2.c2array._byteranges(answer)


def test_a_part_that_ends_where_the_frame_does_is_kept():
    # ... unless it ends because the frame does, which is the one short read
    # `read_range` allows: a frame shorter than the prefetch an open asks for
    # comes back clipped, and refusing it would write the dataset off as streamed
    parts = [(100, b"12345", 105)]
    assert blosc2.c2array._span_of(parts, 100, 8192, "url") == b"12345"
    assert blosc2.c2array._span_of(parts, 103, 8192, "url") == b"45"
    with pytest.raises(blosc2.proxy_source.PartsMissing):
        blosc2.c2array._span_of(parts, 105, 1, "url")  # nothing there to be short of
    # A server that will not say how long the whole is leaves nothing to check a
    # short answer against, so a short answer is a missing one
    with pytest.raises(blosc2.proxy_source.PartsMissing):
        blosc2.c2array._span_of([(100, b"12345", None)], 100, 6, "url")


def test_a_small_frame_is_read_over_ranges(server, any_chunk_wants_blocks):
    # The whole of the frame arrives in the first read an open asks for, which
    # is the clipped answer above, and the dataset is served in blocks all the same
    data = np.arange(200, dtype="i4").reshape(20, 10)
    array, srv = server(data, chunks=(10, 10), blocks=(5, 10))
    assert len(srv.frame) < blosc2.proxy_source._FRAME_PREFETCH
    p = blosc2.Proxy(array, mode="w")

    assert np.array_equal(p[0:5, 0:10], data[0:5, 0:10])
    assert array.block_source() is not None
    assert not _bytes(srv, "chunk")


def test_the_shared_client_keeps_no_cookies():
    # One client serves every C2Array, so a `Set-Cookie` kept from any of them
    # would start authorizing requests that asked for none
    import httpx

    client = blosc2.c2array._sync_client()
    request = httpx.Request("GET", "http://127.0.0.1/api/fetch/x")
    response = httpx.Response(200, headers={"set-cookie": "session=secret; Path=/"}, request=request)
    client.cookies.extract_cookies(response)
    assert not dict(client.cookies)
