#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Filling a pre-sized remote array a chunk at a time, from several writers.

The stand-in here answers the write contract a subscriber is meant to answer:
one chunk per request, into a slot nothing was written to yet, refused with a
409 otherwise.  That refusal is the whole of the coordination -- the frame's own
offsets say which slots are free, so two writers that both believe they own a
chunk are resolved by the array rather than by anything either of them holds.
"""

import concurrent.futures
import contextlib
import json
import os
import pathlib
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

import blosc2

# The stand-in binds a real socket, which wasm32 has no listen(2) for
pytestmark = pytest.mark.skipif(blosc2.IS_WASM, reason="no listening sockets on wasm32")

CHUNKS = (1000,)
BLOCKS = (250,)
NCHUNKS = 6
SHAPE = (CHUNKS[0] * NCHUNKS,)


class _Subscriber:
    """A Caterva2-shaped server over one .b2nd file, that also accepts writes."""

    def __init__(self, path):
        self.path = str(path)
        self.log = []  # (endpoint, status)
        self.lock = threading.Lock()  # what the real server does with holding_lock()
        # One handle for the life of the server, and the only one in this process:
        # a second handle open over a frame another is writing is the stale-handle
        # hazard of `todo/locking-mwmr.md`, and it is silent -- the write reports
        # nothing and the frame is left unreadable
        self.array = blosc2.open(self.path, mode="a", locking=True)
        self.reload()

    def reload(self):
        self.mtime = pathlib.Path(self.path).stat().st_mtime

    @property
    def meta(self):
        array = self.array
        schunk = array.schunk
        return {
            "shape": list(array.shape),
            "chunks": list(array.chunks),
            "blocks": list(array.blocks),
            "dtype": str(array.dtype),
            "mtime": self.mtime,
            "schunk": {
                "cparams": {"typesize": array.dtype.itemsize},
                "nbytes": schunk.nbytes,
                "cbytes": schunk.cbytes,
                "cratio": schunk.cratio,
                "blocksize": schunk.blocksize,
                "vlmeta": schunk.vlmeta.getall(),
            },
        }

    def write_chunk(self, nchunk, chunk):
        """The endpoint's body: refuse a slot that holds anything, then store.

        Serialized, as the server serializes it, and the whole of the check is
        the slot's own tag: UNINIT and nothing else means never written, since a
        writer that stored an all-zero chunk stored something.
        """
        with self.lock:
            array = self.array
            infos = list(array.schunk.iterchunks_info())
            if not 0 <= nchunk < len(infos):
                return 404, {"detail": "no such chunk"}
            if infos[nchunk].special is not blosc2.SpecialValue.UNINIT:
                return 409, {"detail": f"chunk {nchunk} was already written"}
            nbytes = blosc2.get_cbuffer_sizes(chunk)[0]
            if nbytes != array.schunk.chunksize:
                return 400, {"detail": "the chunk does not match the array's chunkshape"}
            array.schunk.update_chunk(nchunk, chunk)
            vlmeta = array.schunk.vlmeta
            if "fill_nonce" not in vlmeta.getall():
                # What names this array, as against another that comes to sit at
                # the same path with the same size
                vlmeta["fill_nonce"] = uuid.uuid4().hex
            # Counted through the handle that wrote, rather than a fresh open of
            # a frame the write just moved
            written = sum(
                1 for i in array.schunk.iterchunks_info() if i.special is not blosc2.SpecialValue.UNINIT
            )
            if written == len(infos) and vlmeta.getall().get("fill_state", "filling") == "filling":
                vlmeta["fill_state"] = "complete"
            self.reload()
            return 200, {"written": written, "nchunks": len(infos), "nchunk": nchunk}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        with contextlib.suppress(ConnectionResetError, BrokenPipeError):
            super().handle()

    def _send(self, status, body, headers=(), endpoint=""):
        self.server.subscriber.log.append((endpoint, status))
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        sub = self.server.subscriber
        endpoint = self.path.split("/")[2]
        if endpoint == "info":
            self._send(200, json.dumps(sub.meta).encode(), endpoint="info")
        elif endpoint == "chunk":
            nchunk = int(self.path.split("nchunk=")[1])
            with sub.lock:
                self._send(200, sub.array.schunk.get_chunk(nchunk), endpoint="chunk")
        elif endpoint == "fetch":
            self._fetch(sub)
        else:
            self._send(404, b"", endpoint=endpoint)

    def do_POST(self):
        sub = self.server.subscriber
        endpoint = self.path.split("/")[2].split("?")[0]
        if endpoint != "chunk":
            self._send(404, b"", endpoint=endpoint)
            return
        nchunk = int(self.path.split("nchunk=")[1])
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        status, answer = sub.write_chunk(nchunk, body)
        self._send(status, json.dumps(answer).encode(), endpoint="write")

    def _fetch(self, sub):
        """Ranges over the frame's bytes, or the slice itself when none is asked.

        Both halves of what a subscriber serves: `C2Array.__getitem__` asks for a
        slice and gets a cframe of it, while the block path asks for byte ranges
        of the file.  A fill has to be visible through both.
        """
        query = parse_qs(urlparse(self.path).query)
        frame = pathlib.Path(sub.path).read_bytes()
        wanted = self.headers.get("Range")
        if not wanted:
            with sub.lock:
                array = sub.array
                sliced = array[_parse_slice(query.get("slice_", [""])[0], array.ndim)]
            self._send(200, blosc2.asarray(sliced).to_cframe(), endpoint="fetch")
            return
        spans = []
        for span in wanted.removeprefix("bytes=").split(","):
            first, _, last = span.partition("-")
            spans.append((int(first), min(int(last), len(frame) - 1) if last else len(frame) - 1))
        # Sorted and merged, the way Starlette answers several ranges
        spans.sort()
        merged = [spans[0]]
        for first, last in spans[1:]:
            if first <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], last))
            else:
                merged.append((first, last))
        if len(merged) == 1:
            first, last = merged[0]
            self._send(
                206,
                frame[first : last + 1],
                [
                    ("Content-Range", f"bytes {first}-{last}/{len(frame)}"),
                    ("Accept-Ranges", "bytes"),
                ],
                endpoint="fetch",
            )
            return
        boundary = "c2boundary"
        body = b""
        for first, last in merged:
            body += (
                f"--{boundary}\r\nContent-Type: application/octet-stream\r\n"
                f"Content-Range: bytes {first}-{last}/{len(frame)}\r\n\r\n"
            ).encode()
            body += frame[first : last + 1] + b"\r\n"
        body += f"--{boundary}--\r\n".encode()
        self._send(
            206,
            body,
            [("Content-Type", f"multipart/byteranges; boundary={boundary}"), ("Accept-Ranges", "bytes")],
            endpoint="fetch",
        )


def _parse_slice(text, ndim):
    """What `blosc2.slice_to_string` wrote, read back."""
    if not text:
        return slice(None)
    parts = []
    for part in text.split(","):
        part = part.strip()
        if ":" in part:
            first, _, last = part.partition(":")
            parts.append(slice(int(first) if first else None, int(last) if last else None))
        else:
            parts.append(int(part))
    return tuple(parts) if len(parts) > 1 else parts[0]


@pytest.fixture
def subscriber(tmp_path):
    """A pre-sized, unwritten array and a server over it."""
    path = tmp_path / "run.b2nd"
    presized = blosc2.uninit(SHAPE, dtype=np.int32, chunks=CHUNKS, blocks=BLOCKS, urlpath=str(path))
    del presized  # the server's handle is to be the only one over this file
    sub = _Subscriber(path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.subscriber = sub
    threading.Thread(target=server.serve_forever, daemon=True).start()
    urlbase = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        yield blosc2.C2Array("run.b2nd", urlbase=urlbase), sub
    finally:
        server.shutdown()
        server.server_close()


def _chunk(nchunk, value=None):
    """A chunk of the array's geometry, tagged by which chunk it is."""
    data = np.full(CHUNKS, nchunk if value is None else value, dtype=np.int32)
    return blosc2.compress2(data, typesize=4, blocksize=BLOCKS[0] * 4)


def test_a_chunk_written_is_read_back(subscriber):
    array, sub = subscriber
    array.update_chunk(2, _chunk(2))
    assert np.all(array[2 * CHUNKS[0] : 3 * CHUNKS[0]] == 2)
    # ... and nothing else was touched
    assert np.all(array[0 : CHUNKS[0]] == 0)


def test_a_second_write_is_refused(subscriber):
    array, sub = subscriber
    array.update_chunk(1, _chunk(1))
    with pytest.raises(blosc2.ChunkAlreadyWritten):
        array.update_chunk(1, _chunk(1, value=99))
    assert np.all(array[CHUNKS[0] : 2 * CHUNKS[0]] == 1)  # the first write stands


def test_a_chunk_of_the_wrong_shape_is_refused(subscriber):
    array, sub = subscriber
    wrong = blosc2.compress2(np.zeros(CHUNKS[0] // 2, dtype=np.int32), typesize=4)
    with pytest.raises(Exception):  # noqa: B017 -- an HTTP 400, whatever httpx calls it
        array.update_chunk(0, wrong)
    assert not array.written_chunks().any()


def test_written_chunks_tracks_the_fill(subscriber):
    array, sub = subscriber
    assert list(array.written_chunks()) == [False] * NCHUNKS
    array.update_chunk(3, _chunk(3))
    assert list(array.written_chunks()) == [False, False, False, True, False, False]
    array.update_chunk(0, _chunk(0))
    assert list(array.written_chunks()) == [True, False, False, True, False, False]


def test_a_written_chunk_of_zeros_counts_as_written(subscriber):
    """The reason a pre-sized array is filled with `uninit` and not with `zeros`.

    Compressing an all-zero buffer gives a run-length chunk, so a slot written
    with one is special again -- but tagged as zeros, not as uninitialized, which
    is what keeps it distinguishable from a slot nobody has reached yet.
    """
    array, sub = subscriber
    array.update_chunk(4, _chunk(4, value=0))
    assert array.written_chunks()[4]
    assert np.all(array[4 * CHUNKS[0] : 5 * CHUNKS[0]] == 0)
    with pytest.raises(blosc2.ChunkAlreadyWritten):
        array.update_chunk(4, _chunk(4))


def test_a_fill_leaves_the_chunks_before_it_where_they_were(subscriber):
    """What makes an append-only fill cheap to read alongside."""
    array, sub = subscriber
    array.update_chunk(0, _chunk(0, value=42))
    placed = array.get_chunk(0)
    for nchunk in range(1, NCHUNKS):
        array.update_chunk(nchunk, _chunk(nchunk))
    assert array.get_chunk(0) == placed
    assert np.all(array[0 : CHUNKS[0]] == 42)
    for nchunk in range(1, NCHUNKS):
        assert np.all(array[nchunk * CHUNKS[0] : (nchunk + 1) * CHUNKS[0]] == nchunk)


def test_concurrent_writers_fill_the_array(subscriber):
    array, sub = subscriber
    urlbase = array.urlbase

    def fill(nchunk):
        # A writer of its own, as a separate process would have
        writer = blosc2.C2Array("run.b2nd", urlbase=urlbase)
        writer.update_chunk(nchunk, _chunk(nchunk))
        return nchunk

    with concurrent.futures.ThreadPoolExecutor(max_workers=NCHUNKS) as pool:
        assert sorted(pool.map(fill, range(NCHUNKS))) == list(range(NCHUNKS))

    assert array.written_chunks().all()
    expected = np.repeat(np.arange(NCHUNKS, dtype=np.int32), CHUNKS[0])
    np.testing.assert_array_equal(array[:], expected)


def test_two_writers_racing_for_one_chunk_leave_one_winner(subscriber):
    array, sub = subscriber
    urlbase = array.urlbase
    barrier = threading.Barrier(2)

    def fill(value):
        writer = blosc2.C2Array("run.b2nd", urlbase=urlbase)
        barrier.wait()
        try:
            writer.update_chunk(5, _chunk(5, value=value))
            return "won"
        except blosc2.ChunkAlreadyWritten:
            return "lost"

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(fill, (7, 8)))
    assert outcomes == ["lost", "won"]
    stored = np.unique(array[5 * CHUNKS[0] : 6 * CHUNKS[0]])
    assert len(stored) == 1
    assert stored[0] in (7, 8)


def test_a_reader_sees_chunks_that_land_after_it_read(subscriber):
    array, sub = subscriber
    array.update_chunk(0, _chunk(0))
    assert np.all(array[0 : CHUNKS[0]] == 0)  # reads, and indexes, the frame
    array.update_chunk(1, _chunk(1))
    assert np.all(array[CHUNKS[0] : 2 * CHUNKS[0]] == 1)


@pytest.mark.asyncio
async def test_chunks_can_be_written_off_the_event_loop(subscriber):
    array, sub = subscriber
    answer = await array.aupdate_chunk(2, _chunk(2))
    assert answer["written"] == 1
    with pytest.raises(blosc2.ChunkAlreadyWritten):
        await array.aupdate_chunk(2, _chunk(2))
    await array.aclose()
    assert np.all(array[2 * CHUNKS[0] : 3 * CHUNKS[0]] == 2)


def _fill(array, values=None):
    for nchunk in range(NCHUNKS):
        array.update_chunk(nchunk, _chunk(nchunk, value=None if values is None else values))


def test_a_filling_array_is_stamped_afresh_on_every_write(subscriber):
    """A cache of an array still being filled has to be thrown away, not kept.

    What it holds of a chunk nobody had written is the zeros an unwritten chunk
    reads as, and the run-length offset it had; once a writer fills that slot
    both are wrong, and nothing in the cache tells them from the chunks that are
    still good.
    """
    array, sub = subscriber
    stamps = []
    for nchunk in range(3):
        array.update_chunk(nchunk, _chunk(nchunk))
        stamps.append(blosc2.C2Array("run.b2nd", urlbase=array.urlbase).stamp)
    assert len(set(stamps)) == len(stamps)


def test_a_complete_array_keeps_one_stamp(subscriber):
    """Once every slot is claimed the array cannot change, so a cache of it stands."""
    array, sub = subscriber
    _fill(array)

    def stamp():
        return blosc2.C2Array("run.b2nd", urlbase=array.urlbase).stamp

    complete = stamp()
    assert complete.startswith("n")
    # An mtime that moved for reasons of its own is not a reason to refetch
    os.utime(sub.path, (time.time() + 10, time.time() + 10))
    sub.reload()
    assert stamp() == complete


def test_two_arrays_at_one_path_are_told_apart(subscriber, tmp_path):
    """The hole a size and an mtime leave, which is what the nonce closes.

    Both arrays here are filled with constant chunks, so they compress to exactly
    the same size; the mtime is then made equal by hand.  Nothing but the nonce
    separates them, and a cache of the first served against the second would be
    wrong in every chunk.
    """
    array, sub = subscriber
    _fill(array, values=1)
    first = blosc2.C2Array("run.b2nd", urlbase=array.urlbase)
    first_stamp, first_size = first.stamp, pathlib.Path(sub.path).stat().st_size

    # A different array comes to sit at the same path, of the same size
    replacement = tmp_path / "replacement.b2nd"
    presized = blosc2.uninit(SHAPE, dtype=np.int32, chunks=CHUNKS, blocks=BLOCKS, urlpath=str(replacement))
    del presized
    sub.array = blosc2.open(str(replacement), mode="a", locking=True)
    sub.path = str(replacement)
    for nchunk in range(NCHUNKS):
        sub.write_chunk(nchunk, _chunk(nchunk, value=2))
    sub.reload()

    assert pathlib.Path(sub.path).stat().st_size == first_size  # same bytes on disk
    os.utime(sub.path, (first.meta["mtime"], first.meta["mtime"]))
    sub.reload()
    second = blosc2.C2Array("run.b2nd", urlbase=array.urlbase)
    assert second.meta["mtime"] == first.meta["mtime"]  # ... and the same mtime
    assert second.stamp != first_stamp


def test_an_array_with_no_nonce_is_stamped_as_before(tmp_path):
    """An ordinary dataset, never filled a chunk at a time, is unchanged by this."""
    path = tmp_path / "plain.b2nd"
    blosc2.asarray(np.arange(4000, dtype=np.int32), chunks=(1000,), blocks=(250,), urlpath=str(path))
    sub = _Subscriber(path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.subscriber = sub
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        array = blosc2.C2Array("plain.b2nd", urlbase=f"http://127.0.0.1:{server.server_address[1]}/")
        assert array.stamp == f"{sub.mtime}:{array.meta['schunk']['cbytes']}"
    finally:
        server.shutdown()
        server.server_close()


def test_a_cache_of_a_complete_array_survives_a_second_run(subscriber, tmp_path):
    """What the nonce is for: the finished array is the one read again and again.

    The cache is reopened after the array's mtime has moved under it, which is
    what a republish or a copy does.  Nothing was refetched -- the stamp says it
    is the same array, and a complete one cannot have changed.
    """
    array, sub = subscriber
    _fill(array)
    cache = str(tmp_path / "cache.b2nd")
    proxy = blosc2.Proxy(blosc2.C2Array("run.b2nd", urlbase=array.urlbase), urlpath=cache, mode="w")
    expected = proxy[:]
    del proxy

    os.utime(sub.path, (time.time() + 10, time.time() + 10))
    sub.reload()
    sub.log.clear()
    proxy = blosc2.Proxy(blosc2.C2Array("run.b2nd", urlbase=array.urlbase), urlpath=cache, mode="a")
    np.testing.assert_array_equal(proxy[:], expected)
    assert not [entry for entry in sub.log if entry[0] in ("chunk", "fetch")]
