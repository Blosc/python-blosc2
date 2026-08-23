#!/usr/bin/env python

#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Would block-granular reads beat whole-chunk ones for a given Caterva2 dataset?

A ``Proxy`` over a ``C2Array`` used to fetch one whole compressed chunk per
request, through ``api/chunk``.  A chunk is made of blocks, which blosc2
compresses and decompresses independently, and Caterva2 serves a *stored*
dataset straight from its file, so ``api/fetch`` honours a ``Range`` header and
a slice can fetch only the blocks it touches.  Whether that pays depends on
three things this script measures against a server of your choosing:

- whether the dataset **serves ranges at all**.  One stored from a file does;
  one the server computes -- a lazy expression, an HDF5 leaf, a ``.b2z``
  member -- is streamed, cannot honour a range, and keeps to whole chunks;
- the **request plan**: how many requests each mode issues and how many bytes
  they carry, read from the frame's own chunk headers for a few hundred bytes;
- the **wall time** of each pattern, which is the shipped code fetching real
  slices from a real server.

Usage
-----
    # a local array, served by a stand-in server over loopback
    python cat2-block-granularity.py mydata.b2nd

    # ... with a network put back in front of every request
    python cat2-block-granularity.py mydata.b2nd --latency-ms 45 --bandwidth-mbs 10

    # ... served the way a computed dataset is, which is what the fallback costs
    python cat2-block-granularity.py mydata.b2nd --streamed

    # against a real server
    python cat2-block-granularity.py @public/examples/kevlar-tomo.b2nd \\
        --urlbase https://cat2.cloud/demo

    # ... and what filling a pre-sized array costs, one chunk per request
    python cat2-block-granularity.py mydata.b2nd --write

    # ... an authenticated dataset
    python cat2-block-granularity.py @personal/mine.b2nd --urlbase http://localhost:8000 \\
        --username me@example.com --password foobar11

Three modes are timed, each of them the shipped code with one thing changed:

- ``chunks``: one ``api/chunk`` request per touched chunk.  What a proxy over a
  C2Array did before blocks existed, and what it still does for a dataset that
  cannot serve ranges or whose chunks are too small to be worth taking apart.
- ``blocks``: one request for the block offsets of each chunk worth taking
  apart, then one per coalesced run of the blocks wanted.  Two dependent waves.
- ``multipart``: the same two waves, each collapsed into a single request --
  RFC 7233 lets one ``Range`` header name many spans, and Caterva2 answers
  ``multipart/byteranges``.  No object store offers this.

The stand-in server answers ``api/info``, ``api/fetch`` and ``api/chunk``
the way Caterva2 does, ranges and multipart included (it sorts and merges the
spans it is given, as Starlette does, which is what the client has to survive).
Its request and byte counts are exact.  Its *times* are not a server's:
loopback answers in a fraction of a millisecond, where a server over a WAN
takes tens of milliseconds, which is the regime the whole trade lives in.
``--latency-ms`` and ``--bandwidth-mbs`` put a stated network back in front of
each request; cat2.cloud from Europe measures about ``--latency-ms 45
--bandwidth-mbs 10``.  The simulated bandwidth is *per request*, so eight
parallel ones get eight times as much of it -- which is about right for an
object store and about wrong for one server, and is why ``multipart`` can
come out behind ``blocks`` there while it wins against the real thing.

``--write`` measures the other direction: an array is laid out empty and filled
a chunk at a time, which is how several writers fill one array at once.  Three
things, and the first is the only one that goes over the wire:

- the **fill**, serial and then ``--concurrency`` writers at once.  The
  server serializes the writes themselves -- each takes the frame's
  exclusive lock -- so what overlaps is the round trip, and the gain is whatever
  share of a write that was.  Over loopback it is almost none; put a network in
  front with ``--latency-ms`` and it is most of it;
- what the **server pays to store one chunk**, into an empty slot and over a live
  one, timed locally where a round trip would bury the difference.  A slot
  holding nothing is appended past the offsets and moves no other chunk; one
  holding a chunk has every byte of payload after it read and written back.
  That difference is why a fill writes each slot once and refuses a second write;
- what **reading the progress** costs, from the frame's offsets against walking
  its chunks.  The offsets are one decompress whatever the count; the walk is a
  read per chunk, so the two cross over as an array grows.

Against a real server ``--write`` needs ``--write-target``: an empty
pre-sized array to fill, since laying one out is not this script's business on
someone else's server.  Only the serial fill runs there -- a slot is written
once, so a second timed fill needs a second array.

Bytes counted are payload: the multipart envelope (about a hundred bytes per
part) and the HTTP headers of every request are not in them.
"""

import argparse
import concurrent.futures
import http.server
import itertools
import json
import math
import pathlib
import shutil
import statistics
import struct
import tempfile
import threading
import time

import blosc2
from blosc2 import c2array

CHUNK_HEADER = blosc2.proxy_source._CHUNK_HEADER_LEN


#
# A stand-in server, so this runs with no service to point at
#


UNINIT = 0x4
"""What a frame codes in a chunk's flags byte for a slot never written to."""


class Cat2Server:
    """Caterva2's read endpoints over one local .b2nd file, and its write one."""

    def __init__(self, urlpath, streamed=False, writable=False):
        self.path = pathlib.Path(urlpath)
        self.name = self.path.name
        self.size = self.path.stat().st_size
        # A writable dataset is opened once, for the life of the server, and
        # written through that one handle: a second handle over a frame this one
        # writes leaves it unreadable, and says nothing while doing so
        self.writable = writable
        self.array = blosc2.open(str(self.path), mode="a" if writable else "r", locking=writable)
        self.lock = threading.Lock()
        # A dataset the server would compute rather than store: served by a
        # body builder, which has no way to honour a Range
        self.streamed = streamed

    def close(self):
        """Let go of the file this held open, so the scratch tree can be removed.

        A writable server keeps one handle for its whole life, and a run that
        fills several arrays leaves one behind per array otherwise -- still
        holding files that `shutil.rmtree` then unlinks under them.
        """
        self.array = None

    def write_chunk(self, nchunk, chunk):
        """Caterva2's write contract: one chunk, into a slot that holds none.

        The refusal is the whole of the coordination between writers, and the
        check is O(1) -- a lazy chunk is its header, where walking the array
        would make a fill cost the square of its length.
        """
        with self.lock:
            schunk = self.array.schunk
            if not 0 <= nchunk < schunk.nchunks:
                return 404, {"detail": "no such chunk"}
            nbytes, _, blocksize = blosc2.get_cbuffer_sizes(chunk)
            if nbytes != schunk.chunksize or blocksize != schunk.blocksize:
                return 400, {"detail": "the chunk does not match the array's geometry"}
            with schunk.holding_lock():
                if (schunk.get_lazychunk(nchunk)[31] >> 4) & 0x7 != UNINIT:
                    return 409, {"detail": f"chunk {nchunk} was already written"}
                schunk.update_chunk(nchunk, chunk)
            self.size = self.path.stat().st_size
            return 200, {"nchunk": nchunk}

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

    def read(self, start, end):
        """The bytes at [start, end], seeked to rather than materialized."""
        with self.path.open("rb") as frame:
            frame.seek(start)
            return frame.read(end - start + 1)


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    BOUNDARY = "c2boundary"

    def log_message(self, *args):
        pass

    def _send(self, status, body, headers=()):
        self.send_response(status)
        for name, value in headers:
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _dataset(self):
        """Which of the served datasets this request names.

        One of them until a fill is being measured, when there is a second: the
        array being filled, which is not the array being read.
        """
        target = getattr(self.server, "target", None)
        if target is not None and self.path.split("?")[0].endswith(target.name):
            return target
        return self.server.cat2

    def do_POST(self):  # BaseHTTPRequestHandler's own spelling
        srv = self._dataset()
        endpoint = self.path.split("/")[2].split("?")[0]
        if endpoint != "chunk" or not srv.writable:
            self._send(404, b"")
            return
        nchunk = int(self.path.split("nchunk=")[1])
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        status, answer = srv.write_chunk(nchunk, body)
        self._send(status, json.dumps(answer).encode())

    def do_GET(self):  # BaseHTTPRequestHandler's own spelling
        srv = self._dataset()
        endpoint = self.path.split("/")[2]
        if endpoint == "info":
            self._send(200, json.dumps(srv.meta()).encode())
        elif endpoint == "chunk":
            nchunk = int(self.path.split("nchunk=")[1])
            self._send(200, srv.array.schunk.get_chunk(nchunk))
        elif endpoint == "fetch":
            self._fetch(srv)
        else:
            self._send(404, b"")

    def _fetch(self, srv):
        wanted = self.headers.get("Range")
        if srv.streamed:
            # What the streaming paths answer since they were made honest: a 416
            # instead of the whole body with a 200 that no client could notice
            if wanted:
                self._send(416, b"", [("Accept-Ranges", "none")])
            else:
                self._send(200, srv.read(0, srv.size - 1), [("Accept-Ranges", "none")])
            return
        if not wanted:
            self._send(200, srv.read(0, srv.size - 1), [("Accept-Ranges", "bytes")])
            return
        spans = []
        for span in wanted.removeprefix("bytes=").split(","):
            start, end = (int(n) for n in span.split("-"))
            spans.append((start, min(end, srv.size - 1)))
        # Starlette sorts the spans and merges the ones that touch, and answers a
        # plain 206 when only one is left, so a client cannot count on a part per
        # span nor on the order it asked in
        spans.sort()
        merged = [spans[0]]
        for start, end in spans[1:]:
            if start <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if len(merged) == 1:
            start, end = merged[0]
            self._send(
                206,
                srv.read(start, end),
                [("Content-Range", f"bytes {start}-{end}/{srv.size}"), ("Accept-Ranges", "bytes")],
            )
            return
        body = b""
        for start, end in merged:
            body += (
                f"--{self.BOUNDARY}\r\nContent-Type: application/octet-stream\r\n"
                f"Content-Range: bytes {start}-{end}/{srv.size}\r\n\r\n"
            ).encode()
            body += srv.read(start, end) + b"\r\n"
        body += f"--{self.BOUNDARY}--\r\n".encode()
        self._send(
            206,
            body,
            [
                ("Content-Type", f"multipart/byteranges; boundary={self.BOUNDARY}"),
                ("Accept-Ranges", "bytes"),
            ],
        )


def stand_in(urlpath, streamed=False):
    """Serve *urlpath* as ``@public/<name>``, and return (server, urlbase, path).

    The array a fill writes into is installed later, by `make_presize`, which
    lays out a fresh one per timed fill; until then there is only the one served
    here.
    """
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.cat2 = Cat2Server(urlpath, streamed)
    server.target = None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    urlbase = f"http://127.0.0.1:{server.server_address[1]}/"
    return server, urlbase, f"@public/{pathlib.Path(urlpath).name}"


#
# What each mode would ask for
#


def default_patterns(shape):
    """Slices worth asking about, for an array of any shape (as in the fsspec bench)."""
    mid = [s // 2 for s in shape]
    return [
        ("point", tuple(mid)),
        ("line, last dim", (*mid[:-1], slice(None))),
        ("line, first dim", (slice(None), *mid[1:])),
        (
            "window (1/64 per dim)",
            tuple(slice(m, m + max(1, s // 64)) for m, s in zip(mid, shape, strict=True)),
        ),
        (
            "slab (1% of dim 0)",
            (slice(mid[0], mid[0] + max(1, shape[0] // 100)), *[slice(None)] * (len(shape) - 1)),
        ),
        (
            "slab (10% of dim 0)",
            (slice(mid[0], mid[0] + max(1, shape[0] // 10)), *[slice(None)] * (len(shape) - 1)),
        ),
    ]


def chunk_cbytes(source, nchunks):
    """The compressed size of each chunk, from 16 bytes of its header.

    Read rather than guessed at: the distance to the next chunk is an upper
    bound only, and a frame with a hole in it would make the chunk mode look
    dearer than it is.  One request for the lot where the server takes
    several ranges, which is the same trick the fetch path uses.
    """
    live = [n for n in nchunks if int(source._offsets[n]) >= 0]
    heads = source.read_ranges([(int(source._offsets[n]), 16) for n in live]) if live else []
    sizes = dict.fromkeys(nchunks, 0)  # a run-length chunk has no bytes in the file
    for nchunk, head in zip(live, heads, strict=True):
        sizes[nchunk] = struct.unpack("<i", head[12:16])[0]
    return sizes


def request_plan(proxy, array, item):
    """What each mode asks the server for, to serve *item*.

    Follows `Proxy._fetch_by_block` step for step -- which chunks the slice
    touches, which of those are worth taking apart, one read for the block
    offsets of each, then the coalesced runs of the blocks wanted -- so the
    counts are the ones the fetch below will produce, not an idea of them.
    """
    source = array.block_source()
    wanted = proxy._wanted_blocks(item)
    sizes = chunk_cbytes(source, list(wanted))
    # The whole fetch is what a batching transport weighs, so the same wave the
    # fetch judges by is what is asked here; judging chunk by chunk would report
    # a plan the fetch below does not follow
    wave = {n: len(bs) for n, bs in wanted.items()} if source.max_ranges > 1 else None
    layouts, runs, whole = [], [], []
    nblocks_wanted = 0
    for nchunk, nblocks in wanted.items():
        nblocks = list(nblocks)
        nblocks_wanted += len(nblocks)
        if not sizes[nchunk]:  # a run-length chunk: free in every mode
            continue
        if not array.wants_blocks(nchunk, len(nblocks), wave) or source.chunk_layout(nchunk) is None:
            whole.append(nchunk)  # a chunk not worth taking apart, or with nothing to take apart
            continue
        layouts.append(CHUNK_HEADER + 4 * array.blocks_per_chunk)
        runs += [size for _, size, _ in source.block_plan(nchunk, nblocks)]
    chunk_bytes = sum(sizes.values())
    block_bytes = sum(layouts) + sum(runs) + sum(sizes[n] for n in whole)
    batch = source.max_ranges
    return {
        "chunks touched": len(wanted),
        "blocks wanted": nblocks_wanted,
        "blocks total": len(wanted) * array.blocks_per_chunk,
        "chunk requests": sum(1 for n in wanted if sizes[n]),
        "chunk bytes": chunk_bytes,
        "block requests": len(layouts) + len(runs) + len(whole),
        "multipart requests": (math.ceil(len(layouts) / batch) + math.ceil(len(runs) / batch) + len(whole)),
        "block bytes": block_bytes,
        "ratio": block_bytes / chunk_bytes if chunk_bytes else float("nan"),
    }


#
# What each mode costs
#


def count_traffic(array, source, latency, bandwidth):
    """Tally every HTTP request the fetch makes, and put a network in front of it.

    `C2NDSource._get` is one request whatever it carries, which is the unit the
    modes differ in; `get_chunk` is the other one.  Both waits happen inside the
    proxy's thread pool, so they overlap the way real ones would.
    """
    tally = {"requests": 0, "bytes": 0}

    def charge(nbytes):
        if bandwidth:
            time.sleep(nbytes / bandwidth)
        tally["requests"] += 1
        tally["bytes"] += nbytes

    original_chunk = array.get_chunk

    def get_chunk(nchunk):
        if latency:
            time.sleep(latency)
        chunk = original_chunk(nchunk)
        charge(len(chunk))
        return chunk

    array.get_chunk = get_chunk
    if source is not None:
        original_get = source._get

        def get(spans):
            if latency:
                time.sleep(latency)
            parts = original_get(spans)
            charge(sum(len(part) for part in parts))
            return parts

        source._get = get
    return tally


def timed_slice(open_array, item, mode, concurrency, latency, bandwidth):
    """One cold slice through the shipped code, in the state *mode* names.

    Blocks are turned off by pushing the size threshold out of reach, which is
    exactly the decision `wants_blocks` makes for a small-chunked dataset, so
    ``chunks`` is the shipped code against itself rather than a reimplementation
    of what it used to do.  The frame index is read before the clock starts: it
    costs four small requests once per C2Array, not once per slice.
    """
    threshold = blosc2.proxy_source.BLOCK_MIN_CBYTES
    blosc2.proxy_source.BLOCK_MIN_CBYTES = 1 << 62 if mode == "chunks" else threshold
    try:
        array = open_array()
        array.max_concurrency = concurrency
        source = array.block_source()
        if source is not None:
            source.max_ranges = 1 if mode == "blocks" else c2array.MAX_RANGES_PER_REQUEST
        tally = count_traffic(array, source, latency, bandwidth)
        proxy = blosc2.Proxy(array, mode="w")
        start = time.perf_counter()
        proxy[item]
        return time.perf_counter() - start, tally["requests"], tally["bytes"]
    finally:
        blosc2.proxy_source.BLOCK_MIN_CBYTES = threshold


def fill_chunks(source, limit):
    """The dataset's own compressed chunks, which is what a fill would carry.

    Real chunks rather than synthetic ones, so the bytes on the wire and the
    work the server does storing them are the dataset's own.  Capped, because a
    fill is timed per chunk and a large array would only repeat the measurement.
    """
    nchunks = math.prod(math.ceil(s / c) for s, c in zip(source.shape, source.chunks, strict=True))
    # `get_chunk` is the one both a local array and a `C2Array` answer, so the
    # bytes are the dataset's own whether it is a file here or a dataset there
    return [source.get_chunk(n) for n in range(min(nchunks, limit))]


def timed_fill(open_array, chunks, writers, latency, bandwidth):
    """Write *chunks* into a pre-sized array, and say what it cost.

    One `C2Array` per writer, as separate processes would have.  What overlaps
    is the round trip: the server serializes the writes themselves, since
    each one takes the frame's exclusive lock.
    """
    tally = {"requests": 0, "bytes": 0}
    tally_lock = threading.Lock()
    writers = max(writers, 1)
    # One array per writer, built before the clock starts: opening one is an
    # `api/info` of its own, and a writer opens once however many chunks it goes
    # on to send.  Building them inside the timing would charge every chunk for a
    # round trip no writer actually makes
    arrays = [open_array() for _ in range(writers)]
    work = list(enumerate(chunks))
    shares = [work[index::writers] for index in range(writers)]

    def run(assignment):
        array, share = assignment
        for nchunk, chunk in share:
            if latency:
                time.sleep(latency)
            if bandwidth:
                time.sleep(len(chunk) / bandwidth)
            array.update_chunk(nchunk, chunk)
            with tally_lock:
                tally["requests"] += 1
                tally["bytes"] += len(chunk)

    start = time.perf_counter()
    if writers == 1:
        run((arrays[0], work))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=writers) as pool:
            list(pool.map(run, zip(arrays, shares, strict=True)))
    return time.perf_counter() - start, tally["requests"], tally["bytes"]


def local_write_cost(presize, chunks, reps):
    """What the *server* pays to store a chunk, into an empty slot and over a live one.

    Measured on local files rather than over HTTP: this is the difference the
    write-once rule buys, and a round trip would bury it.  A slot that holds
    nothing is appended to and moves no other chunk; one that holds a chunk is
    written in place, and every byte of payload after it is read and written back
    to close the gap the old chunk left.

    The rewrite has to carry a chunk of a *different* compressed size, or it
    measures the wrong thing: replacing a chunk with bytes of its own length
    leaves nothing to close, and the frame skips the move entirely.  None when
    the dataset has no two chunks that differ in size to do it with.
    """
    middle = len(chunks) // 2
    other = next((c for c in chunks if len(c) != len(chunks[middle])), None)
    empty, live = [], []
    for _ in range(reps):
        path = presize()
        array = blosc2.open(path, mode="a", locking=True)
        for nchunk, chunk in enumerate(chunks):
            start = time.perf_counter()
            array.schunk.update_chunk(nchunk, chunk)
            empty.append(time.perf_counter() - start)
        if other is not None:
            # Every slot holds something now, so this one compacts instead
            start = time.perf_counter()
            array.schunk.update_chunk(middle, other)
            live.append(time.perf_counter() - start)
        del array
    return statistics.median(empty), (statistics.median(live) if live else None)


def connection_setup(urlbase, path, token, reps):
    """What a request costs before any bytes move, pooled against a client each.

    `api/info` rather than a chunk, so what is left in the number is the
    connection and the round trip rather than the transfer.
    """
    import httpx

    url = c2array._server_url(urlbase, f"api/info/{path}")
    headers = c2array._auth_headers(token)
    pooled = c2array._sync_client()

    def median(get):
        times = []
        for _ in range(reps):
            start = time.perf_counter()
            get()
            times.append(time.perf_counter() - start)
        return statistics.median(times)

    return median(lambda: pooled.get(url, headers=headers)), median(
        lambda: httpx.get(url, headers=headers, timeout=c2array.TIMEOUT)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="a remote dataset path with --urlbase, else a local .b2nd")
    parser.add_argument("--urlbase", help="a Caterva2 server; without it, one is stood in")
    parser.add_argument("--username", help="log in to the server as this user")
    parser.add_argument("--password", help="the password to log in with")
    parser.add_argument("--token", help="an authorization cookie, instead of logging in")
    parser.add_argument(
        "--streamed",
        action="store_true",
        help="stand-in only: serve the dataset the way a computed one is served",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="also measure filling a pre-sized array a chunk at a time",
    )
    parser.add_argument(
        "--write-target",
        help="with --urlbase and --write: an empty pre-sized array to fill (else one is laid out)",
    )
    parser.add_argument(
        "--fill-chunks", type=int, default=10, help="how many chunks a timed fill writes (default: 10)"
    )
    parser.add_argument("--concurrency", type=int, default=8, help="parallel requests (default: 8)")
    parser.add_argument("--reps", type=int, default=5, help="timed repetitions (default: 5)")
    parser.add_argument("--max-mb", type=float, default=200, help="skip patterns fetching more than this")
    parser.add_argument(
        "--latency-ms", type=float, default=0, help="round-trip latency to simulate per request"
    )
    parser.add_argument(
        "--bandwidth-mbs", type=float, default=0, help="MB/s to simulate per request (cat2.cloud: ~10)"
    )
    args = parser.parse_args()

    server = None
    if args.urlbase:
        urlbase, path = args.urlbase, args.dataset
        if args.write and not args.write_target:
            parser.error("--write against a server needs --write-target: an empty array to fill")
    else:
        server, urlbase, path = stand_in(args.dataset, args.streamed)
    token = args.token
    if args.username:
        token = c2array.login(args.username, args.password, urlbase)

    scratch = tempfile.mkdtemp(prefix="cat2-fill-") if args.write and server else None
    presize = make_presize(args.dataset, scratch, server) if scratch else None
    try:
        report(args, urlbase, path, token, presize)
    finally:
        if server is not None:
            server.shutdown()
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def make_presize(source_path, scratch, server):
    """Lay out an empty array of the dataset's geometry, ready to be filled.

    A fresh one per call: a slot is written once, so a second timed fill needs a
    second array.  Costs a couple of hundred bytes whatever the geometry -- an
    unwritten chunk lives in the offsets and nowhere else.
    """
    source = blosc2.open(str(source_path))
    counter = itertools.count()

    def presize(serve=False):
        path = str(pathlib.Path(scratch) / f"fill-{next(counter)}.b2nd")
        laid_out = blosc2.uninit(
            source.shape, dtype=source.dtype, chunks=source.chunks, blocks=source.blocks, urlpath=path
        )
        del laid_out  # the server's handle is to be the only one over this file
        if serve:
            if server.target is not None:
                server.target.close()  # its handle is done with; the next array gets its own
            server.target = Cat2Server(path, writable=True)
        return path

    return presize


def report(args, urlbase, path, token, presize=None):
    latency, bandwidth = args.latency_ms / 1e3, args.bandwidth_mbs * 1e6

    def open_array():
        return c2array.C2Array(path, urlbase=urlbase, auth_token=token)

    array = open_array()
    nchunks = math.prod(math.ceil(s / c) for s, c in zip(array.shape, array.chunks, strict=True))
    per_chunk = array.cbytes / nchunks if nchunks else 0
    print(
        f"{path} at {urlbase}\n"
        f"  shape={array.shape} dtype={array.dtype} chunks={array.chunks} blocks={array.blocks}\n"
        f"  {nchunks} chunks, {array.blocks_per_chunk} blocks/chunk, "
        f"{per_chunk / 1e6:.2f} MB per chunk, cratio {array.cratio:.1f}x"
    )

    opening = _opened(array)
    source = array.block_source()
    if source is None:
        found_out = (
            f"{opening['requests']} request found that out, and it is never asked again"
            if opening["requests"]
            else f"nothing was asked.\n  Chunks under the {blosc2.proxy_source.BLOCK_MIN_CBYTES / 1e6:.0f} MB "
            "a round trip costs are not worth taking apart, and api/info\n  tells a computed "
            "dataset from a stored one for free"
        )
        print(
            f"\n  byte ranges: not served -- {found_out}.\n"
            "  A proxy over this fetches whole chunks, exactly as it always did."
        )
        _time_patterns(args, open_array, array, ["chunks"], latency, bandwidth)
        if args.write:
            _fill_section(args, urlbase, token, path, presize, latency, bandwidth)
        return
    source.read_ranges([(0, 16), (64, 16)])  # two spans that cannot merge into one
    print(
        f"  byte ranges: served, multipart: {'yes' if source.max_ranges > 1 else 'no'}"
        f" ({source.max_ranges} spans per request)\n"
        f"  frame index: {opening['requests']} requests, {opening['bytes']} bytes, once per C2Array"
    )

    proxy = blosc2.Proxy(array, mode="w")
    print(
        f"\n  {'pattern':22s} {'chunks':>6s} {'blocks':>13s} {'chunk mode':>20s} "
        f"{'block mode':>20s} {'multipart':>9s}   ratio"
    )
    plans = {}
    for name, item in default_patterns(array.shape):
        plan = plans[name] = request_plan(proxy, array, item)
        print(
            f"  {name:22s} {plan['chunks touched']:6d} "
            f"{plan['blocks wanted']:6d}/{plan['blocks total']:<6d} "
            f"{plan['chunk requests']:5d} req {plan['chunk bytes'] / 1e6:7.2f} MB "
            f"{plan['block requests']:5d} req {plan['block bytes'] / 1e6:7.2f} MB "
            f"{plan['multipart requests']:5d} req {plan['ratio'] * 100:6.1f}%"
        )

    pooled, fresh = connection_setup(urlbase, path, token, args.reps)
    print(
        f"\n  connection setup (api/info): {pooled * 1e3:6.1f} ms pooled, {fresh * 1e3:6.1f} ms with "
        f"a client per request ({fresh / pooled:.1f}x)"
    )
    _time_patterns(args, open_array, array, ["chunks", "blocks", "multipart"], latency, bandwidth, plans)
    if args.write:
        _fill_section(args, urlbase, token, path, presize, latency, bandwidth)


def _fill_section(args, urlbase, token, path, presize, latency, bandwidth):
    """The write path, over the same connection the reads were measured on."""
    local = args.dataset if presize is not None else None
    source = blosc2.open(str(local)) if local else c2array.C2Array(path, urlbase, token)
    _report_fill(args, urlbase, token, source, presize, args.write_target, latency, bandwidth)


def _report_fill(args, urlbase, token, source, presize, target_path, latency, bandwidth):
    """What filling a pre-sized array costs, and what the write-once rule buys."""
    chunks = fill_chunks(source, args.fill_chunks)
    payload = sum(len(chunk) for chunk in chunks)
    print(
        f"\n  fill: {len(chunks)} chunks, {payload / 1e6:.2f} MB of the dataset's own "
        f"compressed bytes\n"
        f"    {'writers':16s} {'requests':>8s} {'bytes':>10s} {'total':>9s} {'per chunk':>11s}"
    )
    runs = [("serial", 1)]
    if args.concurrency > 1:
        runs.append((f"{args.concurrency} at once", args.concurrency))
    serial = None
    filled = filled_path = None
    for label, writers in runs:
        if presize is None and serial is not None:
            # A real target's slots are one-shot, and the bench does not lay out
            # a second array on someone else's server
            print(f"    {'(a second fill needs a second empty array)':16s}")
            break
        path = target_path
        if presize is not None:
            filled_path = presize(serve=True)
            path = f"@public/{pathlib.Path(filled_path).name}"

        def open_array(remote=path):
            return c2array.C2Array(remote, urlbase=urlbase, auth_token=token)

        elapsed, requests, nbytes = timed_fill(open_array, chunks, writers, latency, bandwidth)
        # `is None`, not falsiness: a fill fast enough to measure as 0.0 is a
        # measurement, and taking it for "not measured yet" would make the
        # concurrent run its own baseline and report a speedup of 1.0x
        serial = elapsed if serial is None else serial
        filled = open_array()
        speedup = f"  {serial / elapsed:.1f}x" if writers > 1 else ""
        print(
            f"    {label:16s} {requests:8d} {nbytes / 1e6:9.2f} MB {elapsed:8.3f} s "
            f"{elapsed / len(chunks) * 1e3:9.1f} ms{speedup}"
        )

    if presize is not None:
        empty, live = local_write_cost(lambda: presize(serve=False), chunks, args.reps)
        # What the rewrite has to shift, which is what its cost is made of: the
        # ratio below is this dataset's, and grows with whatever follows a chunk
        tail = sum(len(chunk) for chunk in chunks[len(chunks) // 2 + 1 :])
        rewrite = (
            f"    over a live chunk  {live * 1e3:8.2f} ms   {live / empty:.1f}x here, reading and "
            f"writing back the {tail / 1e6:.2f} MB after it"
            if live is not None
            else "    over a live chunk        n/a   every chunk here compresses to the same size, "
            "which is the case that never moves"
        )
        print(
            f"\n  what the server pays to store one chunk (local, median of {args.reps})\n"
            f"    into an empty slot {empty * 1e3:8.2f} ms   appended past the offsets; "
            f"no other chunk moves\n{rewrite}"
        )

    if filled is not None:
        start = time.perf_counter()
        written = filled.written_chunks()
        remote = time.perf_counter() - start
        print(
            f"\n  reading how far a fill has got ({int(written.sum())}/{written.size} written)\n"
            f"    written_chunks()   {remote * 1e3:8.2f} ms   over HTTP: the frame's header, "
            "then the offsets it locates"
        )
    if filled_path is not None:
        # The same question the server asks itself on every write, both ways
        # round and both local, since one of them is not a thing to ask remotely
        blosc2.FsspecNDSource(filled_path).written_chunks()  # fsspec's first use is its own cost
        start = time.perf_counter()
        offsets = blosc2.FsspecNDSource(filled_path).written_chunks()
        index = time.perf_counter() - start
        array = blosc2.open(filled_path)
        start = time.perf_counter()
        walked = sum(1 for info in array.schunk.iterchunks_info() if info.special.name != "UNINIT")
        walk = time.perf_counter() - start
        print(
            f"    ... the same, local {index * 1e3:8.2f} ms   one decompress of the offsets, "
            f"whatever the count\n"
            f"    iterchunks_info()  {walk * 1e3:8.2f} ms   {walk / offsets.size * 1e6:.1f} us per "
            f"chunk ({walked} written), which is what grows with the array"
        )


def _no_chunks(array):
    """Why api/chunk cannot serve this dataset, if it cannot."""
    import httpx

    try:
        array.get_chunk(0)
    except httpx.HTTPStatusError as exc:
        return f"api/chunk answers {exc.response.status_code} for {array.path}"
    return None


def _opened(array):
    """What reading the frame index cost, which is paid once per C2Array."""
    fresh = c2array.C2Array(array.path, urlbase=array.urlbase, auth_token=array.auth_token)
    tally = {"requests": 0, "bytes": 0}
    original = c2array.C2NDSource.read_range

    def counted(self, offset, size):
        tally["requests"] += 1  # before the read: a refused probe is a request too
        data = original(self, offset, size)
        tally["bytes"] += len(data)
        return data

    c2array.C2NDSource.read_range = counted
    try:
        fresh.block_source()
    finally:
        c2array.C2NDSource.read_range = original
    return tally


def _time_patterns(args, open_array, array, modes, latency, bandwidth, plans=None):
    unavailable = _no_chunks(array)
    if unavailable:
        # A container member (a .b2z or .h5 leaf) is fetchable but not chunk-wise:
        # api/chunk resolves a path without its inner key, so it 404s.  A proxy
        # over one cannot read it at all, whatever mode it would have used
        print(f"\n  nothing to time: {unavailable}")
        return
    if latency or bandwidth:
        print(
            f"\n  simulating a network: {latency * 1e3:.0f} ms per request"
            + (f", {bandwidth / 1e6:.1f} MB/s across it" if bandwidth else "")
        )
    header = "".join(f"{mode:>28s}" for mode in modes)
    print(f"\n  {'pattern':22s}{header}" + ("  vs chunks" if len(modes) > 1 else ""))
    for name, item in default_patterns(array.shape):
        if plans and plans[name]["chunk bytes"] > args.max_mb * 1e6:
            print(f"  {name:22s}   skipped ({plans[name]['chunk bytes'] / 1e6:.0f} MB > --max-mb)")
            continue
        results = {}
        for mode in modes:
            runs = [
                timed_slice(open_array, item, mode, args.concurrency, latency, bandwidth)
                for _ in range(args.reps)
            ]
            results[mode] = (statistics.median(r[0] for r in runs), runs[0][1], runs[0][2])
        line = "".join(
            f"{results[mode][1]:5d} req {results[mode][2] / 1e6:7.2f} MB {results[mode][0]:6.3f}s"
            for mode in modes
        )
        speedup = ""
        if len(modes) > 1:  # what the last mode, which is the shipped one, is worth
            speedup = f"  {results[modes[0]][0] / results[modes[-1]][0]:8.1f}x"
        print(f"  {name:22s}{line}{speedup}", flush=True)


if __name__ == "__main__":
    main()
