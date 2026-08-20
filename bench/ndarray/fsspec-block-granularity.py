#!/usr/bin/env python

#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Would block-granular downloads beat chunk-granular ones for a given array?

``blosc2.open(url, lazy=True)`` fetches one whole compressed chunk per range
request.  A chunk is made of blocks, which blosc2 compresses and decompresses
independently, so a slice could in principle fetch only the blocks it touches.
Whether that is worth the extra round trip it costs (the block offsets live in
the chunk header, which has to be read first) depends on two numbers this script
measures:

- the **touch ratio**: what fraction of the bytes of the chunks a slice touches
  its blocks actually account for.  Exact, computed locally from the array's own
  chunk headers, no network involved;
- the **wall time** of the two request patterns against a real object store.

Usage
-----
    # touch ratios only, on any local array
    python fsspec-block-granularity.py mydata.b2nd

    # ... and time real slices against a local S3 server (pip install "moto[server]")
    python fsspec-block-granularity.py mydata.b2nd --moto

    # ... and time both request patterns against real S3
    python fsspec-block-granularity.py mydata.b2nd \\
        --replay s3://noaa-goes16/ABI-L1b-RadF/2020/001/00/OR_ABI-L1b-RadF-M6C02_G16_s20200010000216_e20200010009524_c20200010009570.nc --anon

``--moto`` is the end-to-end one: it uploads the array and slices it back through
``blosc2.open(url, lazy=True)`` twice, once with block fetching and once with it
turned off, so what is timed is the shipped code on a real HTTP object store.
Its request and byte counts are exact.  Its *times* are not S3's: moto answers
over loopback in a millisecond and at ~700 MB/s, where an object store takes tens
to hundreds of milliseconds at a few MB/s per stream, which is the regime the
whole trade lives in.  ``--latency-ms`` and ``--bandwidth-mbs`` put a stated
network back in front of each request; measured against S3 from Europe, that is
about ``--latency-ms 240 --bandwidth-mbs 3.5``, and in-region more like
``--latency-ms 15 --bandwidth-mbs 90``.

The replay target is *any* object at least as large as the biggest request; its
contents are never used.  What is being timed is the request shape — how many
ranges, of what sizes, in how many dependent phases — which is what separates
the two designs.  Using a public object means the measurement needs no bucket of
its own, and the client stack (s3fs, aiobotocore, HTTPS, real latency) is the
one blosc2 would use.

Three modes are timed:

- ``chunk``: one range per touched chunk, ``max_concurrency`` at a time. What
  ``lazy=True`` does today.
- ``blocks``: one range per touched chunk for the header and block offsets,
  then one range per (coalesced) run of wanted blocks. Two dependent phases.
- ``blocks, cached``: the same without the header phase, which is what a second
  slice of the same array costs once the offsets have been read once.
"""

import argparse
import itertools
import math
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import blosc2

GAP = 4096  # merge ranges separated by less than this into one request


def chunk_layout(schunk, nchunk, cache):
    """(cbytes, bstarts, extents) of a chunk, as a byte-range reader would see it.

    ``bstarts`` is *not* sorted -- a multithreaded compressor writes blocks in
    completion order -- so a block's extent is the distance to the next larger
    offset, not to its neighbour in the array.  The extents are computed that way
    here, rather than read from the lazy chunk's trailer, because that is all a
    byte-range reader over the network can do: it is an upper bound where a chunk
    has holes, which is what such a reader would fetch.
    """
    if nchunk in cache:
        return cache[nchunk]
    # A lazy chunk is header + bstarts + trailer, so this reads a few hundred
    # bytes per chunk instead of the whole array
    chunk = schunk.get_lazychunk(nchunk)
    nbytes, cbytes, blocksize = blosc2.get_cbuffer_sizes(chunk)
    nblocks = (nbytes + blocksize - 1) // blocksize
    if (chunk[31] >> 4) & 0x7:  # run-length chunk: no bytes in the file at all
        res = (0, np.empty(0, np.int64), np.empty(0, np.int64))
    else:
        if chunk[2] & 0x02:  # memcpyed: raw blocks, no bstarts section
            bstarts = 32 + np.arange(nblocks, dtype=np.int64) * blocksize
            extents = np.full(nblocks, blocksize, dtype=np.int64)
            extents[-1] = nbytes - (nblocks - 1) * blocksize
        else:
            if len(chunk) < 32 + 4 * nblocks:  # an in-memory array: no lazy chunks
                chunk = schunk.get_chunk(nchunk)
            bstarts = np.frombuffer(chunk[32 : 32 + 4 * nblocks], dtype="<i4").astype(np.int64)
            bounds = np.sort(np.append(bstarts, cbytes))
            extents = bounds[np.searchsorted(bounds, bstarts, "right")] - bstarts
        res = (cbytes, bstarts, extents)
    cache[nchunk] = res
    return res


def coalesce(ranges):
    """Sizes of the requests *ranges* becomes once near-adjacent ones are merged."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    sizes, start, end = [], ranges[0][0], ranges[0][0] + ranges[0][1]
    for offset, size in ranges[1:]:
        if offset <= end + GAP:
            end = max(end, offset + size)
        else:
            sizes.append(end - start)
            start, end = offset, offset + size
    sizes.append(end - start)
    return sizes


def touched(shape, chunks, blocks, item):
    """{nchunk: [nblock]} for the slice *item*."""
    ndim = len(shape)
    item = tuple(item) + (slice(None),) * (ndim - len(item))
    spans = []
    for dim, index in enumerate(item):
        if isinstance(index, slice):
            start, stop, _ = index.indices(shape[dim])
        else:
            start = index if index >= 0 else index + shape[dim]
            stop = start + 1
        spans.append((start, stop))
    chunk_grid = [math.ceil(s / c) for s, c in zip(shape, chunks, strict=True)]
    blocks_in_chunk = [math.ceil(c / b) for c, b in zip(chunks, blocks, strict=True)]
    out = {}
    ranges = [range(s // chunks[d], (e - 1) // chunks[d] + 1) for d, (s, e) in enumerate(spans)]
    for coords in itertools.product(*ranges):
        nchunk = int(np.ravel_multi_index(coords, chunk_grid))
        per_dim = []
        for dim in range(ndim):
            lo = max(spans[dim][0] - coords[dim] * chunks[dim], 0)
            hi = min(spans[dim][1] - coords[dim] * chunks[dim], chunks[dim])
            per_dim.append(range(lo // blocks[dim], (hi - 1) // blocks[dim] + 1))
        out[nchunk] = [int(np.ravel_multi_index(b, blocks_in_chunk)) for b in itertools.product(*per_dim)]
    return out


def request_plan(array, item):
    """The requests each mode would issue for *item*: (chunk sizes, header sizes, block sizes)."""
    schunk, cache = array.schunk, {}
    chunk_sizes, header_sizes, block_sizes = [], [], []
    nblocks_touched = 0
    for nchunk, nblocks in touched(array.shape, array.chunks, array.blocks, item).items():
        cbytes, bstarts, extents = chunk_layout(schunk, nchunk, cache)
        if not cbytes:  # special chunk: free in both modes
            continue
        chunk_sizes.append(int(cbytes))
        header_sizes.append(32 + 4 * len(bstarts))
        nblocks_touched += len(nblocks)
        block_sizes += coalesce([(int(bstarts[i]), int(extents[i])) for i in nblocks])
    return chunk_sizes, header_sizes, block_sizes, nblocks_touched


def default_patterns(shape):
    """Slices worth asking about, for an array of any shape."""
    mid = [s // 2 for s in shape]
    point = tuple(mid)
    line_last = (*mid[:-1], slice(None))
    line_first = (slice(None), *mid[1:])
    window = tuple(slice(m, m + max(1, s // 64)) for m, s in zip(mid, shape, strict=True))
    slab = (slice(mid[0], mid[0] + max(1, shape[0] // 100)), *[slice(None)] * (len(shape) - 1))
    big_slab = (slice(mid[0], mid[0] + max(1, shape[0] // 10)), *[slice(None)] * (len(shape) - 1))
    return [
        ("point", point),
        ("line, last dim", line_last),
        ("line, first dim", line_first),
        ("window (1/64 per dim)", window),
        ("slab (1% of dim 0)", slab),
        ("slab (10% of dim 0)", big_slab),
    ]


def moto_endpoint():
    """A local S3 server, and fsspec pointed at it. Needs ``pip install moto[server]``."""
    import logging

    import fsspec
    from moto.server import ThreadedMotoServer

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    fsspec.config.conf["s3"] = {
        "endpoint_url": f"http://{host}:{port}",
        "key": "testing",
        "secret": "testing",
        # Not us-east-1: creating a bucket there must carry no location constraint
        "client_kwargs": {"region_name": "eu-west-1"},
    }
    fsspec.filesystem("s3", **fsspec.config.conf["s3"]).mkdir("blosc2-bench")
    return server, "s3://blosc2-bench/bench.b2nd"


def timed_open(urlpath, item, concurrency, blocks, latency=0.0, bandwidth=0.0):
    """Time one cold slice, counting what crossed the wire.

    Blocks are turned off by pushing the size threshold out of reach, which is
    exactly the decision `wants_blocks` makes, so the comparison is the shipped
    code against itself rather than against a reimplementation of the old path.

    A request the source makes is exactly one range GET, so that is where the
    network moto does not have is put back: *latency* seconds before it, and
    *bandwidth* bytes per second across it.  Both waits happen inside the
    proxy's thread pool, so they overlap the way real ones would.
    """
    old = blosc2.proxy_source.BLOCK_MIN_CBYTES
    blosc2.proxy_source.BLOCK_MIN_CBYTES = old if blocks else 1 << 62
    try:
        array = blosc2.open(urlpath, lazy=True, max_concurrency=concurrency)
        traffic = [0, 0]
        for name in ("read_range", "get_chunk"):
            original = getattr(array.src, name)

            def counted(*args, _o=original):
                if latency:
                    time.sleep(latency)
                data = _o(*args)
                if bandwidth:
                    time.sleep(len(data) / bandwidth)
                traffic[0] += 1
                traffic[1] += len(data)
                return data

            setattr(array.src, name, counted)
        t0 = time.perf_counter()
        array[item]
        return time.perf_counter() - t0, traffic[0], traffic[1]
    finally:
        blosc2.proxy_source.BLOCK_MIN_CBYTES = old


def run_moto(local, concurrency, reps, latency, bandwidth):
    """Slice a real array out of a real S3 server, with and without blocks."""
    server, urlpath = moto_endpoint()
    try:
        array = blosc2.open(local)
        print(f"\nuploading {array.schunk.cbytes / 1e6:.1f} MB to {urlpath} ...", flush=True)
        array.save(urlpath, mode="w")
        if latency or bandwidth:
            print(
                f"simulating a network: {latency * 1e3:.0f} ms per request"
                + (f", {bandwidth / 1e6:.1f} MB/s across it" if bandwidth else "")
            )
        print(f"{'pattern':22s} {'chunk mode':>26s} {'block mode':>26s}   speedup")
        for name, item in default_patterns(array.shape):
            runs = {
                mode: [
                    timed_open(urlpath, item, concurrency, mode == "blocks", latency, bandwidth)
                    for _ in range(reps)
                ]
                for mode in ("chunks", "blocks")
            }
            out = {}
            for mode, results in runs.items():
                out[mode] = (statistics.median(r[0] for r in results), results[0][1], results[0][2])
            print(
                f"{name:22s} "
                f"{out['chunks'][1]:4d} req {out['chunks'][2] / 1e6:7.2f} MB {out['chunks'][0]:6.3f}s "
                f"{out['blocks'][1]:5d} req {out['blocks'][2] / 1e6:7.2f} MB {out['blocks'][0]:6.3f}s "
                f"{out['chunks'][0] / out['blocks'][0]:6.1f}x"
            )
    finally:
        server.stop()


class Replayer:
    """Issues the request pattern of a plan against a real object store."""

    def __init__(self, urlpath, concurrency, anon=False, endpoint_url=None):
        import fsspec

        options = {k: v for k, v in {"anon": anon, "endpoint_url": endpoint_url}.items() if v}
        if options:
            fsspec.config.conf.setdefault(urlpath.split("://", 1)[0], {}).update(options)
        self.fs, self.path = fsspec.url_to_fs(urlpath)
        self.size = self.fs.info(self.path)["size"]
        self.concurrency = concurrency
        self.random = random.Random(7)

    def _one(self, size):
        # A fresh offset every time, so nothing is served from a cache anywhere
        size = min(size, self.size)
        offset = self.random.randrange(0, self.size - size + 1)
        return len(self.fs.cat_file(self.path, start=offset, end=offset + size))

    def phase(self, sizes):
        """One wave of parallel range reads, as Proxy.fetch issues them."""
        if not sizes:
            return
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(sizes))) as pool:
            list(pool.map(self._one, sizes))

    def time(self, phases):
        t0 = time.perf_counter()
        for sizes in phases:
            self.phase(sizes)
        return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("urlpath", help="a local .b2nd array to take the geometry from")
    p.add_argument("--replay", help="URL of any large object to replay the request pattern against")
    p.add_argument(
        "--moto", action="store_true", help="upload to a local moto S3 server and time real slices"
    )
    p.add_argument(
        "--latency-ms",
        type=float,
        default=0,
        help="with --moto: round-trip latency to simulate per request (moto has none)",
    )
    p.add_argument(
        "--bandwidth-mbs",
        type=float,
        default=0,
        help="with --moto: MB/s to simulate per request (loopback has ~700, S3 has 3-10)",
    )
    p.add_argument("--anon", action="store_true", help="anonymous access to the replay target")
    p.add_argument("--endpoint-url", help="for S3-compatible endpoints (R2, B2, MinIO...)")
    p.add_argument("--concurrency", type=int, default=8, help="parallel requests (default: 8)")
    p.add_argument("--reps", type=int, default=5, help="timed repetitions (default: 5)")
    p.add_argument("--max-mb", type=float, default=45, help="skip patterns fetching more than this")
    args = p.parse_args()

    array = blosc2.open(args.urlpath)
    blocks_per_chunk = math.prod([math.ceil(c / b) for c, b in zip(array.chunks, array.blocks, strict=True)])
    print(
        f"{args.urlpath}: shape={array.shape} dtype={array.dtype} chunks={array.chunks} "
        f"blocks={array.blocks}\n  {array.schunk.nchunks} chunks, {blocks_per_chunk} blocks/chunk, "
        f"cratio {array.schunk.cratio:.1f}x"
    )

    plans = []
    print(
        f"\n  {'pattern':22s} {'chunks':>6s} {'blocks':>13s} {'chunk mode':>18s} {'block mode':>18s}  ratio"
    )
    for name, item in default_patterns(array.shape):
        chunk_sizes, header_sizes, block_sizes, nblocks = request_plan(array, item)
        chunk_bytes, block_bytes = sum(chunk_sizes), sum(header_sizes) + sum(block_sizes)
        ratio = block_bytes / chunk_bytes if chunk_bytes else float("nan")
        print(
            f"  {name:22s} {len(chunk_sizes):6d} {nblocks:6d}/{len(chunk_sizes) * blocks_per_chunk:<6d} "
            f"{len(chunk_sizes):5d} req {chunk_bytes / 1e6:7.2f} MB "
            f"{len(header_sizes) + len(block_sizes):5d} req {block_bytes / 1e6:7.2f} MB {ratio * 100:6.1f}%"
        )
        plans.append((name, chunk_sizes, header_sizes, block_sizes, chunk_bytes, block_bytes))

    if args.moto:
        run_moto(args.urlpath, args.concurrency, args.reps, args.latency_ms / 1e3, args.bandwidth_mbs * 1e6)

    if not args.replay:
        return

    replayer = Replayer(args.replay, args.concurrency, args.anon, args.endpoint_url)
    print(
        f"\nreplaying against {args.replay} ({replayer.size / 1e6:.0f} MB), "
        f"concurrency {args.concurrency}, {args.reps} reps"
    )
    times = {name: {"chunk": [], "blocks": [], "cached": []} for name, *_ in plans}
    for rep in range(args.reps):
        for name, chunk_sizes, header_sizes, block_sizes, chunk_bytes, _ in plans:
            if chunk_bytes > args.max_mb * 1e6:
                continue
            times[name]["chunk"].append(replayer.time([chunk_sizes]))
            times[name]["blocks"].append(replayer.time([header_sizes, block_sizes]))
            times[name]["cached"].append(replayer.time([block_sizes]))
        print(f"  rep {rep + 1}/{args.reps}", flush=True)

    print(f"\n  {'pattern':22s} {'chunk mode':>16s} {'blocks':>17s} {'blocks, cached':>17s}")
    for name, _sizes, _, _, chunk_bytes, block_bytes in plans:
        if not times[name]["chunk"]:
            print(f"  {name:22s}   skipped ({chunk_bytes / 1e6:.0f} MB > --max-mb)")
            continue
        median = {k: statistics.median(v) for k, v in times[name].items()}
        print(
            f"  {name:22s} {chunk_bytes / 1e6:6.2f}MB {median['chunk']:5.2f}s "
            f"{block_bytes / 1e6:6.2f}MB {median['blocks']:5.2f}s {median['chunk'] / median['blocks']:4.1f}x "
            f"{median['cached']:11.2f}s {median['chunk'] / median['cached']:4.1f}x"
        )


if __name__ == "__main__":
    main()
