#!/usr/bin/env python

#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Find the right ``max_concurrency`` for lazy reads against a real object store.

``blosc2.open(url, lazy=True)`` fetches one chunk per range request and overlaps
8 of them by default.  That 8 was chosen by argument, not measurement: the cost
of the thread pool was measured (~10 us per chunk, where there is no latency to
hide) but the gain never was, because nothing that runs offline has a round trip
to hide.  This script answers it against a real endpoint.

It also runs ``afetch()``, whose async path (``aget_chunk`` -> ``fs._cat_file``)
has never executed at all: ``memory://`` is not an async backend, so only its
blocking fallback is covered by the test suite.

Usage
-----
    python fsspec-concurrency.py s3://my-bucket/bench.b2nd

    # S3-compatible endpoints (Cloudflare R2, Backblaze B2, MinIO...)
    python fsspec-concurrency.py s3://my-bucket/bench.b2nd \\
        --endpoint-url https://<account>.r2.cloudflarestorage.com

The target array is written on first use, so the bucket must be writable; pass
``--no-upload`` to benchmark one that is already there.  Credentials come from
the driver (``~/.aws/config``, ``AWS_*`` environment variables, ...), never from
blosc2.

``--endpoint-url`` and ``--anon`` go through ``fsspec.config.conf``, which is
fsspec's own per-protocol default-argument mechanism, because ``blosc2.open()``
has no ``storage_options=`` passthrough of its own.  The equivalent without this
script is a ``~/.config/fsspec/s3.json`` holding ``{"anon": true}``, or the
``FSSPEC_S3_ANON`` / ``FSSPEC_S3_ENDPOINT_URL`` environment variables.

Reading the output
------------------
Wall time should fall roughly as 1/concurrency while requests are the
bottleneck, then flatten once the endpoint, the connection pool, or the local
CPU becomes one.  The knee is the answer.  Times going *up* again, or errors
appearing, means the endpoint is throttling (S3 answers 503 SlowDown): back off
to the last value that scaled.
"""

import argparse
import asyncio
import time

import numpy as np

import blosc2


def configure(protocol, **options):
    """Set fsspec's default arguments for *protocol*, since we cannot pass them."""
    options = {k: v for k, v in options.items() if v}
    if options:
        import fsspec.config

        fsspec.config.conf.setdefault(protocol, {}).update(options)
        print(f"fsspec {protocol} options: {options}")


def build_target(urlpath, nchunks, chunklen, upload):
    """Put an array of a known shape at *urlpath*, unless it is there already."""
    import fsspec

    fs, path = fsspec.url_to_fs(urlpath)
    if fs.exists(path):
        if not upload:
            return
        print(f"overwriting {urlpath}")
    elif not upload:
        raise SystemExit(f"{urlpath} does not exist and --no-upload was passed")

    a = blosc2.arange(0, nchunks * chunklen, dtype=np.int32, chunks=(chunklen,))
    t0 = time.perf_counter()
    a.save(urlpath)
    print(f"uploaded {a.schunk.nchunks} chunks in {time.perf_counter() - t0:.1f} s")


def timed(urlpath, item, max_concurrency, use_afetch=False):
    """Time one cold read: a fresh proxy every time, so nothing is cached."""
    a = blosc2.open(urlpath, lazy=True, max_concurrency=max_concurrency)
    t0 = time.perf_counter()
    if use_afetch:
        asyncio.run(a.afetch(item, max_concurrency=max_concurrency))
    else:
        a[item]
    return time.perf_counter() - t0


def report(label, times, nchunks_touched):
    print(f"\n{label}  ({nchunks_touched} chunks)")
    print(f"  {'concurrency':>11}  {'wall':>8}  {'per chunk':>10}  {'vs serial':>9}")
    serial = times.get(1)
    for concurrency, elapsed in sorted(times.items()):
        speedup = f"{serial / elapsed:.1f}x" if serial else "-"
        print(
            f"  {concurrency:>11}  {elapsed:>7.2f}s  {elapsed / nchunks_touched * 1e3:>9.1f}ms  {speedup:>9}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("urlpath", help="fsspec URL of the benchmark array, e.g. s3://bucket/bench.b2nd")
    p.add_argument("--endpoint-url", help="for S3-compatible endpoints (R2, B2, MinIO...)")
    p.add_argument("--anon", action="store_true", help="anonymous access (public read-only buckets)")
    p.add_argument("--nchunks", type=int, default=200, help="chunks in the uploaded array")
    p.add_argument("--chunklen", type=int, default=250_000, help="int32 items per chunk (1 MB each)")
    p.add_argument("--no-upload", dest="upload", action="store_false", help="use the array as it is")
    p.add_argument(
        "--concurrency",
        default="1,2,4,8,16,32",
        help="comma-separated values to try (default: 1,2,4,8,16,32)",
    )
    p.add_argument("--skip-afetch", action="store_true", help="do not exercise the async path")
    args = p.parse_args()

    protocol = args.urlpath.split("://", 1)[0]
    configure(protocol, endpoint_url=args.endpoint_url, anon=args.anon)
    build_target(args.urlpath, args.nchunks, args.chunklen, args.upload)
    levels = [int(x) for x in args.concurrency.split(",")]

    # One chunk, serially: the round trip everything else is made of
    one = timed(args.urlpath, slice(0, args.chunklen), 1)
    print(f"\nsingle chunk (one round trip): {one * 1e3:.0f} ms")

    slice_chunks = min(16, args.nchunks)
    item = slice(0, slice_chunks * args.chunklen)
    report("slice", {c: timed(args.urlpath, item, c) for c in levels}, slice_chunks)
    report(
        "whole array",
        {c: timed(args.urlpath, slice(None), c) for c in levels},
        args.nchunks,
    )

    if not args.skip_afetch:
        # The async path, which no test has ever run: aget_chunk reaches
        # fs._cat_file directly rather than falling back to the blocking read
        report(
            "slice, afetch (async path)",
            {c: timed(args.urlpath, item, c, use_afetch=True) for c in levels},
            slice_chunks,
        )


if __name__ == "__main__":
    main()
