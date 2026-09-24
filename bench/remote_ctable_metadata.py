#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Benchmark public RemoteCTable parallel reads with fresh HTTPS sessions.

    python bench/remote_ctable_metadata.py URL --workers 1 2 4 8 --repeats 3

One worker includes metadata reuse; it is not the historical library baseline.
"""

import argparse
import json
import statistics
import threading
import time
from contextlib import contextmanager
from unittest.mock import patch

import blosc2

DEFAULT_URL = "https://f001.backblazeb2.com/file/blosc2/readings.b2z"


def metadata(table):
    return {
        "nbytes": table.nbytes,
        "chunks": table.chunks,
        "blocks": table.blocks,
        "schema": table.schema_dict(),
        "attrs": dict(table.attrs),
    }


@contextmanager
def count_concurrent_reads(filesystem):
    """Observe actual requests without changing planning or buffering."""
    original = filesystem.cat_file
    lock = threading.Lock()
    counts = {"active": 0, "peak": 0}

    def counted(*args, **kwargs):
        with lock:
            counts["active"] += 1
            counts["peak"] = max(counts["peak"], counts["active"])
        try:
            return original(*args, **kwargs)
        finally:
            with lock:
                counts["active"] -= 1

    with patch.object(filesystem, "cat_file", counted):
        yield counts


def measured(table, operation):
    before_bytes, before_requests = table.traffic.nbytes, table.traffic.requests
    started = time.perf_counter()
    result = operation()
    return result, {
        "seconds": time.perf_counter() - started,
        "requests": table.traffic.requests - before_requests,
        "bytes": table.traffic.nbytes - before_bytes,
    }


def run(url, workers, row_workers=None):
    started = time.perf_counter()
    with blosc2.open(url, max_concurrency=workers, storage_options={"skip_instance_cache": True}) as table:
        if not isinstance(table, blosc2.RemoteCTable):
            raise ValueError("URL must identify a remote CTable")
        storage = table._remote_storage()
        opened = {
            "seconds": time.perf_counter() - started,
            "requests": table.traffic.requests,
            "bytes": table.traffic.nbytes,
        }
        with count_concurrent_reads(storage._owner.filesystem) as counts:
            description, inspected = measured(table, lambda: metadata(table))
        metadata_peak = storage._peak_metadata_buffer_bytes
        table.max_concurrency = workers if row_workers is None else row_workers
        with count_concurrent_reads(storage._owner.filesystem) as row_counts:
            rows, cold = measured(table, lambda: list(table[:5]))
        row_peak = storage._peak_row_buffer_bytes
        warm_rows, warm = measured(table, lambda: list(table[:5]))
        assert repr(rows) == repr(warm_rows)
        assert warm["requests"] == warm["bytes"] == 0
        result = {
            "workers": workers,
            "row_workers": table.max_concurrency,
            "open": opened,
            "metadata": inspected,
            "cold_rows": cold,
            "warm_rows": warm,
            "peak_metadata_reads": counts["peak"],
            "peak_row_reads": row_counts["peak"],
            "peak_metadata_buffer_bytes": metadata_peak,
            "peak_row_buffer_bytes": row_peak,
            "cache_bytes": table.cache_bytes,
        }
        result["total_seconds"] = sum(result[s]["seconds"] for s in ("open", "metadata", "cold_rows"))
        return result, (description, repr(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", nargs="?", default=DEFAULT_URL)
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--serial-rows", action="store_true", help="Use one worker for row reads")
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 1 for n in args.workers):
        parser.error("repeats and workers must be positive (1 = production serial control)")
    results, expected = [], None
    for repeat in range(args.repeats):
        order = args.workers if repeat % 2 == 0 else args.workers[::-1]
        for workers in order:
            result, values = run(args.url, workers, row_workers=1 if args.serial_rows else None)
            if expected is None:
                expected = values
            assert values == expected, "Metadata or row values changed between runs"
            result["repeat"] = repeat + 1
            results.append(result)
            print(json.dumps(result), flush=True)
    print("\nMedians (seconds; fresh table and HTTP session per run)")
    print("workers=1: production serial control (includes header reuse)")
    print("workers    open  column metadata  open+metadata  cold rows    total  peak meta/rows")
    for workers in args.workers:
        runs = [r for r in results if r["workers"] == workers]
        med = {
            s: statistics.median(r[s]["seconds"] for r in runs) for s in ("open", "metadata", "cold_rows")
        }
        opening = statistics.median(r["open"]["seconds"] + r["metadata"]["seconds"] for r in runs)
        total = statistics.median(r["total_seconds"] for r in runs)
        peak = max(r["peak_metadata_reads"] for r in runs)
        print(
            f"{workers:7d} {med['open']:7.3f} {med['metadata']:16.3f} {opening:14.3f} "
            f"{med['cold_rows']:10.3f} {total:8.3f} {peak:6d}/{max(r['peak_row_reads'] for r in runs)}"
        )


if __name__ == "__main__":
    main()
