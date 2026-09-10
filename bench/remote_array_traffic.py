"""Measure real remote cold/warm reads; run in the blosc2 conda environment.

python bench/remote_array_traffic.py > bench/remote_array_traffic.jsonl

Each trial runs in a fresh process. Transport bytes are aiohttp response bodies
(including metadata/errors), excluding headers and TLS/TCP overhead. Connections
are successful aiohttp connection creations, distinct from HTTP requests.
Uses the existing read-only Backblaze fixtures and the runtime blosc2 S3 profile.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import aiohttp
import numpy as np

import blosc2


def trial(backend, transport):
    counts = Counter()
    send = aiohttp.ClientRequest.send
    connect = aiohttp.TCPConnector._create_connection
    feed = aiohttp.StreamReader.feed_data

    async def counted_send(request, connection):
        counts[request.method] += 1
        return await send(request, connection)

    async def counted_connect(connector, *args, **kwargs):
        result = await connect(connector, *args, **kwargs)
        counts["connections"] += 1
        return result

    def counted_feed(reader, data, *args, **kwargs):
        counts["body_bytes"] += len(data)
        return feed(reader, data, *args, **kwargs)

    root = "s3://blosc2" if transport == "s3" else "https://f001.backblazeb2.com/file/blosc2"
    url = f"{root}/hierarchy.{backend}"
    options = {}
    if transport == "s3":
        options = {"profile": "blosc2", "endpoint_url": "https://s3.us-west-001.backblazeb2.com"}
    kwargs = {"dataset": "d0/a3"} if backend != "zarr" else {}
    if backend == "zarr":
        url += "/d0/a3"
    phases = {}

    def measure(name, operation, array=None):
        before = counts.copy()
        traffic_before = (array.traffic.requests, array.traffic.nbytes) if array is not None else (0, 0)
        start = perf_counter()
        result = operation()
        elapsed = perf_counter() - start
        delta = counts - before
        current = result if array is None else array
        phases[name] = {
            "seconds": elapsed,
            "requests": sum(delta[method] for method in ("GET", "HEAD", "POST", "PUT", "DELETE")),
            "methods": {method: delta[method] for method in ("GET", "HEAD") if delta[method]},
            "connections": delta["connections"],
            "body_bytes": delta["body_bytes"],
            "traffic_requests": current.traffic.requests - traffic_before[0],
            "traffic_bytes": current.traffic.nbytes - traffic_before[1],
        }
        return result

    with (
        patch.object(aiohttp.ClientRequest, "send", counted_send),
        patch.object(aiohttp.TCPConnector, "_create_connection", counted_connect),
        patch.object(aiohttp.StreamReader, "feed_data", counted_feed),
    ):
        array = measure(
            "open",
            lambda: blosc2.open(
                url,
                lazy=True,
                cache_policy=blosc2.CachePolicy.MEMORY,
                max_cache_bytes=64 << 20,
                storage_options=options,
                **kwargs,
            ),
        )
        cold = measure("cold_slice", lambda: array[0, :100, :100], array)
        warm = measure("warm_slice", lambda: array[0, :100, :100], array)
        expected = np.arange(100)[:, None] * 1000 + np.arange(100)
        np.testing.assert_array_equal(cold, expected)
        np.testing.assert_array_equal(warm, cold)
        assert phases["cold_slice"]["body_bytes"] > 0
        assert phases["warm_slice"]["requests"] == phases["warm_slice"]["body_bytes"] == 0
        assert array.cache_bytes <= 64 << 20
    return {
        "backend": backend,
        "transport": transport,
        "url": url,
        "dataset": "d0/a3",
        "shape": array.shape,
        "dtype": str(array.dtype),
        "chunks": array.chunks,
        "blocks": array.blocks,
        "slice": "[0, :100, :100]",
        "result_bytes": cold.nbytes,
        "cache_bytes": array.cache_bytes,
        "blosc2_version": blosc2.__version__,
        "phases": phases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, metavar=("BACKEND", "TRANSPORT"))
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.case:
        print(json.dumps(trial(*args.case)), flush=True)
        return
    for repetition in range(args.repeats):
        for backend in ("b2z", "zarr", "h5"):
            for transport in ("s3", "https"):
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--case", backend, transport],
                    check=True,
                    stdout=subprocess.PIPE,
                    text=True,
                    timeout=240,
                )
                row = json.loads(result.stdout)
                row["repetition"] = repetition + 1
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
