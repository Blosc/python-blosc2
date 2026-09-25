"""Measure lazy reads of local Parquet files directly or through localhost HTTP."""

import argparse
import http.server
import json
import math
import threading
import time
from contextlib import contextmanager, nullcontext
from email.utils import formatdate
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

import blosc2


@contextmanager
def local_http_server(path):
    counts = [0, 0, 0]  # GET requests, GET body bytes, HEAD requests.
    lock = threading.Lock()
    size = path.stat().st_size

    class Ranged(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def headers_for_file(self, length):
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Last-Modified", formatdate(path.stat().st_mtime, usegmt=True))

        def do_HEAD(self):
            with lock:
                counts[2] += 1
            self.send_response(200)
            self.headers_for_file(size)
            self.end_headers()

        def do_GET(self):
            span = self.headers.get("Range")
            first, last = 0, size - 1
            if span:
                start, _, end = span.removeprefix("bytes=").partition("-")
                if start:
                    first = int(start)
                    last = min(int(end), size - 1) if end else size - 1
                else:
                    first = max(0, size - int(end))
            if first > last:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206 if span else 200)
            self.headers_for_file(last - first + 1)
            if span:
                self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.end_headers()
            sent = 0
            with path.open("rb") as source:
                source.seek(first)
                remaining = last - first + 1
                while remaining:
                    data = source.read(min(1 << 20, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    sent += len(data)
                    remaining -= len(data)
            with lock:
                counts[0] += 1
                counts[1] += sent

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Ranged)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:

        def snapshot():
            with lock:
                return tuple(counts)

        yield f"http://127.0.0.1:{server.server_port}/{path.name}", snapshot
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def measure(path, column=None, *, open_only=False, http=False):
    path = Path(path)
    context = local_http_server(path) if http else nullcontext((path, None))
    with context as (source, server_snapshot):
        started = time.perf_counter()
        options = {"block_size": 4096, "cache_type": "none"} if http else None
        with blosc2.open(source, storage_options=options) as remote:
            snapshot = server_snapshot or (lambda: (remote.traffic.requests, remote.traffic.nbytes, 0))
            opened = snapshot()
            open_seconds = time.perf_counter() - started
            column = column or remote.col_names[0]
            if open_only:
                return {
                    "file": path.name,
                    "file_bytes": path.stat().st_size,
                    "rows": len(remote),
                    "column": column,
                    "transport": "http" if http else "file",
                    "open_reads": opened[0],
                    "open_bytes": opened[1],
                    "open_heads": opened[2],
                    "open_seconds": round(open_seconds, 3),
                }
            started = time.perf_counter()
            cold_value = remote[column][-1]
            cold_seconds = time.perf_counter() - started
            after_cold = snapshot()
            cold = tuple(a - b for a, b in zip(after_cold, opened, strict=True))
            started = time.perf_counter()
            warm_value = remote[column][-1]
            warm_seconds = time.perf_counter() - started
            warm = tuple(a - b for a, b in zip(snapshot(), after_cold, strict=True))
            result = {
                "file": path.name,
                "file_bytes": path.stat().st_size,
                "rows": len(remote),
                "column": column,
                "transport": "http" if http else "file",
                "open_reads": opened[0],
                "open_bytes": opened[1],
                "open_heads": opened[2],
                "open_seconds": round(open_seconds, 3),
                "cold_reads": cold[0],
                "cold_bytes": cold[1],
                "cold_heads": cold[2],
                "cold_seconds": round(cold_seconds, 3),
                "warm_reads": warm[0],
                "warm_bytes": warm[1],
                "warm_heads": warm[2],
                "warm_seconds": round(warm_seconds, 3),
                "cache_bytes": remote.cache_bytes,
                "warm_equal": bool(
                    cold_value == warm_value
                    or (isinstance(cold_value, float) and math.isnan(cold_value) and math.isnan(warm_value))
                ),
            }
    parquet = pq.ParquetFile(path)
    result["groups"] = parquet.num_row_groups
    if column in parquet.schema_arrow.names:
        last = parquet.read_row_group(parquet.num_row_groups - 1, columns=[column])
        expected = last.column(0)[-1].as_py()
    elif parquet.schema_arrow.names == [""]:
        root = parquet.read_row_group(parquet.num_row_groups - 1, columns=[""]).column(0).combine_chunks()
        leaf = pc.list_flatten(root)
        for part in column.split("."):
            leaf = leaf.field(part)
        expected = leaf[-1].as_py()
    else:
        expected = None
    if expected is not None:
        result["arrow_equal"] = bool(
            cold_value == expected
            or (isinstance(cold_value, float) and math.isnan(cold_value) and math.isnan(expected))
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--column")
    parser.add_argument("--open-only", action="store_true")
    parser.add_argument(
        "--http", action="store_true", help="serve each file over localhost HTTP with byte ranges"
    )
    args = parser.parse_args()
    failed = False
    for parquet_path in args.paths:
        try:
            print(
                json.dumps(
                    measure(parquet_path, args.column, open_only=args.open_only, http=args.http), default=str
                ),
                flush=True,
            )
        except Exception as error:
            failed = True
            print(
                json.dumps({"file": parquet_path.name, "error": f"{type(error).__name__}: {error}"}),
                flush=True,
            )
    if failed:
        raise SystemExit(1)
