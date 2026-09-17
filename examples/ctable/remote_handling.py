#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Create or access a fixed-width, mask-nullable remote CTable."""

import argparse
import pprint
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2

DEFAULT_PROFILE = "blosc2"
DEFAULT_ENDPOINT_URL = "https://s3.us-west-001.backblazeb2.com"


@dataclass
class Reading:
    id: int = blosc2.field(blosc2.int64())
    station_id: int = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float32(null_storage="mask"))
    humidity: int = blosc2.field(blosc2.int16(null_storage="mask"))
    status: str = blosc2.field(blosc2.string(max_length=8, null_storage="mask"))
    active: bool = blosc2.field(blosc2.bool())


def write_table(args) -> None:
    output = args.write
    if output.suffix != ".b2z":
        raise ValueError("output must end in .b2z")
    if args.rows < 1 or args.batch_size < 1:
        raise ValueError("--rows and --batch-size must be positive")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    statuses = np.array(["ok", "warning", "offline", ""], dtype=object)
    attrs = {
        "version": 1,
        "sampling_interval": 0.5,
        "description": "Synthetic weather-station readings",
        "example_station_ids": [0, 1, 2, 3],
    }

    with blosc2.CTable(
        Reading,
        urlpath=str(output),
        mode="w",
        expected_size=args.rows,
        validate=False,
        create_summary_index=False,
    ) as table:
        for name, value in attrs.items():
            table.attrs[name] = value
        for start in range(0, args.rows, args.batch_size):
            stop = min(start + args.batch_size, args.rows)
            ids = np.arange(start, stop, dtype=np.int64)
            temperature = rng.normal(18, 10, len(ids)).astype(np.float32).astype(object)
            humidity = rng.integers(0, 101, len(ids), dtype=np.int16).astype(object)
            status = statuses[(ids // 7) % len(statuses)].copy()
            temperature[ids % 17 == 0] = None
            humidity[ids % 29 == 0] = None
            status[ids % 41 == 0] = None
            table.extend(
                {
                    "id": ids,
                    "station_id": (ids % 100).astype(np.int32),
                    "temperature": temperature,
                    "humidity": humidity,
                    "status": status,
                    "active": ids % 5 != 0,
                },
                validate=False,
            )

    with blosc2.CTable.open(str(output)) as table:
        assert len(table) == args.rows
        assert table.attrs[:] == attrs
        null_counts = {name: table[name].null_count() for name in ("temperature", "humidity", "status")}

    print(f"Created {output} ({output.stat().st_size / 1_000_000:.1f} MB, {args.rows:,} rows)")
    print(f"Mask-backed null counts: {null_counts}")
    print(f"Now upload {output} to your cloud object storage.")


def access_table(args) -> None:
    storage_options = {}
    if args.url.startswith("s3://"):
        storage_options = {
            "profile": args.profile,
            "client_kwargs": {"endpoint_url": args.endpoint_url},
        }

    print(f"Accessing: {args.url}")
    started = time.perf_counter()
    with blosc2.RemoteCTable(args.url, storage_options=storage_options) as table:
        nbytes = table.nbytes
        metadata = {
            "type": type(table).__name__,
            "source": table.source,
            "rows": table.nrows,
            "columns": table.col_names,
            "chunks": table.chunks,
            "blocks": table.blocks,
            "nbytes": f"{nbytes} ({nbytes / 1024**2:.2f} MiB)",
            "cache_policy": table.cache_policy.name,
            "cache_bytes": f"{table.cache_bytes} ({table.cache_bytes / 1024:.2f} KiB)",
            "schema": table.schema_dict(),
            "attrs": dict(table.attrs),
        }
        metadata_time = time.perf_counter() - started
        metadata_bytes = table.traffic.nbytes
        metadata_requests = table.traffic.requests

        print("\n[Format: Blosc2 B2Z (Lazy RemoteCTable)]")
        for name, value in metadata.items():
            rendered = pprint.pformat(value) if isinstance(value, (dict, list)) else value
            if name == "attrs" and value:
                rendered = "{\n " + rendered[1:]
            print(f"{name:<13}: {rendered}")

        print("\nSample rows (1st fetch):")
        started = time.perf_counter()
        sample = str(table[:5])
        first_time = time.perf_counter() - started
        first_bytes = table.traffic.nbytes - metadata_bytes
        first_requests = table.traffic.requests - metadata_requests
        print(sample)

        started = time.perf_counter()
        _ = str(table[:5])
        second_time = time.perf_counter() - started
        second_bytes = table.traffic.nbytes - metadata_bytes - first_bytes
        second_requests = table.traffic.requests - metadata_requests - first_requests

        print("\nTiming & Network Traffic:")
        print(
            f"  - Metadata open : {metadata_time * 1000:7.1f} ms  "
            f"({metadata_requests} requests, {metadata_bytes / 1024:8.2f} KB transferred)"
        )
        print(
            f"  - 1st row fetch : {first_time * 1000:7.1f} ms  "
            f"({first_requests} requests, {first_bytes / 1024:8.2f} KB transferred)"
        )
        cache_hit = " (cache hit!)" if second_requests == second_bytes == 0 else ""
        print(
            f"  - 2nd row fetch : {second_time * 1000:7.1f} ms  "
            f"({second_requests} requests, {second_bytes / 1024:8.2f} KB transferred){cache_hit}"
        )
        print(
            f"  - Total network : {table.traffic.requests} requests, "
            f"{table.traffic.nbytes / 1024:8.2f} KB transferred"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Remote .b2z CTable URL")
    parser.add_argument("--write", type=Path, metavar="FILE", help="Create a local .b2z CTable instead")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT_URL)
    args = parser.parse_args()

    if args.write is not None and args.url is not None:
        parser.error("URL cannot be combined with --write")
    if args.write is None and args.url is None:
        parser.error("provide a remote URL or use --write FILE.b2z")

    try:
        write_table(args) if args.write is not None else access_table(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
