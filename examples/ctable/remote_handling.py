#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Create or access a CTable with fixed-width, UTF-8, batch-backed and dictionary columns."""

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
    note: str = blosc2.field(blosc2.utf8(null_storage="mask"))
    message: str = blosc2.field(blosc2.vlstring(nullable=True, batch_rows=4096))
    tags: list[int] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.int16(), nullable=True, batch_rows=4096)
    )
    region: str = blosc2.field(blosc2.dictionary(nullable=True))


def make_notes(ids):
    notes = np.array(["", "café", "東京の観測", "🌦️ weather improving"], dtype=object)[ids % 4]
    notes = np.array(
        [f"{text} #{i}" if text else "" for i, text in zip(ids, notes, strict=True)], dtype=object
    )
    notes[ids % 43 == 0] = None
    return notes


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
            messages = [None if i % 47 == 0 else f"sensor {i}: café 東京" for i in ids]
            tags = [None if i % 53 == 0 else [int(i % 7), int(i % 11)] for i in ids]
            regions = np.array(["north", "south", "east", "west"], dtype=object)[ids % 4]
            regions[ids % 59 == 0] = None
            table.extend(
                {
                    "id": ids,
                    "station_id": (ids % 100).astype(np.int32),
                    "temperature": temperature,
                    "humidity": humidity,
                    "status": status,
                    "active": ids % 5 != 0,
                    "note": make_notes(ids),
                    "message": messages,
                    "tags": tags,
                    "region": regions,
                },
                validate=False,
            )
        table.create_index("tags", kind="membership")

    with blosc2.CTable.open(str(output)) as table:
        assert len(table) == args.rows
        assert table.attrs[:] == attrs
        null_counts = {
            name: table[name].null_count() for name in ("temperature", "humidity", "status", "note")
        }
        sample_ids = np.arange(min(args.rows, 8))
        expected = make_notes(sample_ids)
        expected[sample_ids % 43 == 0] = ""
        np.testing.assert_array_equal(table["note"][: len(sample_ids)], expected)
        assert null_counts["note"] == (args.rows + 42) // 43

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
    cache_options = {"cache_dir": args.cache_dir} if args.cache_dir is not None else {}
    with blosc2.open(args.url, storage_options=storage_options or None, **cache_options) as table:
        if not isinstance(table, blosc2.CTable):
            raise ValueError("input must be a CTable archive")
        remote = isinstance(table, blosc2.RemoteCTable)
        nbytes = table.nbytes
        metadata = {
            "type": type(table).__name__,
            "source": table.source if remote else args.url,
            "rows": table.nrows,
            "columns": table.col_names,
            "chunks": table.chunks,
            "blocks": table.blocks,
            "nbytes": f"{nbytes} ({nbytes / 1024**2:.2f} MiB)",
            "schema": table.schema_dict(),
            "attrs": dict(table.attrs),
        }
        if remote:
            metadata["max_concurrency"] = table.max_concurrency
            metadata["temporary_buffers"] = {
                "metadata": table.metadata_buffer_bytes,
                "rows": table.row_buffer_bytes,
            }
            metadata["cache_policy"] = table.cache_policy.name
            metadata["cache_bytes"] = f"{table.cache_bytes} ({table.cache_bytes / 1024:.2f} KiB)"
        metadata_time = time.perf_counter() - started
        metadata_bytes = table.traffic.nbytes if remote else 0
        metadata_requests = table.traffic.requests if remote else 0
        extra_time = 0.0

        print(f"\n[Format: Blosc2 B2Z ({type(table).__name__})]")
        for name, value in metadata.items():
            rendered = pprint.pformat(value) if isinstance(value, (dict, list)) else value
            if name == "attrs" and value:
                rendered = "{\n " + rendered[1:]
            print(f"{name:<13}: {rendered}")

        probe = slice(0, min(5, table.nrows))
        if "message" in table.col_names:
            print("\nBatch-backed message slice (whole compressed batches are the transfer unit):")
            for label in ("cold", "warm"):
                before_bytes = table.traffic.nbytes if remote else 0
                before_requests = table.traffic.requests if remote else 0
                started = time.perf_counter()
                messages = table["message"][probe]
                elapsed = time.perf_counter() - started
                extra_time += elapsed
                transferred = table.traffic.nbytes - before_bytes if remote else 0
                requests = table.traffic.requests - before_requests if remote else 0
                print(
                    f"  - {label:<4} batch read: {elapsed * 1000:7.1f} ms  "
                    f"({requests} requests, {transferred / 1024:8.2f} KB transferred)"
                )
            print(f"    {messages}")

        if (
            "tags" in table.col_names
            and table._get_index_catalog().get("tags", {}).get("kind") == "membership"
        ):
            before_bytes = table.traffic.nbytes if remote else 0
            before_requests = table.traffic.requests if remote else 0
            started = time.perf_counter()
            matching_ids = table[table["tags"].contains(3)]["id"][:5]
            elapsed = time.perf_counter() - started
            extra_time += elapsed
            print("\nIndexed list membership (posting batches; tags stay unopened for an id projection):")
            print(
                f"  - contains(3)     : {elapsed * 1000:7.1f} ms  "
                f"({table.traffic.requests - before_requests if remote else 0} requests, "
                f"{(table.traffic.nbytes - before_bytes if remote else 0) / 1024:8.2f} KB transferred)"
            )
            print(f"    first ids: {matching_ids}")

        if "region" in table.col_names:
            dictionary = table["region"].raw
            print("\nDictionary costs (codes first, then full vocabulary on first decode):")
            before_bytes = table.traffic.nbytes if remote else 0
            before_requests = table.traffic.requests if remote else 0
            started = time.perf_counter()
            _ = dictionary.codes[probe]
            elapsed = time.perf_counter() - started
            extra_time += elapsed
            print(
                f"  - code read       : {elapsed * 1000:7.1f} ms  "
                f"({table.traffic.requests - before_requests if remote else 0} requests, "
                f"{(table.traffic.nbytes - before_bytes if remote else 0) / 1024:8.2f} KB transferred)"
            )
            before_bytes = table.traffic.nbytes if remote else 0
            before_requests = table.traffic.requests if remote else 0
            started = time.perf_counter()
            regions = table["region"][probe]
            elapsed = time.perf_counter() - started
            extra_time += elapsed
            print(
                f"  - first decode    : {elapsed * 1000:7.1f} ms  "
                f"({table.traffic.requests - before_requests if remote else 0} requests, "
                f"{(table.traffic.nbytes - before_bytes if remote else 0) / 1024:8.2f} KB transferred)"
            )
            print(f"    {regions}")

        sample_start = max(0, table.nrows // 2 - 2)
        sample_stop = min(sample_start + 5, table.nrows)
        sample_slice = slice(sample_start, sample_stop)
        print(f"\nSample rows [{sample_start}:{sample_stop}] around the midpoint (1st fetch):")
        started = time.perf_counter()
        sample = str(table[sample_slice])
        first_time = time.perf_counter() - started
        first_bytes = table.traffic.nbytes - metadata_bytes if remote else 0
        first_requests = table.traffic.requests - metadata_requests if remote else 0
        print(sample)

        started = time.perf_counter()
        _ = str(table[sample_slice])
        second_time = time.perf_counter() - started
        second_bytes = table.traffic.nbytes - metadata_bytes - first_bytes if remote else 0
        second_requests = table.traffic.requests - metadata_requests - first_requests if remote else 0

        print("\nTiming & Network Traffic:" if remote else "\nTiming (local reads; no network traffic):")
        print(
            f"  - Metadata open : {metadata_time * 1000:7.1f} ms  "
            f"({metadata_requests} requests, {metadata_bytes / 1024:8.2f} KB transferred)"
        )
        print(
            f"  - 1st row fetch : {first_time * 1000:7.1f} ms  "
            f"({first_requests} requests, {first_bytes / 1024:8.2f} KB transferred)"
        )
        cache_hit = " (cache hit!)" if remote and second_requests == second_bytes == 0 else ""
        print(
            f"  - 2nd row fetch : {second_time * 1000:7.1f} ms  "
            f"({second_requests} requests, {second_bytes / 1024:8.2f} KB transferred){cache_hit}"
        )
        # Sum operation wall times (including decoding/cache work), not printing.
        total_time = metadata_time + first_time + second_time + extra_time
        if "note" in table.col_names and table["note"].is_utf8:
            start = max(0, table.nrows - 5)
            print(f"\nUTF-8 note slice [{start}:{table.nrows}] (may overlap warmed blocks in small tables):")
            for label in ("1st", "2nd"):
                before_bytes = table.traffic.nbytes if remote else 0
                before_requests = table.traffic.requests if remote else 0
                started = time.perf_counter()
                notes = table["note"][start:]
                elapsed = time.perf_counter() - started
                total_time += elapsed
                transferred = table.traffic.nbytes - before_bytes if remote else 0
                requests = table.traffic.requests - before_requests if remote else 0
                print(
                    f"  - {label} note fetch: {elapsed * 1000:7.1f} ms  "
                    f"({requests} requests, {transferred / 1024:8.2f} KB transferred)"
                )
                if label == "1st":
                    print(f"    {notes}")
        if remote:
            print(
                f"\nTotal network : {total_time * 1000:7.1f} ms  "
                f"({table.traffic.requests} requests, {table.traffic.nbytes / 1024:8.2f} KB transferred)"
            )
            print(f"Retained cache: {table.cache_bytes / 1024:8.2f} KB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "url", nargs="?", help="Local .b2z CTable path or remote URL (s3://, http://, https://)"
    )
    parser.add_argument("--write", type=Path, metavar="FILE", help="Create a local .b2z CTable instead")
    parser.add_argument(
        "--cache-dir", type=Path, metavar="DIR", help="Persist remote data in DIR (default: in-memory cache)"
    )
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT_URL)
    args = parser.parse_args()

    if args.write is not None and args.url is not None:
        parser.error("URL cannot be combined with --write")
    if args.write is None and args.url is None:
        parser.error("provide a local path or remote URL, or use --write FILE.b2z")

    try:
        write_table(args) if args.write is not None else access_table(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
