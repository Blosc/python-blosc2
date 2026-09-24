#!/usr/bin/env python3
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Create a PyTables table locally or access it remotely through RemoteCTable."""

import argparse
import pprint
import sys
import time
from pathlib import Path

import numpy as np
import tables

import blosc2

DEFAULT_PROFILE = "blosc2"
DEFAULT_ENDPOINT_URL = "https://s3.us-west-001.backblazeb2.com"
TABLE_NAME = "readings"


class Reading(tables.IsDescription):
    id = tables.Int64Col(pos=0)
    station_id = tables.Int32Col(pos=1)
    temperature = tables.Float32Col(pos=2)
    humidity = tables.Int16Col(pos=3)
    status = tables.StringCol(8, pos=4)
    active = tables.BoolCol(pos=5)
    note = tables.StringCol(32, pos=6)


def write_table(args) -> None:
    output = args.write
    if output.suffix != ".h5":
        raise ValueError("output must end in .h5")
    if args.rows < 1 or args.batch_size < 1:
        raise ValueError("--rows and --batch-size must be positive")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    indexed_columns = []
    if args.full is not None:
        indexed_columns = (
            list(Reading.columns) if args.full == "*" else [name.strip() for name in args.full.split(",")]
        )
        unknown = set(indexed_columns) - set(Reading.columns)
        if not all(indexed_columns) or unknown:
            raise ValueError(f"invalid --full columns: {', '.join(sorted(unknown)) or args.full!r}")
        if len(indexed_columns) != len(set(indexed_columns)):
            raise ValueError("--full columns must not contain duplicates")

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    statuses = np.array([b"ok", b"warning", b"offline", b""], dtype="S8")
    notes = np.array([b"", b"clear", b"cloudy", b"rain"], dtype="S32")

    with tables.open_file(output, mode="w") as h5file:
        filters = tables.Filters(complevel=5, complib="blosc2:zstd", shuffle=True)
        table = h5file.create_table(
            "/",
            TABLE_NAME,
            Reading,
            title="Synthetic weather-station readings",
            filters=filters,
            expectedrows=args.rows,
        )
        table.attrs.version = 1
        table.attrs.sampling_interval = 0.5
        table.attrs.description = "Synthetic weather-station readings"

        for start in range(0, args.rows, args.batch_size):
            stop = min(start + args.batch_size, args.rows)
            ids = np.arange(start, stop, dtype=np.int64)
            data = np.empty(len(ids), dtype=table.dtype)
            data["id"] = ids
            data["station_id"] = ids % 100
            data["temperature"] = rng.normal(18, 10, len(ids)).astype(np.float32)
            data["humidity"] = rng.integers(0, 101, len(ids), dtype=np.int16)
            data["status"] = statuses[(ids // 7) % len(statuses)]
            data["active"] = ids % 5 != 0
            data["note"] = notes[ids % len(notes)]
            table.append(data)
        table.flush()

        if indexed_columns:
            started = time.perf_counter()
            for name in indexed_columns:
                getattr(table.cols, name).create_csindex(filters=filters)
            print(f"Created FULL (CSI) indexes in {time.perf_counter() - started:.2f} s")

    with tables.open_file(output) as h5file:
        table = h5file.root.readings
        assert table.nrows == args.rows
        assert all(getattr(table.cols, name).index.is_csi for name in indexed_columns)

    print(f"Created {output} ({output.stat().st_size / 1_000_000:.1f} MB, {args.rows:,} rows)")
    print(f"FULL (CSI) indexes: {', '.join(indexed_columns) if indexed_columns else 'none'}")
    print(f"Now upload {output} to your cloud object storage.")


def access_table(args) -> None:
    storage_options = None
    if args.url.startswith("s3://"):
        storage_options = {
            "profile": args.profile,
            "client_kwargs": {"endpoint_url": args.endpoint_url},
        }

    print(f"Accessing: {args.url}::{TABLE_NAME}")
    started = time.perf_counter()
    cache_options = {"cache_dir": args.cache_dir} if args.cache_dir is not None else {}
    with blosc2.RemoteCTable(
        args.url, dataset=TABLE_NAME, storage_options=storage_options, **cache_options
    ) as table:
        metadata_time = time.perf_counter() - started
        metadata_bytes = table.traffic.nbytes
        metadata_requests = table.traffic.requests
        metadata = {
            "type": type(table).__name__,
            "rows": table.nrows,
            "columns": table.col_names,
            "schema": table.schema_dict(),
            "attrs": dict(table.attrs),
            "indexes": sorted(table._get_index_catalog()),
        }
        print("\n[Format: PyTables/HDF5]")
        for name, value in metadata.items():
            rendered = pprint.pformat(value) if isinstance(value, (dict, list)) else value
            print(f"{name:<9}: {rendered}")

        sample_start = max(0, table.nrows // 2 - 2)
        sample_stop = min(sample_start + 5, table.nrows)
        print(f"\nSample rows [{sample_start}:{sample_stop}]:")
        print(table[sample_start:sample_stop])

        started = time.perf_counter()
        ids = table.where("(station_id == 42) & active").id[:5]
        query_time = time.perf_counter() - started
        print("\nQuery: (station_id == 42) & active")
        print(f"first ids: {ids}")
        print(
            f"metadata: {metadata_time * 1000:.1f} ms, "
            f"{metadata_requests} requests, {metadata_bytes / 1024:.2f} KiB"
        )
        print(
            f"query:    {query_time * 1000:.1f} ms, "
            f"{table.traffic.requests - metadata_requests} requests, "
            f"{(table.traffic.nbytes - metadata_bytes) / 1024:.2f} KiB"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", help="Remote .h5 URL (s3://, http://, or https://)")
    parser.add_argument("--write", type=Path, metavar="FILE", help="Create a local .h5 table instead")
    parser.add_argument(
        "--full",
        nargs="?",
        const="*",
        metavar="COL1,COL2",
        help="Create FULL (CSI) indexes for every column, or only the comma-separated columns",
    )
    parser.add_argument(
        "--cache-dir", type=Path, metavar="DIR", help="Persist remote data in DIR (default: memory)"
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
        parser.error("provide a remote URL or use --write FILE.h5")
    if args.full is not None and args.write is None:
        parser.error("--full requires --write")

    try:
        write_table(args) if args.write is not None else access_table(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
