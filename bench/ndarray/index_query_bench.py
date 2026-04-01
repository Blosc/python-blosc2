#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import os
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np

import blosc2

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
CHUNK_LEN = 100_000
BLOCK_LEN = 20_000
REPEATS = 5
KINDS = ("ultralight", "light", "medium", "full")
DISTS = ("sorted", "block-shuffled", "random")
RNG_SEED = 0


def fill_ids(ids: np.ndarray, dist: str, rng: np.random.Generator) -> None:
    size = ids.shape[0]
    if dist == "sorted":
        ids[:] = np.arange(size, dtype=np.int64)
        return

    if dist == "block-shuffled":
        nblocks = (size + BLOCK_LEN - 1) // BLOCK_LEN
        order = rng.permutation(nblocks)
        dest = 0
        for src_block in order:
            src_start = int(src_block) * BLOCK_LEN
            src_stop = min(src_start + BLOCK_LEN, size)
            block_size = src_stop - src_start
            ids[dest : dest + block_size] = np.arange(src_start, src_stop, dtype=np.int64)
            dest += block_size
        return

    if dist == "random":
        ids[:] = np.arange(size, dtype=np.int64)
        rng.shuffle(ids)
        return

    raise ValueError(f"unsupported distribution {dist!r}")


def make_source_data(size: int, dist: str) -> np.ndarray:
    dtype = np.dtype([("id", np.int64), ("payload", np.float32)])
    data = np.zeros(size, dtype=dtype)
    fill_ids(data["id"], dist, np.random.default_rng(RNG_SEED))
    return data


def build_array(data: np.ndarray) -> blosc2.NDArray:
    return blosc2.asarray(data, chunks=(CHUNK_LEN,), blocks=(BLOCK_LEN,))


def build_persistent_array(data: np.ndarray, path: Path) -> blosc2.NDArray:
    return blosc2.asarray(data, urlpath=path, mode="w", chunks=(CHUNK_LEN,), blocks=(BLOCK_LEN,))


def benchmark_once(expr, *, use_index: bool) -> tuple[float, int]:
    start = time.perf_counter()
    result = expr.compute(_use_index=use_index)[:]
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def index_sizes(descriptor: dict) -> tuple[int, int]:
    logical = 0
    disk = 0
    for level_info in descriptor["levels"].values():
        dtype = np.dtype(level_info["dtype"])
        logical += dtype.itemsize * level_info["nsegments"]
        if level_info["path"]:
            disk += os.path.getsize(level_info["path"])

    full = descriptor.get("full")
    if full is not None:
        values = blosc2.open(full["values_path"])
        positions = blosc2.open(full["positions_path"])
        logical += values.shape[0] * values.dtype.itemsize
        logical += positions.shape[0] * positions.dtype.itemsize
        disk += os.path.getsize(full["values_path"])
        disk += os.path.getsize(full["positions_path"])
    return logical, disk


def benchmark_size(size: int, size_dir: Path, dist: str) -> list[dict]:
    data = make_source_data(size, dist)
    arr = build_persistent_array(data, size_dir / f"size_{size}_{dist}.b2nd")
    del data
    lo = size // 2
    width = 2_500
    hi = min(size, lo + width)
    expr = blosc2.lazyexpr(f"(id >= {lo}) & (id < {hi})", arr.fields).where(arr)
    base_bytes = size * arr.dtype.itemsize
    compressed_base_bytes = os.path.getsize(arr.urlpath)

    scan_ms = benchmark_once(expr, use_index=False)[0] * 1_000

    rows = []
    for kind in KINDS:
        if arr.indexes:
            arr.drop_index(field="id")
        build_start = time.perf_counter()
        arr.create_index(field="id", kind=kind)
        build_time = time.perf_counter() - build_start
        explanation = expr.explain()
        logical_index_bytes, disk_index_bytes = index_sizes(arr.indexes[0])

        warm_index, index_len = benchmark_once(expr, use_index=True)
        del warm_index
        index_runs = [benchmark_once(expr, use_index=True)[0] for _ in range(REPEATS)]
        index_ms = statistics.median(index_runs) * 1_000

        rows.append(
            {
                "size": size,
                "dist": dist,
                "kind": kind,
                "level": explanation["level"],
                "query_rows": index_len,
                "build_s": build_time,
                "create_idx_ms": build_time * 1_000,
                "scan_ms": scan_ms,
                "index_ms": index_ms,
                "speedup": scan_ms / index_ms,
                "candidate_units": explanation["candidate_units"],
                "total_units": explanation["total_units"],
                "logical_index_bytes": logical_index_bytes,
                "disk_index_bytes": disk_index_bytes,
                "index_pct": logical_index_bytes / base_bytes * 100,
                "index_pct_disk": disk_index_bytes / compressed_base_bytes * 100,
            }
        )
    return rows


def parse_human_size(value: str) -> int:
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError("size must not be empty")

    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    suffix = value[-1].lower()
    if suffix in suffixes:
        number = value[:-1]
        if not number:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}")
        try:
            parsed = int(number)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc
        size = parsed * suffixes[suffix]
    else:
        try:
            size = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size {value!r}") from exc

    if size <= 0:
        raise argparse.ArgumentTypeError("size must be a positive integer")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark python-blosc2 index kinds.")
    parser.add_argument(
        "--size",
        type=parse_human_size,
        help="Benchmark a single array size. Supports suffixes like 1k, 1K, 1M, 1G.",
    )
    parser.add_argument(
        "--dist",
        choices=(*DISTS, "all"),
        default="sorted",
        help="Distribution for the indexed field. Use 'all' to benchmark every distribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = (args.size,) if args.size is not None else SIZES
    dists = DISTS if args.dist == "all" else (args.dist,)

    with tempfile.TemporaryDirectory() as tmpdir:
        size_dir = Path(tmpdir)
        all_results = []
        print("Structured range-query benchmark across index kinds")
        print(f"chunks={CHUNK_LEN:,}, blocks={BLOCK_LEN:,}, repeats={REPEATS}, dist={args.dist}")
        print(
            "size,dist,kind,level,query_rows,build_s,create_idx_ms,scan_ms,index_ms,speedup,"
            "candidate_units,total_units,logical_index_bytes,disk_index_bytes,index_pct,index_pct_disk"
        )
        for dist in dists:
            for size in sizes:
                size_results = benchmark_size(size, size_dir, dist)
                all_results.extend(size_results)
                for result in size_results:
                    print(
                        f"{result['size']},"
                        f"{result['dist']},"
                        f"{result['kind']},"
                        f"{result['level']},"
                        f"{result['query_rows']},"
                        f"{result['build_s']:.4f},"
                        f"{result['create_idx_ms']:.3f},"
                        f"{result['scan_ms']:.3f},"
                        f"{result['index_ms']:.3f},"
                        f"{result['speedup']:.2f},"
                        f"{result['candidate_units']},"
                        f"{result['total_units']},"
                        f"{result['logical_index_bytes']},"
                        f"{result['disk_index_bytes']},"
                        f"{result['index_pct']:.4f},"
                        f"{result['index_pct_disk']:.4f}"
                    )

        print()
        print("Table")
        headers = [
            "rows",
            "dist",
            "kind",
            "level",
            "create_idx_ms",
            "scan_ms",
            "index_ms",
            "speedup",
            "logical_bytes",
            "disk_bytes",
            "index_pct",
            "index_pct_disk",
        ]
        table_rows = []
        for result in all_results:
            table_rows.append(
                [
                    f"{result['size']:,}",
                    result["dist"],
                    result["kind"],
                    result["level"],
                    f"{result['create_idx_ms']:.3f}",
                    f"{result['scan_ms']:.3f}",
                    f"{result['index_ms']:.3f}",
                    f"{result['speedup']:.2f}x",
                    f"{result['logical_index_bytes']:,}",
                    f"{result['disk_index_bytes']:,}",
                    f"{result['index_pct']:.4f}%",
                    f"{result['index_pct_disk']:.4f}%",
                ]
            )

        widths = [len(header) for header in headers]
        for row in table_rows:
            widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]

        def format_row(row: list[str]) -> str:
            return "  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))

        print(format_row(headers))
        print(format_row(["-" * width for width in widths]))
        for row in table_rows:
            print(format_row(row))


if __name__ == "__main__":
    main()
