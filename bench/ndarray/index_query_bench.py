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
from blosc2 import indexing as blosc2_indexing

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
CHUNK_LEN = 100_000
BLOCK_LEN = 20_000
DEFAULT_REPEATS = 3
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


def base_array_path(size_dir: Path, size: int, dist: str) -> Path:
    return size_dir / f"size_{size}_{dist}.b2nd"


def indexed_array_path(size_dir: Path, size: int, dist: str, kind: str) -> Path:
    return size_dir / f"size_{size}_{dist}.{kind}.b2nd"


def benchmark_scan_once(expr) -> tuple[float, int]:
    start = time.perf_counter()
    result = expr.compute(_use_index=False)[:]
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def benchmark_index_once(arr: blosc2.NDArray, cond) -> tuple[float, int]:
    start = time.perf_counter()
    result = arr[cond][:]
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

    light = descriptor.get("light")
    if light is not None:
        for key in ("values_path", "bucket_positions_path", "offsets_path"):
            array = blosc2.open(light[key])
            logical += int(np.prod(array.shape)) * array.dtype.itemsize
            disk += os.path.getsize(light[key])

    reduced = descriptor.get("reduced")
    if reduced is not None:
        values = blosc2.open(reduced["values_path"])
        positions = blosc2.open(reduced["positions_path"])
        offsets = blosc2.open(reduced["offsets_path"])
        logical += values.shape[0] * values.dtype.itemsize
        logical += positions.shape[0] * positions.dtype.itemsize
        logical += offsets.shape[0] * offsets.dtype.itemsize
        disk += os.path.getsize(reduced["values_path"])
        disk += os.path.getsize(reduced["positions_path"])
        disk += os.path.getsize(reduced["offsets_path"])

    full = descriptor.get("full")
    if full is not None:
        values = blosc2.open(full["values_path"])
        positions = blosc2.open(full["positions_path"])
        logical += values.shape[0] * values.dtype.itemsize
        logical += positions.shape[0] * positions.dtype.itemsize
        disk += os.path.getsize(full["values_path"])
        disk += os.path.getsize(full["positions_path"])
    return logical, disk


def _source_data_factory(size: int, dist: str):
    data = None

    def get_data() -> np.ndarray:
        nonlocal data
        if data is None:
            data = make_source_data(size, dist)
        return data

    return get_data


def _valid_index_descriptor(arr: blosc2.NDArray, kind: str) -> dict | None:
    for descriptor in arr.indexes:
        if descriptor.get("version") != blosc2_indexing.INDEX_FORMAT_VERSION:
            continue
        if (
            descriptor.get("field") == "id"
            and descriptor.get("kind") == kind
            and not descriptor.get("stale", False)
        ):
            return descriptor
    return None


def _open_or_build_persistent_array(path: Path, get_data) -> blosc2.NDArray:
    if path.exists():
        return blosc2.open(path, mode="a")
    blosc2.remove_urlpath(path)
    return build_persistent_array(get_data(), path)


def _open_or_build_indexed_array(path: Path, get_data, kind: str) -> tuple[blosc2.NDArray, float]:
    if path.exists():
        arr = blosc2.open(path, mode="a")
        if _valid_index_descriptor(arr, kind) is not None:
            return arr, 0.0
        if arr.indexes:
            arr.drop_index(field="id")
        blosc2.remove_urlpath(path)

    arr = build_persistent_array(get_data(), path)
    build_start = time.perf_counter()
    arr.create_index(field="id", kind=kind)
    return arr, time.perf_counter() - build_start


def benchmark_size(size: int, size_dir: Path, dist: str, query_width: int) -> list[dict]:
    get_data = _source_data_factory(size, dist)
    arr = _open_or_build_persistent_array(base_array_path(size_dir, size, dist), get_data)
    lo = size // 2
    hi = min(size, lo + query_width)
    condition = blosc2.lazyexpr(f"(id >= {lo}) & (id < {hi})", arr.fields)
    expr = condition.where(arr)
    base_bytes = size * arr.dtype.itemsize
    compressed_base_bytes = os.path.getsize(arr.urlpath)

    scan_ms = benchmark_scan_once(expr)[0] * 1_000

    rows = []
    for kind in KINDS:
        idx_arr, build_time = _open_or_build_indexed_array(indexed_array_path(size_dir, size, dist, kind), get_data, kind)
        idx_cond = blosc2.lazyexpr(f"(id >= {lo}) & (id < {hi})", idx_arr.fields)
        idx_expr = idx_cond.where(idx_arr)
        explanation = idx_expr.explain()
        logical_index_bytes, disk_index_bytes = index_sizes(idx_arr.indexes[0])
        cold_time, index_len = benchmark_index_once(idx_arr, idx_cond)

        rows.append(
            {
                "size": size,
                "dist": dist,
                "kind": kind,
                "query_rows": index_len,
                "build_s": build_time,
                "create_idx_ms": build_time * 1_000,
                "scan_ms": scan_ms,
                "cold_ms": cold_time * 1_000,
                "cold_speedup": scan_ms / (cold_time * 1_000),
                "warm_ms": None,
                "warm_speedup": None,
                "candidate_units": explanation["candidate_units"],
                "total_units": explanation["total_units"],
                "logical_index_bytes": logical_index_bytes,
                "disk_index_bytes": disk_index_bytes,
                "index_pct": logical_index_bytes / base_bytes * 100,
                "index_pct_disk": disk_index_bytes / compressed_base_bytes * 100,
                "_arr": idx_arr,
                "_cond": idx_cond,
            }
        )
    return rows


def measure_warm_queries(rows: list[dict], repeats: int) -> None:
    if repeats <= 0:
        return
    for result in rows:
        arr = result["_arr"]
        cond = result["_cond"]
        index_runs = [benchmark_index_once(arr, cond)[0] for _ in range(repeats)]
        warm_ms = statistics.median(index_runs) * 1_000 if index_runs else None
        result["warm_ms"] = warm_ms
        result["warm_speedup"] = None if warm_ms is None else result["scan_ms"] / warm_ms


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
        "--query-width",
        type=parse_human_size,
        default=1_000,
        help="Width of the range predicate. Supports suffixes like 1k, 1K, 1M, 1G. Default: 1000.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of repeated warm-query measurements after the first cold query. Default: 3.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Directory where benchmark arrays and index sidecars should be written and kept.",
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
    if args.repeats < 0:
        raise SystemExit("--repeats must be >= 0")
    sizes = (args.size,) if args.size is not None else SIZES
    dists = DISTS if args.dist == "all" else (args.dist,)

    if args.outdir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_benchmarks(sizes, dists, Path(tmpdir), args.dist, args.query_width, args.repeats)
    else:
        args.outdir.mkdir(parents=True, exist_ok=True)
        run_benchmarks(sizes, dists, args.outdir, args.dist, args.query_width, args.repeats)


def run_benchmarks(
    sizes: tuple[int, ...],
    dists: tuple[str, ...],
    size_dir: Path,
    dist_label: str,
    query_width: int,
    repeats: int,
) -> None:
    all_results = []
    print("Structured range-query benchmark across index kinds")
    print(
        f"chunks={CHUNK_LEN:,}, blocks={BLOCK_LEN:,}, repeats={repeats}, dist={dist_label}, "
        f"query_width={query_width:,}"
    )
    for dist in dists:
        for size in sizes:
            size_results = benchmark_size(size, size_dir, dist, query_width)
            all_results.extend(size_results)

    print()
    print("Cold Query Table")
    print_table(
        all_results,
        [
            ("rows", lambda result: f"{result['size']:,}"),
            ("dist", lambda result: result["dist"]),
            ("kind", lambda result: result["kind"]),
            ("create_idx_ms", lambda result: f"{result['create_idx_ms']:.3f}"),
            ("scan_ms", lambda result: f"{result['scan_ms']:.3f}"),
            ("cold_ms", lambda result: f"{result['cold_ms']:.3f}"),
            ("speedup", lambda result: f"{result['cold_speedup']:.2f}x"),
            ("logical_bytes", lambda result: f"{result['logical_index_bytes']:,}"),
            ("disk_bytes", lambda result: f"{result['disk_index_bytes']:,}"),
            ("index_pct", lambda result: f"{result['index_pct']:.4f}%"),
            ("index_pct_disk", lambda result: f"{result['index_pct_disk']:.4f}%"),
        ],
    )
    if repeats > 0:
        measure_warm_queries(all_results, repeats)
        print()
        print("Warm Query Table")
        print_table(
            all_results,
            [
                ("rows", lambda result: f"{result['size']:,}"),
                ("dist", lambda result: result["dist"]),
                ("kind", lambda result: result["kind"]),
                ("create_idx_ms", lambda result: f"{result['create_idx_ms']:.3f}"),
                ("scan_ms", lambda result: f"{result['scan_ms']:.3f}"),
                ("warm_ms", lambda result: f"{result['warm_ms']:.3f}" if result["warm_ms"] is not None else "-"),
                (
                    "speedup",
                    lambda result: f"{result['warm_speedup']:.2f}x"
                    if result["warm_speedup"] is not None
                    else "-",
                ),
                ("logical_bytes", lambda result: f"{result['logical_index_bytes']:,}"),
                ("disk_bytes", lambda result: f"{result['disk_index_bytes']:,}"),
                ("index_pct", lambda result: f"{result['index_pct']:.4f}%"),
                ("index_pct_disk", lambda result: f"{result['index_pct_disk']:.4f}%"),
            ],
        )


def _format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def _table_rows(results: list[dict], columns: list[tuple[str, callable]]) -> tuple[list[str], list[list[str]], list[int]]:
    headers = [header for header, _ in columns]
    widths = [len(header) for header in headers]
    rows = [[formatter(result) for _, formatter in columns] for result in results]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    return headers, rows, widths


def print_table(results: list[dict], columns: list[tuple[str, callable]]) -> None:
    headers, rows, widths = _table_rows(results, columns)
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


if __name__ == "__main__":
    main()
