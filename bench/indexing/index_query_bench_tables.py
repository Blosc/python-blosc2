#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import os
import re
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import tables

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
CHUNK_LEN = 100_000
DEFAULT_REPEATS = 3
KINDS = ("ultralight", "light", "medium", "full")
DISTS = ("sorted", "block-shuffled", "random")
RNG_SEED = 0
TABLE_NAME = "data"
DATA_FILTERS = tables.Filters(complevel=5, complib="blosc2:zstd", shuffle=True)


def dtype_token(dtype: np.dtype) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", np.dtype(dtype).name).strip("_")


def make_ordered_ids(size: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        values = np.zeros(size, dtype=dtype)
        values[size // 2 :] = True
        return values

    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        start = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            start = max(int(info.min), -(unique_count // 2))
        positions = np.arange(size, dtype=np.int64)
        values = start + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        return np.linspace(-span / 2, span / 2, num=size, endpoint=False, dtype=dtype)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def fill_ids(ids: np.ndarray, ordered_ids: np.ndarray, dist: str, rng: np.random.Generator) -> None:
    size = ids.shape[0]
    if dist == "sorted":
        ids[:] = ordered_ids
        return

    if dist == "block-shuffled":
        nblocks = (size + CHUNK_LEN - 1) // CHUNK_LEN
        order = rng.permutation(nblocks)
        dest = 0
        for src_block in order:
            src_start = int(src_block) * CHUNK_LEN
            src_stop = min(src_start + CHUNK_LEN, size)
            block_size = src_stop - src_start
            ids[dest : dest + block_size] = ordered_ids[src_start:src_stop]
            dest += block_size
        return

    if dist == "random":
        ids[:] = ordered_ids
        rng.shuffle(ids)
        return

    raise ValueError(f"unsupported distribution {dist!r}")


def make_source_data(size: int, dist: str, id_dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype([("id", id_dtype), ("payload", np.float32)])
    data = np.zeros(size, dtype=dtype)
    fill_ids(data["id"], make_ordered_ids(size, id_dtype), dist, np.random.default_rng(RNG_SEED))
    return data


def _source_data_factory(size: int, dist: str, id_dtype: np.dtype):
    data = None

    def get_data() -> np.ndarray:
        nonlocal data
        if data is None:
            data = make_source_data(size, dist, id_dtype)
        return data

    return get_data


def _ordered_ids_factory(size: int, id_dtype: np.dtype):
    ordered_ids = None

    def get_ordered_ids() -> np.ndarray:
        nonlocal ordered_ids
        if ordered_ids is None:
            ordered_ids = make_ordered_ids(size, id_dtype)
        return ordered_ids

    return get_ordered_ids


def base_table_path(size_dir: Path, size: int, dist: str, id_dtype: np.dtype) -> Path:
    return size_dir / f"size_{size}_{dist}_{dtype_token(id_dtype)}.h5"


def indexed_table_path(size_dir: Path, size: int, dist: str, kind: str, id_dtype: np.dtype) -> Path:
    return size_dir / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{kind}.h5"


def build_persistent_table(data: np.ndarray, path: Path) -> tuple[tables.File, tables.Table]:
    h5 = tables.open_file(path, mode="w")
    table = h5.create_table(
        "/",
        TABLE_NAME,
        obj=data,
        filters=DATA_FILTERS,
        expectedrows=len(data),
        chunkshape=CHUNK_LEN,
    )
    h5.flush()
    return h5, table


def benchmark_once(table: tables.Table, condition: str) -> tuple[float, int]:
    start = time.perf_counter()
    result = table.read_where(condition)
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def pytables_index_sizes(h5: tables.File) -> int:
    total = 0
    if "/_i_data" not in h5:
        return total
    for node in h5.walk_nodes("/_i_data"):
        dtype = getattr(node, "dtype", None)
        shape = getattr(node, "shape", None)
        if dtype is None or shape is None:
            continue
        nitems = 1
        for dim in shape:
            nitems *= int(dim)
        total += nitems * dtype.itemsize
    return total


def _valid_index(table: tables.Table, kind: str) -> bool:
    if not table.cols.id.is_indexed:
        return False
    return table.colindexes["id"].kind == kind


def _open_or_build_base_table(path: Path, get_data) -> tuple[tables.File, tables.Table]:
    if path.exists():
        h5 = tables.open_file(path, mode="a")
        return h5, getattr(h5.root, TABLE_NAME)
    path.unlink(missing_ok=True)
    return build_persistent_table(get_data(), path)


def _open_or_build_indexed_table(path: Path, get_data, kind: str) -> tuple[tables.File, tables.Table, float]:
    if path.exists():
        h5 = tables.open_file(path, mode="a")
        table = getattr(h5.root, TABLE_NAME)
        if _valid_index(table, kind):
            return h5, table, 0.0
        h5.close()
        path.unlink()

    h5, table = build_persistent_table(get_data(), path)
    build_start = time.perf_counter()
    table.cols.id.create_index(kind=kind)
    h5.flush()
    return h5, table, time.perf_counter() - build_start


def _query_bounds(ordered_ids: np.ndarray, query_width: int) -> tuple[object, object]:
    if ordered_ids.size == 0:
        raise ValueError("benchmark arrays must not be empty")

    lo_idx = ordered_ids.size // 2
    hi_idx = min(ordered_ids.size - 1, lo_idx + max(query_width - 1, 0))
    return ordered_ids[lo_idx].item(), ordered_ids[hi_idx].item()


def _literal(value: object, dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return "True" if bool(value) else "False"
    if dtype.kind == "f":
        return repr(float(value))
    if dtype.kind in {"i", "u"}:
        return str(int(value))
    raise ValueError(f"unsupported dtype for literal formatting: {dtype}")


def _condition_expr(lo: object, hi: object, dtype: np.dtype) -> str:
    return f"(id >= {_literal(lo, dtype)}) & (id <= {_literal(hi, dtype)})"


def benchmark_size(size: int, size_dir: Path, dist: str, query_width: int, id_dtype: np.dtype) -> list[dict]:
    get_data = _source_data_factory(size, dist, id_dtype)
    get_ordered_ids = _ordered_ids_factory(size, id_dtype)
    base_h5, base_table = _open_or_build_base_table(base_table_path(size_dir, size, dist, id_dtype), get_data)
    lo, hi = _query_bounds(get_ordered_ids(), query_width)
    condition = _condition_expr(lo, hi, id_dtype)
    base_bytes = size * np.dtype([("id", id_dtype), ("payload", np.float32)]).itemsize
    compressed_base_bytes = os.path.getsize(base_h5.filename)

    scan_ms = benchmark_once(base_table, condition)[0] * 1_000

    rows = []
    for kind in KINDS:
        idx_h5, idx_table, build_time = _open_or_build_indexed_table(
            indexed_table_path(size_dir, size, dist, kind, id_dtype), get_data, kind
        )
        cold_time, index_len = benchmark_once(idx_table, condition)
        indexed_file_bytes = os.path.getsize(idx_h5.filename)
        disk_index_bytes = max(0, indexed_file_bytes - compressed_base_bytes)
        logical_index_bytes = pytables_index_sizes(idx_h5)

        rows.append(
            {
                "size": size,
                "dist": dist,
                "kind": kind,
                "query_rows": index_len,
                "create_idx_ms": build_time * 1_000,
                "scan_ms": scan_ms,
                "cold_ms": cold_time * 1_000,
                "cold_speedup": scan_ms / (cold_time * 1_000),
                "warm_ms": None,
                "warm_speedup": None,
                "logical_index_bytes": logical_index_bytes,
                "disk_index_bytes": disk_index_bytes,
                "index_pct": logical_index_bytes / base_bytes * 100,
                "index_pct_disk": disk_index_bytes / compressed_base_bytes * 100,
                "_h5": idx_h5,
                "_table": idx_table,
                "_condition": condition,
            }
        )

    base_h5.close()
    return rows


def measure_warm_queries(rows: list[dict], repeats: int) -> None:
    if repeats <= 0:
        return
    for result in rows:
        table = result["_table"]
        condition = result["_condition"]
        index_runs = [benchmark_once(table, condition)[0] for _ in range(repeats)]
        warm_ms = statistics.median(index_runs) * 1_000 if index_runs else None
        result["warm_ms"] = warm_ms
        result["warm_speedup"] = None if warm_ms is None else result["scan_ms"] / warm_ms


def close_rows(rows: list[dict]) -> None:
    for result in rows:
        h5 = result.pop("_h5", None)
        result.pop("_table", None)
        result.pop("_condition", None)
        if h5 is not None and h5.isopen:
            h5.close()


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
    parser = argparse.ArgumentParser(description="Benchmark PyTables OPSI index kinds.")
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
        "--dtype",
        default="float64",
        help="NumPy dtype for the indexed field. Examples: float64, float32, int16, bool. Default: float64.",
    )
    parser.add_argument(
        "--dist",
        choices=(*DISTS, "all"),
        default="all",
        help="Data distribution to benchmark. Default: all.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        help="Optional directory to keep and reuse generated HDF5 files.",
    )
    return parser.parse_args()


def _format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def print_table(rows: list[dict], columns: list[tuple[str, callable]]) -> None:
    header = [name for name, _ in columns]
    body = [[formatter(row) for _, formatter in columns] for row in rows]
    widths = [len(name) for name in header]
    for row in body:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    print(_format_row(header, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in body:
        print(_format_row(row, widths))


def run_benchmark() -> None:
    args = parse_args()
    try:
        id_dtype = np.dtype(args.dtype)
    except TypeError as exc:
        raise SystemExit(f"unsupported dtype {args.dtype!r}") from exc
    if id_dtype.kind not in {"b", "i", "u", "f"}:
        raise SystemExit(f"--dtype only supports bool, integer, and floating-point dtypes; got {id_dtype}")
    sizes = (args.size,) if args.size is not None else SIZES
    dists = DISTS if args.dist == "all" else (args.dist,)
    dist_label = args.dist
    repeats = max(0, args.repeats)
    query_width = args.query_width

    if args.outdir is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _run_benchmark(Path(tmpdir), sizes, dists, dist_label, repeats, query_width, id_dtype)
    else:
        size_dir = args.outdir.expanduser()
        size_dir.mkdir(parents=True, exist_ok=True)
        _run_benchmark(size_dir, sizes, dists, dist_label, repeats, query_width, id_dtype)


def _run_benchmark(
    size_dir: Path,
    sizes: tuple[int, ...],
    dists: tuple[str, ...],
    dist_label: str,
    repeats: int,
    query_width: int,
    id_dtype: np.dtype,
) -> None:
    all_results = []
    print("Structured range-query benchmark across PyTables index kinds")
    print(
        f"chunks={CHUNK_LEN:,}, repeats={repeats}, dist={dist_label}, "
        f"query_width={query_width:,}, dtype={id_dtype.name}, complib={DATA_FILTERS.complib}"
    )
    try:
        for dist in dists:
            for size in sizes:
                size_results = benchmark_size(size, size_dir, dist, query_width, id_dtype)
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
    finally:
        close_rows(all_results)


if __name__ == "__main__":
    run_benchmark()
