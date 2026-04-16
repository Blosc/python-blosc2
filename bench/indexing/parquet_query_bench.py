#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
DEFAULT_REPEATS = 3
DISTS = ("sorted", "block-shuffled", "permuted", "random")
LAYOUTS = ("row-group", "page-index")
RNG_SEED = 0
DEFAULT_ROW_GROUP_SIZE = 1_250_000
DEFAULT_MAX_ROWS_PER_PAGE = 10_000
DEFAULT_COMPRESSION = "snappy"
DATASET_LAYOUT_VERSION = "payload-ramp-v1"


def dtype_token(dtype: np.dtype) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", np.dtype(dtype).name).strip("_")


def payload_slice(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.float32)


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


def ordered_id_slice(size: int, start: int, stop: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if stop <= start:
        return np.empty(0, dtype=dtype)

    if dtype == np.dtype(np.bool_):
        values = np.zeros(stop - start, dtype=dtype)
        true_start = max(start, size // 2)
        if true_start < stop:
            values[true_start - start :] = True
        return values

    positions = np.arange(start, stop, dtype=np.int64)
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        base = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            base = max(int(info.min), -(unique_count // 2))
        values = base + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        values = positions.astype(np.float64, copy=False) - (span / 2)
        return values.astype(dtype, copy=False)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def ordered_id_at(size: int, index: int, dtype: np.dtype) -> object:
    return ordered_id_slice(size, index, index + 1, dtype)[0].item()


def ordered_ids_from_positions(positions: np.ndarray, size: int, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if positions.size == 0:
        return np.empty(0, dtype=dtype)

    if dtype == np.dtype(np.bool_):
        return (positions >= (size // 2)).astype(dtype, copy=False)

    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        unique_count = min(size, int(info.max) - int(info.min) + 1)
        base = int(info.min) if unique_count < size and dtype.kind == "i" else 0
        if dtype.kind == "i" and unique_count < size:
            base = max(int(info.min), -(unique_count // 2))
        values = base + (positions * unique_count) // size
        return values.astype(dtype, copy=False)

    if dtype.kind == "f":
        span = max(1, size)
        values = positions.astype(np.float64, copy=False) - (span / 2)
        return values.astype(dtype, copy=False)

    raise ValueError(f"unsupported dtype for benchmark: {dtype}")


def _block_order(size: int, block_len: int) -> np.ndarray:
    nblocks = (size + block_len - 1) // block_len
    return np.random.default_rng(RNG_SEED).permutation(nblocks)


def _fill_block_shuffled_ids(
    ids: np.ndarray, size: int, start: int, stop: int, block_len: int, order: np.ndarray
) -> None:
    cursor = start
    out_cursor = 0
    while cursor < stop:
        dest_block = cursor // block_len
        block_offset = cursor % block_len
        src_block = int(order[dest_block])
        src_start = src_block * block_len + block_offset
        take = min(stop - cursor, block_len - block_offset, size - src_start)
        ids[out_cursor : out_cursor + take] = ordered_id_slice(size, src_start, src_start + take, ids.dtype)
        cursor += take
        out_cursor += take


def _permuted_position_params(size: int) -> tuple[int, int]:
    if size <= 1:
        return 1, 0
    rng = np.random.default_rng(RNG_SEED)
    step = int(rng.integers(1, size))
    while math.gcd(step, size) != 1:
        step += 1
        if step >= size:
            step = 1
    offset = int(rng.integers(0, size))
    return step, offset


def _fill_permuted_ids(ids: np.ndarray, size: int, start: int, stop: int, step: int, offset: int) -> None:
    positions = np.arange(start, stop, dtype=np.int64)
    shuffled_positions = (positions * step + offset) % size
    ids[:] = ordered_ids_from_positions(shuffled_positions, size, ids.dtype)


def _randomized_ids(size: int, dtype: np.dtype) -> np.ndarray:
    ids = make_ordered_ids(size, dtype)
    np.random.default_rng(RNG_SEED).shuffle(ids)
    return ids


def parquet_path(
    outdir: Path,
    size: int,
    dist: str,
    id_dtype: np.dtype,
    layout: str,
    row_group_size: int,
    max_rows_per_page: int,
    compression: str,
) -> Path:
    return (
        outdir
        / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{DATASET_LAYOUT_VERSION}.layout-{layout}.rg-{row_group_size}.page-{max_rows_per_page}.codec-{compression}.parquet"
    )


def build_parquet_file(
    size: int,
    dist: str,
    id_dtype: np.dtype,
    path: Path,
    *,
    row_group_size: int,
    max_rows_per_page: int,
    compression: str,
    write_page_index: bool,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    schema = pa.schema([("id", pa.from_numpy_dtype(id_dtype)), ("payload", pa.float32())])
    block_order = _block_order(size, max_rows_per_page) if dist == "block-shuffled" else None
    permuted_step, permuted_offset = _permuted_position_params(size) if dist == "permuted" else (1, 0)
    random_ids = _randomized_ids(size, id_dtype) if dist == "random" else None

    start_time = time.perf_counter()
    writer = pq.ParquetWriter(
        path,
        schema,
        compression=compression,
        write_statistics=True,
        write_page_index=write_page_index,
        max_rows_per_page=max_rows_per_page,
    )
    try:
        for start in range(0, size, row_group_size):
            stop = min(start + row_group_size, size)
            ids = np.empty(stop - start, dtype=id_dtype)
            if dist == "sorted":
                ids[:] = ordered_id_slice(size, start, stop, id_dtype)
            elif dist == "block-shuffled":
                _fill_block_shuffled_ids(ids, size, start, stop, max_rows_per_page, block_order)
            elif dist == "permuted":
                _fill_permuted_ids(ids, size, start, stop, permuted_step, permuted_offset)
            elif dist == "random":
                ids[:] = random_ids[start:stop]
            else:
                raise ValueError(f"unsupported distribution {dist!r}")

            payload = payload_slice(start, stop)
            table = pa.table({"id": ids, "payload": payload}, schema=schema)
            writer.write_table(table, row_group_size=row_group_size)
    finally:
        writer.close()
    return time.perf_counter() - start_time


def _query_bounds(size: int, query_width: int, dtype: np.dtype) -> tuple[object, object]:
    lo_idx = size // 2
    hi_idx = min(size - 1, lo_idx + max(query_width - 1, 0))
    return ordered_id_at(size, lo_idx, dtype), ordered_id_at(size, hi_idx, dtype)


def benchmark_scan_once(path: Path, lo, hi) -> tuple[float, int]:
    start = time.perf_counter()
    table = pq.read_table(path, use_threads=True)
    ids = table["id"].to_numpy()
    mask = (ids >= lo) & (ids <= hi)
    result_len = int(np.count_nonzero(mask))
    elapsed = time.perf_counter() - start
    return elapsed, result_len


def benchmark_filtered_once(path: Path, lo, hi) -> tuple[float, int]:
    start = time.perf_counter()
    table = pq.read_table(path, filters=[("id", ">=", lo), ("id", "<=", hi)], use_threads=True)
    ids = table["id"].to_numpy()
    result_len = int(np.count_nonzero((ids >= lo) & (ids <= hi)))
    elapsed = time.perf_counter() - start
    return elapsed, result_len


def parquet_payload_bytes(path: Path) -> int:
    metadata = pq.ParquetFile(path).metadata
    payload = 0
    for row_group_idx in range(metadata.num_row_groups):
        row_group = metadata.row_group(row_group_idx)
        for column_idx in range(row_group.num_columns):
            payload += int(row_group.column(column_idx).total_compressed_size)
    return payload


def median(values: list[float]) -> float:
    return statistics.median(values)


def benchmark_layout(
    size: int,
    outdir: Path,
    dist: str,
    query_width: int,
    id_dtype: np.dtype,
    layout: str,
    row_group_size: int,
    max_rows_per_page: int,
    compression: str,
    repeats: int,
) -> dict:
    path = parquet_path(outdir, size, dist, id_dtype, layout, row_group_size, max_rows_per_page, compression)
    write_page_index = layout == "page-index"
    create_s = build_parquet_file(
        size,
        dist,
        id_dtype,
        path,
        row_group_size=row_group_size,
        max_rows_per_page=max_rows_per_page,
        compression=compression,
        write_page_index=write_page_index,
    )
    lo, hi = _query_bounds(size, query_width, id_dtype)

    scan_times = []
    filtered_times = []
    scan_rows = None
    filtered_rows = None
    for _ in range(repeats):
        scan_elapsed, scan_rows = benchmark_scan_once(path, lo, hi)
        filtered_elapsed, filtered_rows = benchmark_filtered_once(path, lo, hi)
        scan_times.append(scan_elapsed * 1_000)
        filtered_times.append(filtered_elapsed * 1_000)

    if scan_rows != filtered_rows:
        raise AssertionError(f"filtered rows mismatch: scan={scan_rows}, filtered={filtered_rows}")

    file_bytes = os.path.getsize(path)
    payload_bytes = parquet_payload_bytes(path)
    overhead_bytes = file_bytes - payload_bytes

    return {
        "size": size,
        "dist": dist,
        "layout": layout,
        "create_ms": create_s * 1_000,
        "scan_ms": median(scan_times),
        "filtered_ms": median(filtered_times),
        "speedup": median(scan_times) / median(filtered_times),
        "file_bytes": file_bytes,
        "payload_bytes": payload_bytes,
        "overhead_bytes": overhead_bytes,
        "payload_pct": (payload_bytes / file_bytes * 100) if file_bytes else 0.0,
        "overhead_pct": (overhead_bytes / file_bytes * 100) if file_bytes else 0.0,
        "query_rows": int(filtered_rows),
        "path": path,
    }


def print_results(
    results: list[dict],
    *,
    row_group_size: int,
    max_rows_per_page: int,
    repeats: int,
    dist: str,
    query_width: int,
    id_dtype: np.dtype,
    compression: str,
) -> None:
    print("Parquet range-query benchmark via pyarrow filtered reads")
    print(
        f"row_group_size={row_group_size:,}, max_rows_per_page={max_rows_per_page:,}, repeats={repeats}, "
        f"dist={dist}, query_width={query_width:,}, dtype={id_dtype.name}, compression={compression}"
    )
    print("Note: filtered reads are measured with pyarrow.parquet.read_table(filters=...).")
    print("      Pruning behavior depends on what the current PyArrow reader can exploit.")
    print()
    print(
        f"{'rows':<10} {'dist':<8} {'layout':<11} {'create_ms':>12} {'scan_ms':>9} {'filtered_ms':>12} "
        f"{'speedup':>9} {'file_bytes':>12} {'payload':>12} {'overhead':>12} {'query_rows':>11}"
    )
    print(
        f"{'-' * 10} {'-' * 8} {'-' * 11} {'-' * 12} {'-' * 9} {'-' * 12} {'-' * 9} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 11}"
    )
    for row in results:
        print(
            f"{row['size']:<10,} {row['dist']:<8} {row['layout']:<11} {row['create_ms']:12.3f} "
            f"{row['scan_ms']:9.3f} {row['filtered_ms']:12.3f} {row['speedup']:9.2f}x "
            f"{row['file_bytes']:12,} {row['payload_bytes']:12,} {row['overhead_bytes']:12,} {row['query_rows']:11,}"
        )


def parse_human_int(value: str) -> int:
    value = value.strip().lower().replace("_", "")
    multipliers = {"k": 1_000, "m": 1_000_000}
    if value[-1:] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="10M", help="Number of rows, or 'all'. Default: 10M.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for generated Parquet files.")
    parser.add_argument("--dist", choices=(*DISTS, "all"), default="permuted", help="Row distribution.")
    parser.add_argument("--layout", choices=(*LAYOUTS, "all"), default="all", help="Parquet layout to benchmark.")
    parser.add_argument("--query-width", type=parse_human_int, default=1, help="Query width. Default: 1.")
    parser.add_argument("--dtype", default="float64", help="Indexed id dtype. Default: float64.")
    parser.add_argument(
        "--row-group-size",
        type=parse_human_int,
        default=DEFAULT_ROW_GROUP_SIZE,
        help="Parquet row group size. Default: 1.25M.",
    )
    parser.add_argument(
        "--max-rows-per-page",
        type=parse_human_int,
        default=DEFAULT_MAX_ROWS_PER_PAGE,
        help="Parquet max rows per page. Default: 10k.",
    )
    parser.add_argument("--compression", default=DEFAULT_COMPRESSION, help="Parquet compression codec.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Benchmark repeats. Default: 3.")
    args = parser.parse_args()

    id_dtype = np.dtype(args.dtype)
    sizes = SIZES if args.size == "all" else (parse_human_int(args.size),)
    dists = DISTS if args.dist == "all" else (args.dist,)
    layouts = LAYOUTS if args.layout == "all" else (args.layout,)

    results = []
    for size in sizes:
        for dist in dists:
            for layout in layouts:
                results.append(
                    benchmark_layout(
                        size,
                        args.outdir,
                        dist,
                        args.query_width,
                        id_dtype,
                        layout,
                        args.row_group_size,
                        args.max_rows_per_page,
                        args.compression,
                        args.repeats,
                    )
                )

    print_results(
        results,
        row_group_size=args.row_group_size,
        max_rows_per_page=args.max_rows_per_page,
        repeats=args.repeats,
        dist=args.dist,
        query_width=args.query_width,
        id_dtype=id_dtype,
        compression=args.compression,
    )


if __name__ == "__main__":
    main()
