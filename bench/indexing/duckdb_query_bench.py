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

import duckdb
import numpy as np
import pyarrow as pa

SIZES = (1_000_000, 2_000_000, 5_000_000, 10_000_000)
DEFAULT_REPEATS = 3
DISTS = ("sorted", "block-shuffled", "permuted", "random")
LAYOUTS = ("zonemap", "art-index")
RNG_SEED = 0
DEFAULT_BATCH_SIZE = 1_250_000
DATASET_LAYOUT_VERSION = "payload-ramp-v1"

COLD_COLUMNS = [
    ("rows", lambda result: f"{result['size']:,}"),
    ("dist", lambda result: result["dist"]),
    ("layout", lambda result: result["layout"]),
    ("create_ms", lambda result: f"{result['create_ms']:.3f}"),
    ("scan_ms", lambda result: f"{result['cold_scan_ms']:.3f}"),
    ("query_ms", lambda result: f"{result['cold_ms']:.3f}"),
    ("speedup", lambda result: f"{result['cold_speedup']:.2f}x"),
    ("db_bytes", lambda result: f"{result['db_bytes']:,}"),
    ("query_rows", lambda result: f"{result['query_rows']:,}"),
]

WARM_COLUMNS = [
    ("rows", lambda result: f"{result['size']:,}"),
    ("dist", lambda result: result["dist"]),
    ("layout", lambda result: result["layout"]),
    ("create_ms", lambda result: f"{result['create_ms']:.3f}"),
    ("scan_ms", lambda result: f"{result['warm_scan_ms']:.3f}"),
    ("query_ms", lambda result: f"{result['warm_ms']:.3f}" if result["warm_ms"] is not None else "-"),
    ("speedup", lambda result: f"{result['warm_speedup']:.2f}x" if result["warm_speedup"] is not None else "-"),
    ("db_bytes", lambda result: f"{result['db_bytes']:,}"),
    ("query_rows", lambda result: f"{result['query_rows']:,}"),
]


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


def duckdb_sql_type(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return "BOOLEAN"
    if dtype == np.dtype(np.int8):
        return "TINYINT"
    if dtype == np.dtype(np.int16):
        return "SMALLINT"
    if dtype == np.dtype(np.int32):
        return "INTEGER"
    if dtype == np.dtype(np.int64):
        return "BIGINT"
    if dtype == np.dtype(np.uint8):
        return "UTINYINT"
    if dtype == np.dtype(np.uint16):
        return "USMALLINT"
    if dtype == np.dtype(np.uint32):
        return "UINTEGER"
    if dtype == np.dtype(np.uint64):
        return "UBIGINT"
    if dtype == np.dtype(np.float32):
        return "REAL"
    if dtype == np.dtype(np.float64):
        return "DOUBLE"
    raise ValueError(f"unsupported duckdb dtype: {dtype}")


def duckdb_path(outdir: Path, size: int, dist: str, id_dtype: np.dtype, layout: str, batch_size: int) -> Path:
    return (
        outdir
        / f"size_{size}_{dist}_{dtype_token(id_dtype)}.{DATASET_LAYOUT_VERSION}.layout-{layout}.batch-{batch_size}.duckdb"
    )


def _duckdb_wal_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.wal")


def _remove_duckdb_path(path: Path) -> None:
    if path.exists():
        path.unlink()
    wal_path = _duckdb_wal_path(path)
    if wal_path.exists():
        wal_path.unlink()


def _valid_duckdb_file(path: Path, layout: str) -> bool:
    if not path.exists():
        return False

    con = None
    try:
        con = duckdb.connect(str(path), read_only=True)
        has_data = bool(
            con.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = 'data'"
            ).fetchone()[0]
        )
        if not has_data:
            return False
        if layout == "art-index":
            index_count = con.execute(
                "SELECT COUNT(*) FROM duckdb_indexes() WHERE schema_name = 'main' AND table_name = 'data' "
                "AND index_name = 'data_id_idx'"
            ).fetchone()[0]
            return bool(index_count)
        return layout == "zonemap"
    except duckdb.Error:
        return False
    finally:
        if con is not None:
            con.close()


def build_duckdb_file(
    size: int,
    dist: str,
    id_dtype: np.dtype,
    path: Path,
    *,
    layout: str,
    batch_size: int,
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_duckdb_path(path)

    id_type = duckdb_sql_type(id_dtype)
    block_order = _block_order(size, batch_size) if dist == "block-shuffled" else None
    permuted_step, permuted_offset = _permuted_position_params(size) if dist == "permuted" else (1, 0)
    random_ids = _randomized_ids(size, id_dtype) if dist == "random" else None

    start_time = time.perf_counter()
    con = duckdb.connect(str(path))
    try:
        con.execute("PRAGMA threads=8")
        con.execute(f"CREATE TABLE data (id {id_type}, payload FLOAT)")
        for start in range(0, size, batch_size):
            stop = min(start + batch_size, size)
            ids = np.empty(stop - start, dtype=id_dtype)
            if dist == "sorted":
                ids[:] = ordered_id_slice(size, start, stop, id_dtype)
            elif dist == "block-shuffled":
                _fill_block_shuffled_ids(ids, size, start, stop, batch_size, block_order)
            elif dist == "permuted":
                _fill_permuted_ids(ids, size, start, stop, permuted_step, permuted_offset)
            elif dist == "random":
                ids[:] = random_ids[start:stop]
            else:
                raise ValueError(f"unsupported distribution {dist!r}")

            payload = payload_slice(start, stop)
            batch = pa.table({"id": ids, "payload": payload})
            con.register("batch_arrow", batch)
            con.execute("INSERT INTO data SELECT * FROM batch_arrow")
            con.unregister("batch_arrow")

        if layout == "art-index":
            con.execute("CREATE INDEX data_id_idx ON data(id)")
        elif layout != "zonemap":
            raise ValueError(f"unsupported layout {layout!r}")

        con.execute("CHECKPOINT")
    finally:
        con.close()
    return time.perf_counter() - start_time


def _open_or_build_duckdb_file(
    size: int,
    dist: str,
    id_dtype: np.dtype,
    path: Path,
    *,
    layout: str,
    batch_size: int,
) -> float:
    if _valid_duckdb_file(path, layout):
        return 0.0
    return build_duckdb_file(size, dist, id_dtype, path, layout=layout, batch_size=batch_size)


def _query_bounds(size: int, query_width: int, dtype: np.dtype) -> tuple[object, object]:
    lo_idx = size // 2
    hi_idx = min(size - 1, lo_idx + max(query_width - 1, 0))
    return ordered_id_at(size, lo_idx, dtype), ordered_id_at(size, hi_idx, dtype)


def _literal(value: object, dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.bool_):
        return "TRUE" if bool(value) else "FALSE"
    if dtype.kind == "f":
        return repr(float(value))
    if dtype.kind in {"i", "u"}:
        return str(int(value))
    raise ValueError(f"unsupported dtype for literal formatting: {dtype}")


def _condition_sql(lo: object, hi: object, dtype: np.dtype, *, exact_query: bool = False) -> str:
    if exact_query:
        if lo != hi:
            raise ValueError(f"exact queries require a single lookup value, got lo={lo!r}, hi={hi!r}")
        return f"id = {_literal(lo, dtype)}"
    return f"id >= {_literal(lo, dtype)} AND id <= {_literal(hi, dtype)}"


def benchmark_scan_once(path: Path, lo, hi, dtype: np.dtype, *, exact_query: bool = False) -> tuple[float, float, float, int]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        condition_sql = _condition_sql(lo, hi, dtype, exact_query=exact_query)
        # Force the filtered baseline down the table-scan path instead of the ART index path.
        con.execute("SET index_scan_max_count = 0")
        con.execute("SET index_scan_percentage = 0")
        query = f"SELECT * FROM data WHERE {condition_sql}"

        cold_start = time.perf_counter()
        table = con.execute(query).arrow().read_all()
        cold_elapsed = time.perf_counter() - cold_start

        start = time.perf_counter()
        table = con.execute(query).arrow().read_all()
        result_len = len(table)
        warm_elapsed = time.perf_counter() - start

        third_start = time.perf_counter()
        con.execute(query).arrow().read_all()
        third_elapsed = time.perf_counter() - third_start
        return cold_elapsed, warm_elapsed, third_elapsed, result_len
    finally:
        con.close()


def benchmark_filtered_once(path: Path, lo, hi, dtype: np.dtype, *, exact_query: bool = False) -> tuple[float, int]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        condition_sql = _condition_sql(lo, hi, dtype, exact_query=exact_query)
        start = time.perf_counter()
        table = con.execute(f"SELECT * FROM data WHERE {condition_sql}").arrow().read_all()
        ids = table["id"].to_numpy()
        result_len = int(np.count_nonzero((ids >= lo) & (ids <= hi)))
        elapsed = time.perf_counter() - start
        return elapsed, result_len
    finally:
        con.close()


def benchmark_filtered_once_con(
    con: duckdb.DuckDBPyConnection, lo, hi, dtype: np.dtype, *, exact_query: bool = False
) -> tuple[float, int]:
    condition_sql = _condition_sql(lo, hi, dtype, exact_query=exact_query)
    start = time.perf_counter()
    table = con.execute(f"SELECT * FROM data WHERE {condition_sql}").arrow().read_all()
    ids = table["id"].to_numpy()
    result_len = int(np.count_nonzero((ids >= lo) & (ids <= hi)))
    elapsed = time.perf_counter() - start
    return elapsed, result_len


def median(values: list[float]) -> float:
    return statistics.median(values)


def benchmark_layout(
    size: int,
    outdir: Path,
    dist: str,
    query_width: int,
    id_dtype: np.dtype,
    layout: str,
    batch_size: int,
    repeats: int,
    exact_query: bool,
) -> dict:
    path = duckdb_path(outdir, size, dist, id_dtype, layout, batch_size)
    create_s = _open_or_build_duckdb_file(size, dist, id_dtype, path, layout=layout, batch_size=batch_size)
    lo, hi = _query_bounds(size, query_width, id_dtype)

    cold_scan_elapsed, warm_scan_elapsed, third_scan_elapsed, scan_rows = benchmark_scan_once(
        path, lo, hi, id_dtype, exact_query=exact_query
    )

    con = duckdb.connect(str(path), read_only=True)
    try:
        cold_elapsed, filtered_rows = benchmark_filtered_once_con(con, lo, hi, id_dtype, exact_query=exact_query)
        warm_times = [
            benchmark_filtered_once_con(con, lo, hi, id_dtype, exact_query=exact_query)[0] * 1_000
            for _ in range(repeats)
        ]
    finally:
        con.close()

    if scan_rows != filtered_rows:
        raise AssertionError(f"filtered rows mismatch: scan={scan_rows}, filtered={filtered_rows}")

    cold_scan_ms = cold_scan_elapsed * 1_000
    warm_scan_ms = warm_scan_elapsed * 1_000
    cold_ms = cold_elapsed * 1_000
    warm_ms = median(warm_times) if warm_times else None
    if layout == "zonemap":
        cold_ms = third_scan_elapsed * 1_000

    return {
        "size": size,
        "dist": dist,
        "layout": layout,
        "create_ms": create_s * 1_000,
        "cold_scan_ms": cold_scan_ms,
        "warm_scan_ms": warm_scan_ms,
        "cold_ms": cold_ms,
        "cold_speedup": cold_scan_ms / cold_ms,
        "warm_ms": warm_ms,
        "warm_speedup": None if warm_ms is None else warm_scan_ms / warm_ms,
        "db_bytes": os.path.getsize(path),
        "query_rows": int(filtered_rows),
        "path": path,
    }


def parse_human_int(value: str) -> int:
    value = value.strip().lower().replace("_", "")
    multipliers = {"k": 1_000, "m": 1_000_000}
    if value[-1:] in multipliers:
        return int(float(value[:-1]) * multipliers[value[-1]])
    return int(value)


def print_results(
    results: list[dict],
    *,
    batch_size: int,
    repeats: int,
    dist: str,
    query_width: int,
    id_dtype: np.dtype,
    exact_query: bool,
) -> None:
    print("DuckDB range-query benchmark via SQL filtered reads")
    print(
        f"batch_size={batch_size:,}, repeats={repeats}, dist={dist}, query_width={query_width:,}, "
        f"dtype={id_dtype.name}, query_single_value={exact_query}"
    )
    print("Note: 'zonemap' is DuckDB's default table layout with automatic min/max pruning.")
    print("      'art-index' adds an explicit secondary index on id.")
    if exact_query:
        print("      Filter predicate uses `id = value`.")
    else:
        print("      Filter predicate uses `id >= lo AND id <= hi`.")
    cold_widths = table_widths(results, COLD_COLUMNS)
    print()
    print("Cold Query Table")
    print_table(results, COLD_COLUMNS, cold_widths)
    if repeats > 0:
        warm_widths = table_widths(results, WARM_COLUMNS)
        shared_width_by_header = {}
        for (header, _), width in zip(COLD_COLUMNS, cold_widths, strict=True):
            shared_width_by_header[header] = width
        for (header, _), width in zip(WARM_COLUMNS, warm_widths, strict=True):
            shared_width_by_header[header] = max(shared_width_by_header.get(header, 0), width)
        warm_widths = [shared_width_by_header[header] for header, _ in WARM_COLUMNS]
        print()
        print("Warm Query Table")
        print_table(results, WARM_COLUMNS, warm_widths)


def _format_row(cells: list[str], widths: list[int]) -> str:
    return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths, strict=True))


def _table_rows(results: list[dict], columns: list[tuple[str, callable]]) -> tuple[list[str], list[list[str]], list[int]]:
    headers = [header for header, _ in columns]
    widths = [len(header) for header in headers]
    rows = [[formatter(result) for _, formatter in columns] for result in results]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    return headers, rows, widths


def print_table(results: list[dict], columns: list[tuple[str, callable]], widths: list[int] | None = None) -> None:
    headers, rows, computed_widths = _table_rows(results, columns)
    widths = computed_widths if widths is None else widths
    print(_format_row(headers, widths))
    print(_format_row(["-" * width for width in widths], widths))
    for row in rows:
        print(_format_row(row, widths))


def table_widths(results: list[dict], columns: list[tuple[str, callable]]) -> list[int]:
    _, _, widths = _table_rows(results, columns)
    return widths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", default="10M", help="Number of rows, or 'all'. Default: 10M.")
    parser.add_argument("--outdir", type=Path, required=True, help="Directory for generated DuckDB files.")
    parser.add_argument("--dist", choices=(*DISTS, "all"), default="permuted", help="Row distribution.")
    parser.add_argument("--layout", choices=(*LAYOUTS, "all"), default="all", help="DuckDB layout to benchmark.")
    parser.add_argument("--query-width", type=parse_human_int, default=1, help="Query width. Default: 1.")
    parser.add_argument(
        "--query-single-value",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use `id = value` instead of a range predicate. Requires query-width=1.",
    )
    parser.add_argument("--dtype", default="float64", help="Indexed id dtype. Default: float64.")
    parser.add_argument(
        "--batch-size",
        type=parse_human_int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size used while loading the table. Default: 1.25M.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, help="Benchmark repeats. Default: 3.")
    args = parser.parse_args()

    if args.query_single_value and args.query_width != 1:
        raise ValueError("--query-single-value requires --query-width 1")

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
                        args.batch_size,
                        args.repeats,
                        args.query_single_value,
                    )
                )

    print_results(
        results,
        batch_size=args.batch_size,
        repeats=args.repeats,
        dist=args.dist,
        query_width=args.query_width,
        id_dtype=id_dtype,
        exact_query=args.query_single_value,
    )


if __name__ == "__main__":
    main()
