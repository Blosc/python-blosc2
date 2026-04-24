#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Benchmark variable-length CTable columns.

Covers:
1. append / extend performance
2. full-scan query performance
3. getitem performance for single rows and small slices

Examples
--------
python bench/ctable/varlen.py --rows 200000 --batch-size 1000
python bench/ctable/varlen.py --rows 500000 --storages batch vl --repeats 5
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from dataclasses import dataclass

import blosc2


@dataclass
class RowBatch:
    id: int = blosc2.field(blosc2.int64())
    group: int = blosc2.field(blosc2.int32())
    score: float = blosc2.field(blosc2.float64())
    tags: list[str] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.string(max_length=24), nullable=True, storage="batch", batch_rows=1024)
    )


@dataclass
class RowVL:
    id: int = blosc2.field(blosc2.int64())
    group: int = blosc2.field(blosc2.int32())
    score: float = blosc2.field(blosc2.float64())
    tags: list[str] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.string(max_length=24), nullable=True, storage="vl")
    )


def make_row(i: int) -> tuple[int, int, float, list[str] | None]:
    group = i % 97
    score = float((i * 13) % 1000) / 10.0
    mod = i % 11
    if mod == 0:
        tags = None
    elif mod == 1:
        tags = []
    elif mod <= 4:
        tags = [f"t{i % 1000}"]
    elif mod <= 7:
        tags = [f"t{i % 1000}", f"g{group}"]
    else:
        tags = [f"t{i % 1000}", f"g{group}", f"s{int(score)}"]
    return i, group, score, tags


def make_rows(nrows: int) -> list[tuple[int, int, float, list[str] | None]]:
    return [make_row(i) for i in range(nrows)]


def choose_row_type(storage: str):
    if storage == "batch":
        return RowBatch
    if storage == "vl":
        return RowVL
    raise ValueError(f"Unsupported storage: {storage!r}")


def best_time(fn, *, repeats: int) -> float:
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best


def median_time(fn, *, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def build_table_by_append(row_type, rows) -> blosc2.CTable:
    t = blosc2.CTable(row_type, expected_size=len(rows))
    for row in rows:
        t.append(row)
    return t


def build_table_by_extend(row_type, rows, batch_size: int) -> blosc2.CTable:
    t = blosc2.CTable(row_type, expected_size=len(rows))
    for start in range(0, len(rows), batch_size):
        t.extend(rows[start : start + batch_size])
    return t


def query_count(table: blosc2.CTable) -> int:
    view = table.where((table.group >= 10) & (table.group < 50) & (table.score >= 25.0))
    return len(view)


def query_with_list_touch(table: blosc2.CTable) -> int:
    view = table.where((table.group >= 10) & (table.group < 50) & (table.score >= 25.0))
    total = 0
    for cell in view.tags:
        total += 0 if cell is None else len(cell)
    return total


def bench_getitem_single(table: blosc2.CTable, indices: list[int]) -> int:
    total = 0
    col = table.tags
    for idx in indices:
        cell = col[idx]
        total += 0 if cell is None else len(cell)
    return total


def bench_getitem_slices(table: blosc2.CTable, starts: list[int], width: int) -> int:
    total = 0
    col = table.tags
    for start in starts:
        cells = col[start : start + width]
        for cell in cells:
            total += 0 if cell is None else len(cell)
    return total


def format_rate(n: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    return f"{n / seconds:,.0f}/s"


def run_storage_bench(storage: str, rows, *, batch_size: int, repeats: int, nsamples: int, slice_width: int) -> None:
    row_type = choose_row_type(storage)
    print(f"\n=== storage={storage} ===")

    append_time = best_time(lambda: build_table_by_append(row_type, rows), repeats=repeats)
    extend_time = best_time(lambda: build_table_by_extend(row_type, rows, batch_size), repeats=repeats)

    table = build_table_by_extend(row_type, rows, batch_size)

    q1 = query_count(table)
    scan_count_time = median_time(lambda: query_count(table), repeats=repeats)

    q2 = query_with_list_touch(table)
    scan_touch_time = median_time(lambda: query_with_list_touch(table), repeats=repeats)

    max_start = max(1, len(table) - slice_width - 1)
    indices = [((i * 104729) % max_start) for i in range(nsamples)]
    starts = [((i * 8191) % max_start) for i in range(nsamples)]
    clustered_indices = [i % min(max_start, 4096) for i in range(nsamples)]
    clustered_starts = [i % min(max_start, 2048) for i in range(nsamples)]

    single_sum = bench_getitem_single(table, indices)
    single_time = median_time(lambda: bench_getitem_single(table, indices), repeats=repeats)
    single_clustered_time = median_time(
        lambda: bench_getitem_single(table, clustered_indices), repeats=repeats
    )

    slice_sum = bench_getitem_slices(table, starts, slice_width)
    slice_time = median_time(lambda: bench_getitem_slices(table, starts, slice_width), repeats=repeats)
    slice_clustered_time = median_time(
        lambda: bench_getitem_slices(table, clustered_starts, slice_width), repeats=repeats
    )

    print("append/extend")
    print(f"  append rows:     {append_time:8.4f} s   {format_rate(len(rows), append_time)}")
    print(f"  extend rows:     {extend_time:8.4f} s   {format_rate(len(rows), extend_time)}")
    print(f"  append/extend:   {append_time / extend_time:8.2f}x slower")

    print("scan queries")
    print(f"  count only:      {scan_count_time:8.4f} s   matches={q1:,}")
    print(f"  count+list use:  {scan_touch_time:8.4f} s   payload={q2:,}")

    print("getitem")
    print(
        f"  single row random:   {single_time:8.4f} s   "
        f"{format_rate(nsamples, single_time)}   checksum={single_sum}"
    )
    print(
        f"  single row local:    {single_clustered_time:8.4f} s   "
        f"{format_rate(nsamples, single_clustered_time)}"
    )
    print(
        f"  slice[{slice_width}] random: {slice_time:8.4f} s   "
        f"{format_rate(nsamples, slice_time)}   checksum={slice_sum}"
    )
    print(
        f"  slice[{slice_width}] local:  {slice_clustered_time:8.4f} s   "
        f"{format_rate(nsamples, slice_clustered_time)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=200_000, help="Number of rows to generate.")
    parser.add_argument("--batch-size", type=int, default=1_000, help="Batch size used for extend().")
    parser.add_argument("--repeats", type=int, default=5, help="Repetitions per benchmark.")
    parser.add_argument(
        "--storages",
        nargs="+",
        choices=("batch", "vl"),
        default=["batch", "vl"],
        help="List-column storage backends to benchmark.",
    )
    parser.add_argument(
        "--getitem-samples",
        type=int,
        default=20_000,
        help="Number of random single-row / slice probes.",
    )
    parser.add_argument("--slice-width", type=int, default=8, help="Width of small-slice getitem benchmark.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = make_rows(args.rows)

    print("CTable variable-length column benchmark")
    print(f"rows={args.rows:,}  batch_size={args.batch_size:,}  repeats={args.repeats}")
    print(f"getitem_samples={args.getitem_samples:,}  slice_width={args.slice_width}")

    for storage in args.storages:
        run_storage_bench(
            storage,
            rows,
            batch_size=args.batch_size,
            repeats=args.repeats,
            nsamples=args.getitem_samples,
            slice_width=args.slice_width,
        )


if __name__ == "__main__":
    main()
