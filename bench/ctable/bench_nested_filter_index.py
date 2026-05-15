#!/usr/bin/env python
#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Benchmark nested leaf filter/index performance vs flat columns.

Compares a CTable with flat column names against an equivalent one that uses
dotted nested column names (physically stored under hierarchical _cols/ paths).
Both tables hold the same data; each filter/index/aggregate operation is timed
on both to show the overhead (or absence thereof) introduced by the nested layout.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass

import numpy as np

import blosc2


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


@dataclass
class FlatRow:
    trip_begin_lon: float = blosc2.field(blosc2.float64())
    trip_begin_lat: float = blosc2.field(blosc2.float64())
    trip_end_lon: float = blosc2.field(blosc2.float64())
    trip_end_lat: float = blosc2.field(blosc2.float64())
    payment_fare: float = blosc2.field(blosc2.float64(ge=0))


@dataclass
class NestedRow:
    """Same physical columns as FlatRow but accessed via dotted names after creation."""

    trip_begin_lon: float = blosc2.field(blosc2.float64())
    trip_begin_lat: float = blosc2.field(blosc2.float64())
    trip_end_lon: float = blosc2.field(blosc2.float64())
    trip_end_lat: float = blosc2.field(blosc2.float64())
    payment_fare: float = blosc2.field(blosc2.float64(ge=0))


def _build_data(n: int) -> dict:
    rng = np.random.default_rng(42)
    return {
        "trip_begin_lon": rng.uniform(-88.0, -87.5, n).astype(np.float64),
        "trip_begin_lat": rng.uniform(41.6, 42.0, n).astype(np.float64),
        "trip_end_lon": rng.uniform(-88.0, -87.5, n).astype(np.float64),
        "trip_end_lat": rng.uniform(41.6, 42.0, n).astype(np.float64),
        "payment_fare": rng.uniform(3.0, 50.0, n).astype(np.float64),
    }


def _build_flat(data: dict, n: int) -> "blosc2.CTable":
    t = blosc2.CTable(FlatRow, expected_size=n)
    t.extend(data)
    return t


def _build_nested(data: dict, n: int) -> "blosc2.CTable":
    t = blosc2.CTable(NestedRow, expected_size=n)
    t.extend(data)
    # Rename to dotted nested names
    t.rename_column("trip_begin_lon", "trip.begin.lon")
    t.rename_column("trip_begin_lat", "trip.begin.lat")
    t.rename_column("trip_end_lon", "trip.end.lon")
    t.rename_column("trip_end_lat", "trip.end.lat")
    t.rename_column("payment_fare", "payment.fare")
    return t


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _timeit(fn, repeats: int = 5) -> float:
    gc.collect()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark nested vs flat column filter/index/aggregate")
    p.add_argument("--rows", type=int, default=1_000_000, help="Number of rows (default: 1M)")
    p.add_argument("--repeats", type=int, default=5, help="Timing repeats (default: 5)")
    args = p.parse_args()

    N = args.rows
    R = args.repeats

    print(f"Building tables with {N:,} rows …")
    data = _build_data(N)
    flat_data = data.copy()  # flat uses underscore names
    nested_data = {
        "trip_begin_lon": data["trip_begin_lon"],
        "trip_begin_lat": data["trip_begin_lat"],
        "trip_end_lon": data["trip_end_lon"],
        "trip_end_lat": data["trip_end_lat"],
        "payment_fare": data["payment_fare"],
    }

    tf = _build_flat(flat_data, N)
    tn = _build_nested(nested_data, N)
    print(f"  flat   col_names: {tf.col_names}")
    print(f"  nested col_names: {tn.col_names}")
    print()

    # Build indexes on the fare column for index-accelerated queries
    print("Building indexes …")
    tf.create_index("payment_fare")
    tn.create_index("payment.fare")
    print()

    header = f"{'Operation':<45} {'flat (ms)':>12} {'nested (ms)':>13} {'ratio':>8}"
    print(header)
    print("-" * len(header))

    def bench(label, flat_fn, nested_fn):
        t_flat = _timeit(flat_fn, R) * 1000
        t_nested = _timeit(nested_fn, R) * 1000
        ratio = t_nested / t_flat if t_flat > 0 else float("nan")
        print(f"{label:<45} {t_flat:>12.3f} {t_nested:>13.3f} {ratio:>8.3f}x")

    bench(
        "where (string expr, full scan)",
        lambda: tf.where("payment_fare > 20"),
        lambda: tn.where("payment.fare > 20"),
    )

    bench(
        "where (string expr, full scan, nrows)",
        lambda: tf.where("payment_fare > 20").nrows,
        lambda: tn.where("payment.fare > 20").nrows,
    )

    bench(
        "where (LazyExpr, full scan)",
        lambda: tf.where(tf["payment_fare"] > 20),
        lambda: tn.where(tn["payment.fare"] > 20),
    )

    bench(
        "where (auto index-accelerated, nrows)",
        lambda: tf.where("payment_fare > 20").nrows,
        lambda: tn.where("payment.fare > 20").nrows,
    )

    bench(
        "column mean (full scan)",
        lambda: tf["payment_fare"].mean(),
        lambda: tn["payment.fare"].mean(),
    )

    bench(
        "column sum (full scan)",
        lambda: tf["payment_fare"].sum(),
        lambda: tn["payment.fare"].sum(),
    )

    bench(
        "column min (full scan)",
        lambda: tf["trip_begin_lon"].min(),
        lambda: tn["trip.begin.lon"].min(),
    )

    bench(
        "multi-column where (string expr, nrows)",
        lambda: tf.where("trip_begin_lon > -87.7 and payment_fare > 10").nrows,
        lambda: tn.where("trip.begin.lon > -87.7 and payment.fare > 10").nrows,
    )

    bench(
        "sort_by (single leaf)",
        lambda: tf.sort_by("payment_fare"),
        lambda: tn.sort_by("payment.fare"),
    )

    print()
    print("ratio < 1 means nested is faster; ratio > 1 means flat is faster.")
    print("Ratios close to 1.0 indicate the nested path adds negligible overhead.")


if __name__ == "__main__":
    main()
