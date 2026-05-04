#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: append() row-by-row vs extend() bulk insert.
#
# Compares three strategies at increasing N to find where extend() wins:
#   1. append() x N        — one call per row, Pydantic path
#   2. extend() x N        — extend([row]) per row, one at a time
#   3. extend() x 1        — single bulk call with all N rows

from dataclasses import dataclass
from time import perf_counter

import blosc2


@dataclass
class Row:
    id:     int     = blosc2.field(blosc2.int64(ge=0))
    score:  float   = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool    = blosc2.field(blosc2.bool(), default=True)


SIZES = [10, 100, 1_000, 10_000, 100_000]

print(f"append() vs extend()  |  sizes: {SIZES}")
print()
print(f"{'N':>10}  {'append×N (s)':>14}  {'extend×N (s)':>14}  {'extend×1 (s)':>14}  {'speedup bulk':>13}")
print(f"{'─'*10}  {'─'*14}  {'─'*14}  {'─'*14}  {'─'*13}")

for N in SIZES:
    data = [[i, float(i % 100), i % 2 == 0] for i in range(N)]

    ct = blosc2.CTable(Row, expected_size=N)
    t0 = perf_counter()
    for row in data:
        ct.append(row)
    t_append = perf_counter() - t0

    ct = blosc2.CTable(Row, expected_size=N)
    t0 = perf_counter()
    for row in data:
        ct.extend([row])
    t_extend_one = perf_counter() - t0

    ct = blosc2.CTable(Row, expected_size=N)
    t0 = perf_counter()
    ct.extend(data)
    t_extend_bulk = perf_counter() - t0

    speedup = t_append / t_extend_bulk if t_extend_bulk > 0 else float("inf")
    print(f"{N:>10,}  {t_append:>14.6f}  {t_extend_one:>14.6f}  {t_extend_bulk:>14.6f}  {speedup:>12.1f}×")

print()
print("speedup bulk = append×N time / extend×1 time (higher is better for extend)")
