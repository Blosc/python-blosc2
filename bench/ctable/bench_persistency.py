#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: persistent vs in-memory CTable
#
# Sections:
#   1. extend() — bulk creation: in-memory vs file-backed
#   2. open()   — time to reopen an existing persistent table
#   3. append() — single-row append: in-memory vs file-backed (after reopen)
#   4. column read — materialising a full column: in-memory vs file-backed
#
# Each measurement is the minimum of NRUNS repetitions to reduce noise.

import os
import shutil
from dataclasses import dataclass
from time import perf_counter

import blosc2

NRUNS = 3
TABLE_DIR = "saved_ctable/bench"


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


def sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def tmin(fn, n: int = NRUNS) -> float:
    """Return the minimum elapsed time (seconds) over *n* calls of *fn*."""
    best = float("inf")
    for _ in range(n):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


def clean() -> None:
    if os.path.exists(TABLE_DIR):
        shutil.rmtree(TABLE_DIR)
    os.makedirs(TABLE_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Section 1: bulk creation — extend()
# ---------------------------------------------------------------------------

sep("1. extend() — bulk insert: in-memory vs TreeStore-backed")

SIZES = [1_000, 10_000, 100_000, 1_000_000]

print(f"{'rows':>12}  {'in-memory (s)':>16}  {'store-backed (s)':>16}  {'overhead':>10}")
print(f"{'----':>12}  {'-------------':>16}  {'---------------':>16}  {'--------':>10}")

for N in SIZES:
    data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]

    def bench_mem(N=N, data=data):
        t = blosc2.CTable(Row, expected_size=N)
        t.extend(data, validate=False)

    def bench_file(N=N, data=data):
        clean()
        t = blosc2.CTable(Row, urlpath=TABLE_DIR + "/ext", mode="w", expected_size=N)
        t.extend(data, validate=False)
        t.close()

    t_mem = tmin(bench_mem)
    t_file = tmin(bench_file)
    overhead = t_file / t_mem if t_mem > 0 else float("nan")
    print(f"{N:>12,}  {t_mem:>16.4f}  {t_file:>16.4f}  {overhead:>9.2f}x")

# ---------------------------------------------------------------------------
# Section 2: open() — reopen an existing table
# ---------------------------------------------------------------------------

sep("2. open() — time to reopen a persistent table")

print(f"{'rows':>12}  {'blosc2.open() (s)':>18}  {'CTable.open() (s)':>20}  {'CTable(..., mode=a) (s)':>24}")
print(f"{'----':>12}  {'----------------':>18}  {'------------------':>20}  {'------------------------':>24}")

for N in SIZES:
    data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]
    clean()
    path = TABLE_DIR + "/reopen"
    t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=N)
    t.extend(data, validate=False)
    t.close()

    def bench_blosc2_open(path=path):
        t2 = blosc2.open(path, mode="r")
        _ = len(t2)

    def bench_open(path=path):
        t2 = blosc2.CTable.open(path, mode="r")
        _ = len(t2)

    def bench_ctor(path=path):
        t2 = blosc2.CTable(Row, urlpath=path, mode="a")
        _ = len(t2)

    t_b2_open = tmin(bench_blosc2_open)
    t_open = tmin(bench_open)
    t_ctor = tmin(bench_ctor)
    print(f"{N:>12,}  {t_b2_open:>18.4f}  {t_open:>20.4f}  {t_ctor:>24.4f}")

# ---------------------------------------------------------------------------
# Section 3: append() — single-row inserts after reopen
# ---------------------------------------------------------------------------

sep("3. append() — 1 000 single-row inserts: in-memory vs TreeStore-backed")

APPEND_N = 1_000
PREALLOCATE = 10_000  # avoid resize noise

print(f"{'backend':>14}  {'total (s)':>12}  {'µs / row':>12}")
print(f"{'-------':>14}  {'---------':>12}  {'--------':>12}")


def bench_append_mem():
    t = blosc2.CTable(Row, expected_size=PREALLOCATE, validate=False)
    for i in range(APPEND_N):
        t.append((i, float(i % 100), True))


clean()
path = TABLE_DIR + "/apath"
blosc2.CTable(Row, urlpath=path, mode="w", expected_size=PREALLOCATE)


def bench_append_file():
    t = blosc2.CTable(Row, urlpath=path, mode="a", validate=False)
    for i in range(APPEND_N):
        t.append((i, float(i % 100), True))


for label, fn in [("in-memory", bench_append_mem), ("file-backed", bench_append_file)]:
    # Reset file table before each run
    if label == "file-backed":
        clean()
        t = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=PREALLOCATE)
        t.close()
    elapsed = tmin(fn)
    us_per_row = elapsed / APPEND_N * 1e6
    print(f"{label:>14}  {elapsed:>12.4f}  {us_per_row:>12.1f}")

# ---------------------------------------------------------------------------
# Section 4: column read — to_numpy() after reopen
# ---------------------------------------------------------------------------

sep("4. column read — to_numpy() on 'id': in-memory vs TreeStore-backed")

print(f"{'rows':>12}  {'in-memory (s)':>16}  {'store-backed (s)':>16}  {'ratio':>8}")
print(f"{'----':>12}  {'-------------':>16}  {'---------------':>16}  {'-----':>8}")

for N in SIZES:
    data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]

    t_mem_table = blosc2.CTable(Row, expected_size=N, validate=False)
    t_mem_table.extend(data, validate=False)

    clean()
    path = TABLE_DIR + "/read"
    t_file_table = blosc2.CTable(Row, urlpath=path, mode="w", expected_size=N)
    t_file_table.extend(data, validate=False)
    t_file_table.close()
    # Reopen read-only (simulates a real read workload)
    t_ro = blosc2.CTable.open(path, mode="r")

    def bench_read_mem(t=t_mem_table):
        _ = t["id"].to_numpy()

    def bench_read_file(t=t_ro):
        _ = t["id"].to_numpy()

    t_m = tmin(bench_read_mem)
    t_f = tmin(bench_read_file)
    ratio = t_f / t_m if t_m > 0 else float("nan")
    print(f"{N:>12,}  {t_m:>16.4f}  {t_f:>16.4f}  {ratio:>7.2f}x")

# Cleanup
clean()
print()
