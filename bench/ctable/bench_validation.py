#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: cost of constraint validation
#
# Measures the overhead of validate=True vs validate=False for:
#   1. append()  — row-by-row, Pydantic path
#   2. extend()  — bulk insert, vectorized NumPy path
#
# at increasing batch sizes to show how validation cost scales.

from dataclasses import dataclass
from time import perf_counter

import numpy as np

import blosc2


@dataclass
class Row:
    id:     int   = blosc2.field(blosc2.int64(ge=0))
    score:  float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool  = blosc2.field(blosc2.bool(), default=True)


def make_data(n: int):
    rng = np.random.default_rng(42)
    ids    = np.arange(n, dtype=np.int64)
    scores = rng.uniform(0, 100, n)
    flags  = rng.integers(0, 2, n, dtype=np.bool_)
    return list(zip(ids.tolist(), scores.tolist(), flags.tolist()))


SIZES = [100, 1_000, 10_000, 100_000, 1_000_000]
APPEND_SIZES = [100, 1_000]   # append row-by-row is slow at large N

# ─────────────────────────────────────────────────────────────────────────────
# 1. append() — validate=True vs validate=False
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("1. append()  —  row-by-row  (Pydantic validation per row)")
print("=" * 65)
print(f"{'N':>10}  {'validate=True':>14}  {'validate=False':>15}  {'overhead':>10}")
print("-" * 65)

for n in APPEND_SIZES:
    data = make_data(n)

    t = blosc2.CTable(Row, expected_size=n, validate=True)
    t0 = perf_counter()
    for row in data:
        t.append(row)
    t_on = perf_counter() - t0

    t = blosc2.CTable(Row, expected_size=n, validate=False)
    t0 = perf_counter()
    for row in data:
        t.append(row)
    t_off = perf_counter() - t0

    overhead = (t_on / t_off) if t_off > 0 else float("inf")
    print(f"{n:>10,}  {t_on:>13.4f}s  {t_off:>14.4f}s  {overhead:>9.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# 2. extend() — validate=True vs validate=False
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("2. extend()  —  bulk insert  (vectorized NumPy validation)")
print("=" * 65)
print(f"{'N':>10}  {'validate=True':>14}  {'validate=False':>15}  {'overhead':>10}")
print("-" * 65)

for n in SIZES:
    data = make_data(n)

    t = blosc2.CTable(Row, expected_size=n, validate=True)
    t0 = perf_counter()
    t.extend(data)
    t_on = perf_counter() - t0

    t = blosc2.CTable(Row, expected_size=n, validate=False)
    t0 = perf_counter()
    t.extend(data)
    t_off = perf_counter() - t0

    overhead = (t_on / t_off) if t_off > 0 else float("inf")
    print(f"{n:>10,}  {t_on:>13.4f}s  {t_off:>14.4f}s  {overhead:>9.2f}x")

# ─────────────────────────────────────────────────────────────────────────────
# 3. extend() — validate=True vs validate=False with structured NumPy array
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("3. extend() with structured NumPy array")
print("=" * 65)
print(f"{'N':>10}  {'validate=True':>14}  {'validate=False':>15}  {'overhead':>10}")
print("-" * 65)

np_dtype = np.dtype([("id", np.int64), ("score", np.float64), ("active", np.bool_)])

for n in SIZES:
    rng = np.random.default_rng(42)
    arr = np.empty(n, dtype=np_dtype)
    arr["id"]     = np.arange(n, dtype=np.int64)
    arr["score"]  = rng.uniform(0, 100, n)
    arr["active"] = rng.integers(0, 2, n, dtype=np.bool_)

    t = blosc2.CTable(Row, expected_size=n, validate=True)
    t0 = perf_counter()
    t.extend(arr)
    t_on = perf_counter() - t0

    t = blosc2.CTable(Row, expected_size=n, validate=False)
    t0 = perf_counter()
    t.extend(arr)
    t_off = perf_counter() - t0

    overhead = (t_on / t_off) if t_off > 0 else float("inf")
    print(f"{n:>10,}  {t_on:>13.4f}s  {t_off:>14.4f}s  {overhead:>9.2f}x")

print()
print("Note: 'overhead' = validate=True time / validate=False time.")
print("      1.00x means validation is free; 2.00x means it doubles the time.")
