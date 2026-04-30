#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: sort_by() performance.
#
# Sections:
#   1. Single-column sort at increasing N (random data)
#   2. Multi-column sort (1, 2, 3 keys) at fixed N
#   3. Already-sorted vs random vs reverse-sorted input
#   4. sort_by() with deletions (holes in _valid_rows)
#   5. inplace=True vs inplace=False

from dataclasses import dataclass
from time import perf_counter

import numpy as np

import blosc2


@dataclass
class Row:
    sensor_id:   int   = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float64())
    region:      int   = blosc2.field(blosc2.int32())
    active:      bool  = blosc2.field(blosc2.bool())

np_dtype = np.dtype([
    ("sensor_id",   np.int32),
    ("temperature", np.float64),
    ("region",      np.int32),
    ("active",      np.bool_),
])

rng = np.random.default_rng(42)

W = 70

def sep(title):
    print(f"\n{'─' * W}")
    print(f"  {title}")
    print(f"{'─' * W}")

def make_table(n, sensor_ids=None):
    data = np.empty(n, dtype=np_dtype)
    data["sensor_id"]   = rng.integers(0, n // 10, size=n, dtype=np.int32) if sensor_ids is None else sensor_ids
    data["temperature"] = 15.0 + rng.random(n) * 25
    data["region"]      = rng.integers(0, 8, size=n, dtype=np.int32)
    data["active"]      = rng.integers(0, 2, size=n, dtype=np.bool_)
    ct = blosc2.CTable(Row, expected_size=n)
    ct.extend(data)
    return ct


# ── 1. Single-column sort at increasing N ────────────────────────────────────

sep("1. Single-column sort  (sensor_id, random input)")
print(f"  {'N':>12}  {'TIME (s)':>10}  {'ms/Mrow':>10}")
print(f"  {'─'*12}  {'─'*10}  {'─'*10}")

for n in [10_000, 100_000, 500_000, 1_000_000]:
    ct = make_table(n)
    t0 = perf_counter()
    ct.sort_by(["sensor_id"], inplace=True)
    elapsed = perf_counter() - t0
    ms_per_mrow = elapsed / n * 1e9 / 1e3
    print(f"  {n:>12,}  {elapsed:>10.4f}  {ms_per_mrow:>10.1f}")


# ── 2. Multi-column sort at fixed N ──────────────────────────────────────────

N = 1_000_000
sep(f"2. Multi-column sort  (N={N:,})")
print(f"  {'KEYS':<30}  {'TIME (s)':>10}  {'SPEEDUP vs 1-key':>18}")
print(f"  {'─'*30}  {'─'*10}  {'─'*18}")

ct_base = make_table(N)
results = {}

for keys in [
    ["sensor_id"],
    ["sensor_id", "region"],
    ["sensor_id", "region", "active"],
]:
    ct = make_table(N)
    t0 = perf_counter()
    ct.sort_by(keys, inplace=True)
    elapsed = perf_counter() - t0
    results[len(keys)] = elapsed
    spd = results[1] / elapsed if elapsed > 0 else float("inf")
    label = " + ".join(keys)
    print(f"  {label:<30}  {elapsed:>10.4f}  {spd:>17.2f}×")


# ── 3. Input order: random vs sorted vs reverse ───────────────────────────────

sep(f"3. Input order effect  (N={N:,}, sort by sensor_id)")
print(f"  {'INPUT ORDER':<20}  {'TIME (s)':>10}")
print(f"  {'─'*20}  {'─'*10}")

sid_max = N // 10

rand_ids    = rng.integers(0, sid_max, size=N, dtype=np.int32)
sorted_ids  = np.repeat(np.arange(sid_max, dtype=np.int32), N // sid_max)
reverse_ids = sorted_ids[::-1].copy()

for label, ids in [
    ("random",   rand_ids),
    ("sorted",   sorted_ids),
    ("reversed", reverse_ids),
]:
    ct = make_table(N, sensor_ids=ids)
    t0 = perf_counter()
    ct.sort_by(["sensor_id"], inplace=True)
    elapsed = perf_counter() - t0
    print(f"  {label:<20}  {elapsed:>10.4f}")


# ── 4. Sort with deletions (holes) ────────────────────────────────────────────

sep(f"4. Sort with deletions  (N={N:,}, sort by sensor_id)")
print(f"  {'DELETED':>10}  {'LIVE ROWS':>10}  {'TIME (s)':>10}")
print(f"  {'─'*10}  {'─'*10}  {'─'*10}")

for frac in [0.0, 0.1, 0.25, 0.5]:
    ct = make_table(N)
    n_del = int(N * frac)
    if n_del:
        ct.delete(list(range(0, N, max(1, N // n_del)))[:n_del])
    live = len(ct)
    t0 = perf_counter()
    ct.sort_by(["sensor_id"], inplace=True)
    elapsed = perf_counter() - t0
    print(f"  {frac*100:>9.0f}%  {live:>10,}  {elapsed:>10.4f}")


# ── 5. inplace=True vs inplace=False ─────────────────────────────────────────

sep(f"5. inplace=True vs inplace=False  (N={N:,})")
print(f"  {'MODE':<20}  {'TIME (s)':>10}  {'NOTE'}")
print(f"  {'─'*20}  {'─'*10}  {'─'*20}")

ct = make_table(N)
t0 = perf_counter()
ct.sort_by(["sensor_id"], inplace=True)
t_inplace = perf_counter() - t0
print(f"  {'inplace=True':<20}  {t_inplace:>10.4f}  modifies table in-place")

ct = make_table(N)
t0 = perf_counter()
ct2 = ct.sort_by(["sensor_id"], inplace=False)
t_copy = perf_counter() - t0
print(f"  {'inplace=False':<20}  {t_copy:>10.4f}  returns new table")
print(f"\n  Copy overhead: {t_copy / t_inplace:.2f}×")
