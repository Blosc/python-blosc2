#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: CTable index kinds vs full-scan speedup.
#
# Sections
# ────────
#  ## BUCKET  — min/max per chunk; only helps on sorted/clustered data
#  ## PARTIAL — exact positions; works on any data layout
#  ## FULL    — exact positions, best performance; works on any data layout
#
# Each section benchmarks range queries (sensor_id > X) on random and
# sorted data, plus equality queries (sensor_id == X) where applicable.
# FULL also covers cardinality and compound filters.

from dataclasses import dataclass
from time import perf_counter

import numpy as np

import blosc2

# ── Config ────────────────────────────────────────────────────────────────────

N    = 1_000_000
REPS = 5

# ── Schema ────────────────────────────────────────────────────────────────────

@dataclass
class Row:
    sensor_id:   int   = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float64())
    region:      int   = blosc2.field(blosc2.int32())

np_dtype = np.dtype([
    ("sensor_id",   np.int32),
    ("temperature", np.float64),
    ("region",      np.int32),
])

rng = np.random.default_rng(42)

SID_MAX = N // 10   # 100_000 unique sensor_id values, ~10 rows each

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_table(sensor_ids):
    DATA = np.empty(N, dtype=np_dtype)
    DATA["sensor_id"]   = sensor_ids
    DATA["temperature"] = 15.0 + rng.random(N) * 25
    DATA["region"]      = rng.integers(0, 8, size=N, dtype=np.int32)
    ct = blosc2.CTable(Row, expected_size=N)
    ct.extend(DATA)
    return ct

def bench_gt(table, threshold, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(table["sensor_id"] > threshold)
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def bench_eq(table, value, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(table["sensor_id"] == value)
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def bench_compound(table, threshold, region, reps=REPS):
    times = []
    for _ in range(reps):
        t0 = perf_counter()
        result = table.where(
            (table["sensor_id"] > threshold) & (table["region"] == region)
        )
        times.append(perf_counter() - t0)
    return float(np.median(times)), len(result)

def drop_all_indexes(ct):
    for col in ("sensor_id", "region"):
        try:
            ct.drop_index(col)
        except Exception:
            pass

FRACS  = [0.999, 0.99, 0.95, 0.90, 0.75, 0.50, 0.25]
LABELS = ["0.1%", "1%", "5%", "10%", "25%", "50%", "75%"]

def print_range_table(results, title, width=70):
    print("```")
    print(f"{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")
    print(f"  {'SELECTIVITY':<14} {'ROWS':>9}  {'SCAN(ms)':>9}  {'IDX(ms)':>9}  {'SPEEDUP':>8}")
    print(f"  {'─'*14} {'─'*9}  {'─'*9}  {'─'*9}  {'─'*8}")
    for label, n, t_scan, t_idx in results:
        speedup = t_scan / t_idx if t_idx > 0 else float("inf")
        marker  = " ←" if speedup >= 2.0 else ("  (slower)" if speedup < 0.9 else "")
        print(f"  {label:<14} {n:>9,}  {t_scan*1e3:>9.1f}  {t_idx*1e3:>9.1f}  {speedup:>7.1f}×{marker}")
    print(f"{'─' * width}")
    print("```")

def bench_range_section(ct, kind, label):
    thresholds = [(lbl, int(SID_MAX * f)) for lbl, f in zip(LABELS, FRACS)]
    drop_all_indexes(ct)
    scan = {lbl: bench_gt(ct, thr) for lbl, thr in thresholds}
    ct.create_index("sensor_id", kind=kind)
    results = []
    for lbl, thr in thresholds:
        t_idx, n = bench_gt(ct, thr)
        t_scan, _ = scan[lbl]
        results.append((lbl, n, t_scan, t_idx))
    print_range_table(results, label)

def bench_eq_section(ct, kind):
    EQ_VALS = [0, SID_MAX // 4, SID_MAX // 2, SID_MAX - 1]
    drop_all_indexes(ct)
    scan_eq = {v: bench_eq(ct, v) for v in EQ_VALS}
    ct.create_index("sensor_id", kind=kind)
    idx_eq  = {v: bench_eq(ct, v) for v in EQ_VALS}
    print("```")
    print(f"  {'VALUE':<12} {'ROWS':>6}  {'SCAN(ms)':>9}  {'IDX(ms)':>9}  {'SPEEDUP':>8}")
    print(f"  {'─'*12} {'─'*6}  {'─'*9}  {'─'*9}  {'─'*8}")
    for v in EQ_VALS:
        t_s, n = scan_eq[v]
        t_i, _ = idx_eq[v]
        spd = t_s / t_i if t_i > 0 else float("inf")
        marker = " ←" if spd >= 2.0 else ""
        print(f"  =={v:<10,} {n:>6,}  {t_s*1e3:>9.1f}  {t_i*1e3:>9.1f}  {spd:>7.1f}×{marker}")
    print("```")

# ── Data ──────────────────────────────────────────────────────────────────────

rand_ids   = rng.integers(0, SID_MAX, size=N, dtype=np.int32)
sorted_ids = np.repeat(np.arange(SID_MAX, dtype=np.int32), N // SID_MAX)

ct_rand   = make_table(rand_ids)
ct_sorted = make_table(sorted_ids)

print(f"# CTable Index Benchmark  |  N={N:,}  REPS={REPS}")
print(f"\n> Random data: sensor_id uniform random in [0, {SID_MAX:,})")
print(f"> Sorted data: sensor_id = 0,0,…,1,1,…,2,2,… (clustered, ~10 rows/value)")


# ══════════════════════════════════════════════════════════════════════════════
# ## BUCKET
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n## BUCKET")
print("\n> Stores min/max per chunk. Can skip chunks whose range doesn't overlap the")
print("> query. Only effective when data is sorted/clustered. Useless on random data.")

print("\n### Range query — random data")
bench_range_section(ct_rand, blosc2.IndexKind.BUCKET, "Random data — BUCKET index")

print("\n### Range query — sorted data")
bench_range_section(ct_sorted, blosc2.IndexKind.BUCKET, "Sorted data — BUCKET index")


# ══════════════════════════════════════════════════════════════════════════════
# ## PARTIAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n## PARTIAL")
print("\n> Stores exact row positions. Works on any data layout.")
print("> Smaller index than FULL; slightly less overhead to build.")

print("\n### Range query — random data")
bench_range_section(ct_rand, blosc2.IndexKind.PARTIAL, "Random data — PARTIAL index")

print("\n### Range query — sorted data")
bench_range_section(ct_sorted, blosc2.IndexKind.PARTIAL, "Sorted data — PARTIAL index")

print("\n### Equality query — random data")
drop_all_indexes(ct_rand)
bench_eq_section(ct_rand, blosc2.IndexKind.PARTIAL)

print("\n### Equality query — sorted data")
drop_all_indexes(ct_sorted)
bench_eq_section(ct_sorted, blosc2.IndexKind.PARTIAL)


# ══════════════════════════════════════════════════════════════════════════════
# ## FULL
# ══════════════════════════════════════════════════════════════════════════════

print("\n\n## FULL")
print("\n> Stores exact row positions with full chunk coverage.")
print("> Best query performance; larger index than PARTIAL.")

print("\n### Range query — random data")
bench_range_section(ct_rand, blosc2.IndexKind.FULL, "Random data — FULL index")

print("\n### Range query — sorted data")
bench_range_section(ct_sorted, blosc2.IndexKind.FULL, "Sorted data — FULL index")

print("\n### Equality query — random data")
drop_all_indexes(ct_rand)
bench_eq_section(ct_rand, blosc2.IndexKind.FULL)

print("\n### Equality query — sorted data")
drop_all_indexes(ct_sorted)
bench_eq_section(ct_sorted, blosc2.IndexKind.FULL)

# ── Cardinality (FULL, sorted) ────────────────────────────────────────────────

print("\n### Cardinality comparison — sorted data, FULL index")
print("\n> Shows how repetition level affects speedup (data always sorted).")

CARD_CASES = [
    ("High rep  (10 uniq)", 10),
    ("Med  rep  (1k uniq)", 1_000),
    ("Low  rep  (1M uniq)", N),
]
SEL_FRACS  = [0.999, 0.99, 0.95, 0.90]
SEL_LABELS = ["0.1%", "1%", "5%", "10%"]

W2 = 72
print("```")
print(f"  {'CARDINALITY':<24}", end="")
for lbl in SEL_LABELS:
    print(f"  {lbl+' sel':>12}", end="")
print()
print("  " + "─" * (W2 - 2))

for case_lbl, n_unique in CARD_CASES:
    reps_per_val = N // n_unique
    ids = np.repeat(np.arange(n_unique, dtype=np.int32), reps_per_val)
    if len(ids) < N:
        ids = np.concatenate([ids, np.zeros(N - len(ids), dtype=np.int32)])
    sid_max = n_unique
    ct_c = make_table(ids)
    thr_list = [(lbl, int(sid_max * f)) for lbl, f in zip(SEL_LABELS, SEL_FRACS)]
    scan_c = {lbl: bench_gt(ct_c, thr) for lbl, thr in thr_list}
    ct_c.create_index("sensor_id", kind=blosc2.IndexKind.FULL)
    print(f"  {case_lbl:<24}", end="")
    for lbl, thr in thr_list:
        t_idx, _ = bench_gt(ct_c, thr)
        t_scan, _ = scan_c[lbl]
        spd = t_scan / t_idx if t_idx > 0 else float("inf")
        print(f"  {spd:>10.1f}×  ", end="")
    print()

print("  " + "─" * (W2 - 2))
print("  (speedup — higher is better)")
print("```")

# ── Compound filters (FULL, sorted) ──────────────────────────────────────────

print("\n### Compound filter — sorted data, FULL index")
print("\n> sensor_id > X  AND  region == Y  |  region in [0,8) → ~12.5% per value")

REGION_TARGET = 3
COMPOUND_THRESHOLDS = [
    ("0.1%+12.5%", int(SID_MAX * 0.999)),
    ("1%+12.5%",   int(SID_MAX * 0.99)),
    ("5%+12.5%",   int(SID_MAX * 0.95)),
    ("10%+12.5%",  int(SID_MAX * 0.90)),
]

drop_all_indexes(ct_sorted)
no_idx = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.create_index("sensor_id", kind=blosc2.IndexKind.FULL)
one_idx_sid = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.drop_index("sensor_id")
ct_sorted.create_index("region", kind=blosc2.IndexKind.FULL)
one_idx_reg = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}
ct_sorted.create_index("sensor_id", kind=blosc2.IndexKind.FULL)
two_idx = {lbl: bench_compound(ct_sorted, thr, REGION_TARGET) for lbl, thr in COMPOUND_THRESHOLDS}

W3 = 80
print("```")
print(f"{'─' * W3}")
print(f"  {'QUERY':<14} {'ROWS':>8}  {'NO IDX':>9}  {'IDX:sid':>9}  {'IDX:reg':>9}  {'2 IDX':>9}  {'BEST'}")
print(f"  {'─'*14} {'─'*8}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*12}")
for lbl, thr in COMPOUND_THRESHOLDS:
    n      = no_idx[lbl][1]
    t_none = no_idx[lbl][0]
    t_sid  = one_idx_sid[lbl][0]
    t_reg  = one_idx_reg[lbl][0]
    t_two  = two_idx[lbl][0]
    best_t = min(t_none, t_sid, t_reg, t_two)
    spd    = t_none / best_t
    winner = ["none", "sid", "reg", "2idx"][[t_none, t_sid, t_reg, t_two].index(best_t)]
    print(
        f"  {lbl:<14} {n:>8,}"
        f"  {t_none*1e3:>8.1f}ms"
        f"  {t_sid*1e3:>8.1f}ms"
        f"  {t_reg*1e3:>8.1f}ms"
        f"  {t_two*1e3:>8.1f}ms"
        f"  {winner}({spd:.1f}×)"
    )
print(f"{'─' * W3}")
print("```")
