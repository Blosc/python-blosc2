#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: single combined where() vs three chained where() calls,
# across five selectivity levels. Results are averaged over REPEATS runs
# and shown as a grouped bar chart (red = single, blue = chained).

from dataclasses import dataclass
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np

import blosc2

N = 1_000_000
REPEATS = 5

# Nominal selectivities and their per-condition thresholds.
# Each condition is "col > threshold" on a uniform [0, 100] column.
# For overall selectivity s with 3 independent conditions each selecting
# fraction p: p = s^(1/3), threshold = 100 * (1 - p).
SELECTIVITIES = [0.01, 0.05, 0.10, 0.25, 0.50]
THRESHOLDS = [100 * (1 - s ** (1 / 3)) for s in SELECTIVITIES]
LABELS = [f"{int(s * 100)}%" for s in SELECTIVITIES]


@dataclass
class Row:
    x: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    y: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    z: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)


# Build table once
print(f"Building CTable with {N:,} rows...")
rng = np.random.default_rng(42)
np_dtype = np.dtype([("x", np.float64), ("y", np.float64), ("z", np.float64)])
data = np.empty(N, dtype=np_dtype)
data["x"] = rng.uniform(0, 100, N)
data["y"] = rng.uniform(0, 100, N)
data["z"] = rng.uniform(0, 100, N)

ct = blosc2.CTable(Row, expected_size=N)
ct.extend(data)
print(f"Done. Running benchmark ({REPEATS} repeats per selectivity).\n")

print(f"{'Selectivity':>12}  {'Single (s)':>12}  {'Chained (s)':>12}  {'Rows matched':>14}")
print("-" * 58)

single_avgs, chained_avgs = [], []
single_errs, chained_errs = [], []

for label, s, thresh in zip(LABELS, SELECTIVITIES, THRESHOLDS):
    # Conditions: ct.x > thresh, ct.y > thresh, ct.z > thresh
    cond_x = ct.x > thresh
    cond_y = ct.y > thresh
    cond_z = ct.z > thresh

    # --- single combined where ---
    single_times = []
    for _ in range(REPEATS):
        t0 = time()
        r = ct.where(cond_x & cond_y & cond_z)
        single_times.append(time() - t0)
    rows_matched = len(r)

    # --- three chained where calls ---
    chained_times = []
    for _ in range(REPEATS):
        t0 = time()
        r = ct.where(cond_x).where(cond_y).where(cond_z)
        chained_times.append(time() - t0)

    s_avg = sum(single_times) / REPEATS
    c_avg = sum(chained_times) / REPEATS
    # error bars: distance from avg to min/max
    s_err = [s_avg - min(single_times), max(single_times) - s_avg]
    c_err = [c_avg - min(chained_times), max(chained_times) - c_avg]

    single_avgs.append(s_avg)
    chained_avgs.append(c_avg)
    single_errs.append(s_err)
    chained_errs.append(c_err)

    actual_pct = rows_matched / N * 100
    print(f"{label:>12}  {s_avg:>12.4f}  {c_avg:>12.4f}  {rows_matched:>12,} ({actual_pct:.1f}%)")

# ---------------------------------------------------------------------------
# Double bar plot
# ---------------------------------------------------------------------------
x = np.arange(len(LABELS))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))

bars_single = ax.bar(
    x - width / 2,
    single_avgs,
    width,
    yerr=np.array(single_errs).T,
    capsize=4,
    color="red",
    alpha=0.85,
    label="Single where(A & B & C)",
    error_kw={"elinewidth": 1.5},
)
bars_chained = ax.bar(
    x + width / 2,
    chained_avgs,
    width,
    yerr=np.array(chained_errs).T,
    capsize=4,
    color="steelblue",
    alpha=0.85,
    label="Chained where(A).where(B).where(C)",
    error_kw={"elinewidth": 1.5},
)

ax.set_xlabel("Selectivity (fraction of rows matched)")
ax.set_ylabel("Time (s)")
ax.set_title(f"Single vs chained where() — {N:,} rows, {REPEATS} repeats each")
ax.set_xticks(x)
ax.set_xticklabels(LABELS)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()

out = "graph_where.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
