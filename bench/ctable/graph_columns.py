#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: insertion time (extend) as a function of column count.
# Fixes row count at N and sweeps over COLUMN_COUNTS, averaging REPEATS runs.

import dataclasses
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np

import blosc2

N = 100_000
REPEATS = 5
COLUMN_COUNTS = [4, 8, 16, 32, 64]


def make_schema_and_data(n_cols: int, n_rows: int):
    """Dynamically build a dataclass schema and matching numpy structured array."""
    fields = []
    np_fields = []
    for i in range(n_cols):
        col = f"col_{i}"
        fields.append((col, float, dataclasses.field(default=blosc2.field(blosc2.float64()))))
        np_fields.append((col, np.float64))

    # Build dataclass dynamically
    dc_fields = [
        (f"col_{i}", float, dataclasses.field(default=blosc2.field(blosc2.float64())))
        for i in range(n_cols)
    ]
    Row = dataclasses.make_dataclass("Row", dc_fields)

    # Build numpy structured array
    rng = np.random.default_rng(42)
    dtype = np.dtype(np_fields)
    data = np.empty(n_rows, dtype=dtype)
    for col, _ in np_fields:
        data[col] = rng.uniform(0, 100, n_rows)

    return Row, data


def bench_extend(Row, data: np.ndarray) -> float:
    ct = blosc2.CTable(Row, expected_size=len(data))
    t0 = time()
    ct.extend(data)
    return time() - t0


print(f"Column-count benchmark — {N:,} rows, {REPEATS} repeats per column count\n")
print(f"{'Columns':>10}  {'Avg (s)':>10}  {'Min (s)':>10}  {'Max (s)':>10}")
print("-" * 46)

avg_times, min_times, max_times = [], [], []

for n_cols in COLUMN_COUNTS:
    Row, data = make_schema_and_data(n_cols, N)
    trials = [bench_extend(Row, data) for _ in range(REPEATS)]
    avg = sum(trials) / REPEATS
    avg_times.append(avg)
    min_times.append(min(trials))
    max_times.append(max(trials))
    print(f"{n_cols:>10}  {avg:>10.4f}  {min(trials):>10.4f}  {max(trials):>10.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
err_lo = [avg - mn for avg, mn in zip(avg_times, min_times)]
err_hi = [mx - avg for avg, mx in zip(avg_times, max_times)]

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    COLUMN_COUNTS,
    avg_times,
    yerr=[err_lo, err_hi],
    fmt="o-",
    capsize=4,
    linewidth=2,
    label=f"avg of {REPEATS} runs",
)
ax.set_xlabel("Number of columns")
ax.set_ylabel("Time (s)")
ax.set_title(f"CTable extend() insertion time vs. column count ({N:,} rows)")
ax.set_xticks(COLUMN_COUNTS)
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

out = "graph_columns.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
