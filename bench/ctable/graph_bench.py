#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: insertion time (extend) as a function of table size.
# Runs REPEATS trials per size and plots the averaged wall-clock time.

from dataclasses import dataclass
from time import perf_counter as time

import matplotlib.pyplot as plt
import numpy as np

import blosc2

SIZES = [100_000, 200_000, 300_000, 400_000, 500_000, 600_000, 700_000, 800_000, 900_000, 1_000_000]
REPEATS = 5


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)
    label: str = blosc2.field(blosc2.string(max_length=16), default="")


def make_data(n: int) -> list:
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 100, n)
    actives = (rng.integers(0, 2, n)).astype(bool)
    return [
        [i, float(scores[i]), bool(actives[i]), f"item_{i % 1000}"]
        for i in range(n)
    ]


def bench_extend(n: int, data: list) -> float:
    ct = blosc2.CTable(Row, expected_size=n)
    t0 = time()
    ct.extend(data)
    return time() - t0


print(f"Insertion benchmark — {REPEATS} repeats per size\n")
print(f"{'Size':>12}  {'Avg (s)':>10}  {'Min (s)':>10}  {'Max (s)':>10}")
print("-" * 50)

avg_times = []
min_times = []
max_times = []

for n in SIZES:
    data = make_data(n)
    trials = [bench_extend(n, data) for _ in range(REPEATS)]
    avg = sum(trials) / REPEATS
    avg_times.append(avg)
    min_times.append(min(trials))
    max_times.append(max(trials))
    print(f"{n:>12,}  {avg:>10.4f}  {min(trials):>10.4f}  {max(trials):>10.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
x = [n / 1_000_000 for n in SIZES]
err_lo = [avg - mn for avg, mn in zip(avg_times, min_times)]
err_hi = [mx - avg for avg, mx in zip(avg_times, max_times)]

fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(
    x,
    avg_times,
    yerr=[err_lo, err_hi],
    fmt="o-",
    capsize=4,
    linewidth=2,
    label=f"avg of {REPEATS} runs",
)
ax.set_xlabel("Number of rows (millions)")
ax.set_ylabel("Time (s)")
ax.set_title("CTable extend() insertion time vs. table size")
ax.set_xticks(x)
ax.set_xticklabels([f"{n / 1_000_000:.1f}M" for n in SIZES])
ax.legend()
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

out = "graph_bench.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
