#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: memory footprint of pandas DataFrame vs CTable (compressed)
# across logarithmic row counts (10 → 100M), for three data regimes:
#   - random: high-entropy, hard to compress
#   - medium: structured but varied, compresses moderately
#   - easy:   low-entropy repetitive data, compresses well
#
# String column omitted so that 100M-row arrays stay tractable;
# numpy structured arrays are passed directly to CTable.extend().

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import blosc2

SIZES = [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

NP_DTYPE = np.dtype([("id", np.int64), ("score", np.float64), ("active", np.bool_)])


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


def make_random(n: int) -> np.ndarray:
    """High-entropy: random ids over wide range, fully random floats, random bools."""
    rng = np.random.default_rng(42)
    data = np.empty(n, dtype=NP_DTYPE)
    data["id"] = rng.integers(0, max(10 * n, 1), n, dtype=np.int64)
    data["score"] = rng.uniform(0, 100, n)
    data["active"] = rng.integers(0, 2, n).astype(bool)
    return data


def make_medium(n: int) -> np.ndarray:
    """Medium entropy: limited-range ids, 50 distinct scores randomly assigned, random bools.
    No noise — values repeat often enough for the compressor to find patterns."""
    rng = np.random.default_rng(7)
    data = np.empty(n, dtype=NP_DTYPE)
    # ids drawn from a range ~1000× smaller than n → many repeats, some compression
    id_range = max(1, n // 1000)
    data["id"] = rng.integers(0, id_range, n, dtype=np.int64)
    # exactly 50 distinct float64 values, randomly scattered (no noise)
    distinct = np.linspace(0, 100, 50)
    data["score"] = distinct[rng.integers(0, 50, n)]
    data["active"] = rng.integers(0, 2, n).astype(bool)
    return data


def make_easy(n: int) -> np.ndarray:
    """Low-entropy: sequential ids, 4 distinct scores, all True booleans."""
    data = np.empty(n, dtype=NP_DTYPE)
    data["id"] = np.arange(n, dtype=np.int64)
    data["score"] = np.tile([0.0, 25.0, 50.0, 75.0], n // 4 + 1)[:n]
    data["active"] = np.ones(n, dtype=bool)
    return data


def build_ctable(data: np.ndarray) -> blosc2.CTable:
    ct = blosc2.CTable(Row, expected_size=len(data))
    ct.extend(data)
    return ct


def fmt_size(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


def to_mb(b: int) -> float:
    return b / 1024 / 1024


# Pandas memory for numeric-only schema is deterministic; compute analytically
# to avoid allocating a 100M-row DataFrame just for bookkeeping.
PANDAS_BYTES_PER_ROW = NP_DTYPE.itemsize  # int64 + float64 + bool = 17 bytes

print("Memory benchmark — pandas vs CTable compressed  (log row counts)\n")
print(
    f"{'Size':>8}  {'pandas (MB)':>13}  {'ct random (MB)':>16}  {'ct medium (MB)':>16}  {'ct easy (MB)':>14}"
)
print("-" * 74)

pandas_mem, ct_random, ct_medium, ct_easy = [], [], [], []

for n in SIZES:
    p_mem = n * PANDAS_BYTES_PER_ROW
    pandas_mem.append(p_mem)

    d = make_random(n)
    ct_r = build_ctable(d)
    del d
    d = make_medium(n)
    ct_m = build_ctable(d)
    del d
    d = make_easy(n)
    ct_e = build_ctable(d)
    del d

    ct_random.append(ct_r.cbytes)
    del ct_r
    ct_medium.append(ct_m.cbytes)
    del ct_m
    ct_easy.append(ct_e.cbytes)
    del ct_e

    print(
        f"{fmt_size(n):>8}  {to_mb(p_mem):>13.4f}"
        f"  {to_mb(ct_random[-1]):>16.4f}"
        f"  {to_mb(ct_medium[-1]):>16.4f}"
        f"  {to_mb(ct_easy[-1]):>14.4f}"
    )

# ---------------------------------------------------------------------------
# Plot  (log/log axes)
# ---------------------------------------------------------------------------
x_labels = [fmt_size(n) for n in SIZES]
x = np.arange(len(SIZES))

fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(18, 5))

for ax, yscale, ylabel in [
    (ax_log, "log", "Memory (MB, log scale)"),
    (ax_lin, "linear", "Memory (MB)"),
]:
    ax.plot(x, [to_mb(v) for v in pandas_mem], "o-", linewidth=2, color="gray", label="pandas")
    ax.plot(
        x,
        [to_mb(v) for v in ct_random],
        "s--",
        linewidth=2,
        color="steelblue",
        label="CTable compressed (random)",
    )
    ax.plot(
        x,
        [to_mb(v) for v in ct_medium],
        "D--",
        linewidth=2,
        color="orange",
        label="CTable compressed (medium)",
    )
    ax.plot(
        x,
        [to_mb(v) for v in ct_easy],
        "^-",
        linewidth=2,
        color="forestgreen",
        label="CTable compressed (easy)",
    )
    ax.set_yscale(yscale)
    ax.set_xlabel("Number of rows")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")

ax_log.set_title("Log scale")
ax_lin.set_title("Linear scale")
handles, labels = ax_log.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Memory footprint: pandas vs CTable — random / medium / easy data", y=1.06)
fig.tight_layout()

out = "graph_memory.png"
plt.savefig(out, dpi=150)
print(f"\nPlot saved to {out}")
plt.show()
