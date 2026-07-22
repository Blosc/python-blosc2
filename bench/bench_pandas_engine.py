#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: DataFrame.apply(f, engine=blosc2.jit) vs plain DataFrame.apply(f)
#
# engine=blosc2.jit calls the vectorized function once per column (the
# default axis=0), so the win comes from the Blosc2 compute engine (operator
# fusion, multi-threading) beating plain NumPy, which evaluates the function
# body one operation at a time and allocates a full-size temporary at each
# step.
#
# Two conditions must both hold for the engine to pay off, and this script
# measures each one separately:
#
#   * enough rows   -- below ~50k the per-call setup dominates (rows sweep)
#   * enough ops    -- a single operation has nothing to fuse (ops sweep)
#
# numexpr is measured alongside as a reference point: on in-memory frames it
# is somewhat faster than the Blosc2 engine, so the argument for
# engine=blosc2.jit is that you write a readable Python function instead of a
# quoted expression string, not that it wins a raw speed race. See
# doc/guides/pandas_engine.md.
#
# Note: axis=1 (row-wise) is NOT a good fit for this engine. It still calls
# the function once per row in a Python loop either way, and for a handful
# of columns the wrapping overhead per call (building a compute-engine proxy
# for a tiny array) is larger than the win, so engine=blosc2.jit is actually
# *slower* than plain apply(axis=1) in that case. Use axis=0 (or restructure
# the computation to operate on whole columns) to get the engine's benefit.
#
# Each measurement is the minimum of NRUNS repetitions to reduce noise.

from pathlib import Path
from time import perf_counter

import numexpr
import numpy as np
import pandas as pd

import blosc2

NRUNS = 5
NROWS = 1_000_000
NCOLS = 8

ROW_SWEEP = (1_000, 10_000, 100_000, 1_000_000, 5_000_000)

OUT_DIR = Path(__file__).resolve().parent.parent / "doc" / "guides" / "pandas_engine"

# dataviz reference palette, same values as bench/optim_tips/common.py
COLOR_TIP = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def make_df(nrows=NROWS, ncols=NCOLS):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {f"c{i}": rng.normal(size=nrows) for i in range(ncols)},
    )


def yeo_johnson(col, lam=0.5):
    """Power transform behind sklearn's PowerTransformer, applied per column.

    np.where evaluates both arms over the whole column, so each is clamped to
    its own domain: without the np.maximum calls, the unselected arm raises a
    negative base to a fractional power and floods the run with NaNs and
    "invalid value encountered in power" warnings.
    """
    pos = (np.power(np.maximum(col, 0.0) + 1.0, lam) - 1.0) / lam
    neg = -(np.power(np.maximum(-col, 0.0) + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)
    return np.where(col >= 0, pos, neg)


# The same transform as a single numexpr expression: legal, but this is what
# the readability argument is about.
YEO_JOHNSON_NX = (
    "where(c >= 0, "
    "((maximum(c, 0.0) + 1.0) ** lam - 1.0) / lam, "
    "-((maximum(-c, 0.0) + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam))"
)


def numexpr_apply(df, expr=YEO_JOHNSON_NX, lam=0.5):
    return pd.DataFrame(
        {col: numexpr.evaluate(expr, local_dict={"c": df[col].values, "lam": lam}) for col in df},
        index=df.index,
    )


# Prefixes of one expression, so the number of fused operations is the only
# variable: mixing in different *kinds* of operation would measure the cost of
# the operations rather than the benefit of fusing them.
OPS_SWEEP = (
    ("1", lambda col: np.sin(col)),
    ("2", lambda col: np.sin(col) * np.cos(col)),
    ("3", lambda col: np.sin(col) * np.cos(col) + col**2),
    ("4", lambda col: np.sin(col) * np.cos(col) + col**2 - np.sqrt(np.abs(col))),
    ("5", lambda col: np.sin(col) * np.cos(col) + col**2 - np.sqrt(np.abs(col)) + np.exp(-col)),
)


def timeit(fn):
    best = float("inf")
    result = None
    for _ in range(NRUNS):
        t0 = perf_counter()
        result = fn()
        best = min(best, perf_counter() - t0)
    return best, result


def speedup(df, func):
    """Plain apply vs engine=blosc2.jit, returning (speedup, t_plain, t_engine)."""
    t_plain, _ = timeit(lambda: df.apply(func))
    t_engine, _ = timeit(lambda: df.apply(func, engine=blosc2.jit))
    return t_plain / t_engine, t_plain, t_engine


def save_plot(row_speedups, ops_speedups, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_rows, ax_ops) = plt.subplots(1, 2, figsize=(8, 3.2))

    ax_rows.semilogx(ROW_SWEEP, row_speedups, "o-", color=COLOR_TIP, linewidth=2)
    ax_rows.set_xlabel("rows (log scale)", color=INK, fontsize=9)
    ax_rows.set_ylabel("speedup vs plain apply", color=INK, fontsize=9)
    ax_rows.set_title(f"{NCOLS} columns, Yeo-Johnson", color=MUTED, fontsize=9)

    labels = [name for name, _ in OPS_SWEEP]
    ax_ops.plot(range(len(labels)), ops_speedups, "o-", color=COLOR_TIP, linewidth=2)
    ax_ops.set_xticks(range(len(labels)))
    ax_ops.set_xticklabels(labels)
    ax_ops.set_xlabel("operations fused into one pass", color=INK, fontsize=9)
    ax_ops.set_title(f"{NROWS:,} rows x {NCOLS} columns", color=MUTED, fontsize=9)

    for ax, values in ((ax_rows, row_speedups), (ax_ops, ops_speedups)):
        # Break-even: below this line the engine is a net loss.
        ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_ylim(0, max(values) * 1.25)
        ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:g}x")
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(labelsize=9, colors=MUTED)

    fig.suptitle(
        "df.apply(f, engine=blosc2.jit): when it pays off",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    df = make_df()

    t_plain, result_plain = timeit(lambda: df.apply(yeo_johnson))
    t_engine, result_engine = timeit(lambda: df.apply(yeo_johnson, engine=blosc2.jit))
    t_numexpr, result_numexpr = timeit(lambda: numexpr_apply(df))

    pd.testing.assert_frame_equal(result_engine, result_plain)
    pd.testing.assert_frame_equal(result_numexpr, result_plain)

    print(f"rows={NROWS}, cols={NCOLS}, transform=Yeo-Johnson")
    print(f"plain df.apply(f):               {t_plain:.4f} s")
    print(f"df.apply(f, engine=blosc2.jit):  {t_engine:.4f} s   {t_plain / t_engine:.2f}x")
    print(f"numexpr per column:              {t_numexpr:.4f} s   {t_plain / t_numexpr:.2f}x")

    print("\nrows sweep (speedup vs plain apply):")
    row_speedups = []
    for nrows in ROW_SWEEP:
        sp, tp, te = speedup(make_df(nrows=nrows), yeo_johnson)
        row_speedups.append(sp)
        print(f"  {nrows:>9,} rows:  plain {tp:.4f} s  engine {te:.4f} s  {sp:.2f}x")

    print("\nops sweep (speedup vs plain apply):")
    ops_speedups = []
    for name, func in OPS_SWEEP:
        sp, tp, te = speedup(df, func)
        ops_speedups.append(sp)
        print(f"  {name.replace(chr(10), ' '):>20}:  plain {tp:.4f} s  engine {te:.4f} s  {sp:.2f}x")

    out_path = OUT_DIR / "speedup.png"
    save_plot(row_speedups, ops_speedups, out_path)
    print(f"\nplot saved to {out_path}")


if __name__ == "__main__":
    main()
