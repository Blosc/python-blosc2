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
# Row-wise (axis=1) computations that combine several columns per row are a
# different story: engine=blosc2.jit + axis=1 still calls the function once
# per row in a Python loop, so it is not the right tool there either. Instead,
# write the function to take the columns as separate array parameters and
# call it directly (no df.apply at all) -- see "Row-wise computations" in
# doc/guides/pandas_engine.md. bench_row_wise() below measures that pattern
# against a plain per-row apply() and vectorized NumPy on a genuine
# per-row-convergence problem (Kepler's equation via Newton-Raphson), where a
# real per-row `break` beats even vectorized NumPy.
#
# Each measurement is the minimum of NRUNS repetitions to reduce noise.

import math
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

# Plain per-row apply(axis=1) is ~1000x slower than the alternatives below;
# keep this sweep small so the benchmark finishes in a reasonable time.
ROW_WISE_APPLY_NROWS = 2_000

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


# The same transform written with a real per-element if/else. It compiles to a
# DSL kernel instead of being traced, so only the matching arm runs for each
# element and no clamping is needed. lam is inlined because apply() passes the
# column alone. The explanation lives here rather than in a docstring: a DSL
# kernel body cannot contain a string literal.
def yeo_johnson_branch(col):
    if col >= 0:
        out = (np.power(col + 1.0, 0.5) - 1.0) / 0.5
    else:
        out = -(np.power(-col + 1.0, 1.5) - 1.0) / 1.5
    return out


def yeo_johnson_scalar(x, lam=0.5):
    """Per-element Python, the shape you would write without any engine."""
    if x >= 0:
        return ((x + 1.0) ** lam - 1.0) / lam
    return -((-x + 1.0) ** (2.0 - lam) - 1.0) / (2.0 - lam)


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


# Kepler's equation, solved by Newton-Raphson: a genuine per-row-convergence
# problem (row["colname"] combines two columns, and rows converge in a
# different number of iterations), used to benchmark the row-wise
# "columns as direct-call parameters" pattern from doc/guides/pandas_engine.md
# against both a plain per-row apply(axis=1) and vectorized NumPy.
def kepler_row_scalar(row):
    m = row["mean_anomaly"]
    ecc = row["eccentricity"]
    e = m + ecc * math.sin(m)
    for _ in range(100):
        diff = (e - ecc * math.sin(e) - m) / (1.0 - ecc * math.cos(e))
        e = e - diff
        if abs(diff) < 1e-12:
            break
    return e


def kepler_numpy(m, ecc):
    e = m + ecc * np.sin(m)
    for _ in range(100):
        diff = (e - ecc * np.sin(e) - m) / (1.0 - ecc * np.cos(e))
        e = e - diff
        if np.max(np.abs(diff)) < 1e-12:
            break
    return e


@blosc2.jit
def kepler_dsl(mean_anomaly, eccentricity):
    e = mean_anomaly + eccentricity * sin(mean_anomaly)  # noqa: F821  # 'sin' resolved as a bare DSL function name
    for _ in range(100):
        diff = (e - eccentricity * sin(e) - mean_anomaly) / (1.0 - eccentricity * cos(e))  # noqa: F821
        e = e - diff
        if abs(diff) < 1e-12:
            break
    return e


def make_kepler_df(nrows):
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "mean_anomaly": rng.uniform(0, 2 * np.pi, nrows),
            "eccentricity": rng.uniform(0.0, 0.95, nrows),
        }
    )


def bench_row_wise():
    # Slice from one frame rather than calling make_kepler_df(n) twice with
    # different n: a fresh same-seeded Generator's bulk draws are not
    # guaranteed to share a common prefix across different requested sizes.
    df_full = make_kepler_df(NROWS)
    df_small = df_full.iloc[:ROW_WISE_APPLY_NROWS]
    m = df_full["mean_anomaly"].to_numpy()
    ecc = df_full["eccentricity"].to_numpy()

    t_apply, result_apply = timeit(lambda: df_small.apply(kepler_row_scalar, axis=1))
    t_numpy, result_numpy = timeit(lambda: kepler_numpy(m, ecc))
    t_dsl, result_dsl = timeit(
        lambda: np.asarray(kepler_dsl(df_full["mean_anomaly"], df_full["eccentricity"]))
    )

    # Cross-check correctness: plain apply on the small frame vs numpy on the
    # same rows, and the direct DSL call vs numpy on the full frame.
    np.testing.assert_allclose(
        result_apply.to_numpy(),
        kepler_numpy(m[:ROW_WISE_APPLY_NROWS], ecc[:ROW_WISE_APPLY_NROWS]),
        atol=1e-9,
    )
    np.testing.assert_allclose(result_dsl, result_numpy, atol=1e-9)

    print("\nrow-wise (axis=1), Kepler's equation via Newton-Raphson:")
    print(f"  plain apply(axis=1),      {ROW_WISE_APPLY_NROWS:>9,} rows:  {t_apply:.4f} s")
    print(f"  vectorized numpy,         {NROWS:>9,} rows:  {t_numpy:.4f} s")
    print(f"  direct DSL call,          {NROWS:>9,} rows:  {t_dsl:.4f} s   {t_numpy / t_dsl:.2f}x vs numpy")
    per_row_apply = t_apply / ROW_WISE_APPLY_NROWS
    per_row_dsl = t_dsl / NROWS
    print(
        f"  per row: apply {per_row_apply * 1e6:.1f} us vs direct DSL call {per_row_dsl * 1e6:.4f} us "
        f"(~{per_row_apply / per_row_dsl:,.0f}x)"
    )


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


SCALAR_ROWS = 50_000


def bench_branch_vs_where(df, t_plain, result_plain):
    """Real per-element if vs the traced np.where form, plus per-element Python.

    The scalar version is timed on a smaller frame and extrapolated: at the full
    size it takes over a second per run.
    """
    t_branch, result_branch = timeit(lambda: df.apply(yeo_johnson_branch, engine=blosc2.jit))
    pd.testing.assert_frame_equal(result_branch, result_plain)

    small = make_df(nrows=SCALAR_ROWS)
    t_small, _ = timeit(lambda: small.apply(lambda col: col.map(yeo_johnson_scalar)))
    t_scalar = t_small * (NROWS / SCALAR_ROWS)

    print("\nreal if vs np.where (both under engine=blosc2.jit):")
    print(f"  per-element Python, real if:   {t_scalar:.4f} s (extrapolated)  {t_plain / t_scalar:.2f}x")
    print(f"  engine, real if (DSL kernel):  {t_branch:.4f} s   {t_plain / t_branch:.2f}x")
    print("  (np.where form is the t_engine figure above)")


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

    bench_branch_vs_where(df, t_plain, result_plain)

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

    bench_row_wise()


if __name__ == "__main__":
    main()
