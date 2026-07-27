#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""String workloads over the Chicago Taxi dataset: Blosc2 vs pandas/polars/DuckDB.

The companion of `compare-query-methods.py`, for the *string* columns rather
than the numeric ones.  Three tasks over `company` (`<U36`) and `payment.type`
(`<U11`), the whole 24.3 M-row table by default:

  filter     startswith(company, 'Taxi') & (payment_type != 'Cash')  -> bool
  transform  'co=' + company + '|pay=' + lower(payment_type)         -> str
  kernel     the same, but branching on whether the company is a cab
             company -- i.e. row-wise control flow, not one expression

All three are timed; only `kernel` is plotted.  It is the shape of the pandas-3
blog kernel (datapythonista.me/blog/whats-new-in-pandas-3): every other engine
has to express it as a mask plus two fully-evaluated branches, whereas blosc2
compiles it to a single masked pass with `@blosc2.dsl_kernel`.

`blosc2 (raw)` is the same blosc2 path with `clevel=0` on operands *and*
result.  Same container, same kernel, compression the only variable -- so the
gap between the two blosc2 bars is the price of compression, and the gap
between their footprints is what that price buys.

Usage:
    python string-ops.py                       # whole table, best of 3
    python string-ops.py --nrows 1000000 --apply
    python string-ops.py --engines blosc2,numpy --nrows 1000000
"""

import argparse
import gc
import hashlib
import time

import numpy as np

import blosc2

PARQUET = "chicago-taxi-flat.parquet"
COLS = ["company", "payment.type"]

# One chunk/block geometry for every blosc2 operand.  Expressions combining two
# NDArrays only take the (miniexpr) fast path when the operands share a chunk
# grid, and asarray() picks the grid from the itemsize -- which differs between
# <U36 and <U11.  Pinning it here is what keeps `filter` on miniexpr.
CHUNKS, BLOCKS = (1 << 16,), (1 << 13,)

RAW = blosc2.CParams(clevel=0)

TASKS = ["filter", "transform", "kernel"]
PLOT_TASK = "kernel"

# NumPy is implemented below but off by default: its `kernel` builds five
# full-width <U temporaries, ~10 GB each at 24 M rows.  Run it with
# `--engines numpy,...` at a row count that fits.
ENGINES = ["blosc2", "blosc2 (raw)", "pandas", "polars", "duckdb"]
# "blosc2 (varlen)" is off by default: it is a representation comparison, not a
# cross-engine one.  Run `--engines "blosc2,blosc2 (varlen)"` to reproduce it.
ALL_ENGINES = [*ENGINES, "numpy", "blosc2 (varlen)"]


def load(path, nrows):
    """First `nrows` rows of the two string columns, as fixed-width NumPy arrays."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    nrows = nrows or pf.metadata.num_rows
    batches, seen = [], 0
    for b in pf.iter_batches(batch_size=min(nrows, 1 << 20), columns=COLS):
        batches.append(b)
        seen += b.num_rows
        if seen >= nrows:
            break
    table = pa.Table.from_batches(batches).slice(0, nrows)
    out = []
    for name in COLS:
        col = table[name].combine_chunks().dictionary_decode()
        out.append(col.to_numpy(zero_copy_only=False).astype(str))
    return out


# --------------------------------------------------------------------------
# blosc2
# --------------------------------------------------------------------------


@blosc2.dsl_kernel
def taxi_label(company, ptype):
    pay = ptype.lower()
    c = company.lower()
    if " cab" in c:
        return "cab|" + c.removesuffix(" cab") + "|" + pay
    return "other|" + c + "|" + pay


def blosc2_setup(co, pt, cparams=None):
    cp = {"cparams": cparams} if cparams is not None else {}
    kw = {"chunks": CHUNKS, "blocks": BLOCKS, **cp}
    return blosc2.asarray(co, **kw), blosc2.asarray(pt, **kw), cp


def blosc2_filter(a, b, cp):
    # strict_miniexpr: a silent fallback to the NumPy path would still give the
    # right answer, so without this the number below would not mean what it says.
    e = blosc2.startswith(a, "Taxi") & (b != "Cash")
    return e.compute(strict_miniexpr=True, **cp)


def blosc2_transform(a, b, cp):
    e = "co=" + a + "|pay=" + blosc2.lower(b)
    return e.compute(strict_miniexpr=True, **cp)


def blosc2_kernel(a, b, cp):
    return blosc2.lazyudf(taxi_label, (a, b)).compute(**cp)


def blosc2_raw_setup(co, pt):
    return blosc2_setup(co, pt, cparams=RAW)


blosc2_raw_filter = blosc2_filter
blosc2_raw_transform = blosc2_transform
blosc2_raw_kernel = blosc2_kernel


# `blosc2 (varlen)` is the same expressions through blosc2.compute_varlen(),
# which evaluates via miniexpr's Arrow varlen entry point and returns a
# Utf8Array.  It packs `transform` to 34.2 B/row against the fixed-width path's
# 264 -- right on DuckDB's 35.9 -- and STILL loses on both time and stored size,
# because blosc2 compresses its results and the fixed-width form's NUL padding
# compresses to almost nothing.  Kept as the evidence for that; run it with
# `--engines "blosc2,blosc2 (varlen)"`, one engine per process, since running
# second costs ~40% on this machine.  `filter` returns bool, so there is nothing
# to pack and it reuses the ordinary path.
blosc2_varlen_setup = blosc2_setup
blosc2_varlen_filter = blosc2_filter


def blosc2_varlen_transform(a, b, cp):
    return blosc2.compute_varlen("co=" + a + "|pay=" + blosc2.lower(b))


def blosc2_varlen_kernel(a, b, cp):
    return blosc2.compute_varlen(blosc2.lazyudf(taxi_label, (a, b)))


# --------------------------------------------------------------------------
# NumPy
# --------------------------------------------------------------------------


def numpy_setup(co, pt):
    return co, pt


def numpy_filter(co, pt):
    return np.strings.startswith(co, "Taxi") & (pt != "Cash")


def numpy_transform(co, pt):
    return np.strings.add(np.strings.add("co=" + co, "|pay="), np.strings.lower(pt))


def numpy_kernel(co, pt):
    c = np.strings.lower(co)
    tail = np.strings.add("|", np.strings.lower(pt))
    # np.strings has no removesuffix(); endswith + slice is the same thing.
    trimmed = np.where(
        np.strings.endswith(c, " cab"), np.strings.slice(c, 0, np.strings.str_len(c) - 4), c
    )
    cab = np.strings.add("cab|" + trimmed, tail)
    other = np.strings.add("other|" + c, tail)
    return np.where(np.strings.find(c, " cab") >= 0, cab, other)


# --------------------------------------------------------------------------
# pandas
# --------------------------------------------------------------------------


def pandas_setup(co, pt):
    import pandas as pd

    return pd.Series(co, dtype="str"), pd.Series(pt, dtype="str")


def pandas_filter(co, pt):
    return co.str.startswith("Taxi") & (pt != "Cash")


def pandas_transform(co, pt):
    return "co=" + co + "|pay=" + pt.str.lower()


def pandas_kernel(co, pt):
    c = co.str.lower()
    tail = "|" + pt.str.lower()
    cab = "cab|" + c.str.removesuffix(" cab") + tail
    return ("other|" + c + tail).where(~c.str.contains(" cab", regex=False), cab)


def pandas_kernel_apply(co, pt):
    """The row-wise spelling of `kernel`, which is how it would first be written.

    Off the scale next to everything else, and reported separately for that
    reason -- it is the baseline `@blosc2.dsl_kernel` exists to replace.
    """
    import pandas as pd

    df = pd.DataFrame({"company": co, "ptype": pt})

    def f(row):
        pay = row["ptype"].lower()
        c = row["company"].lower()
        if " cab" in c:
            return "cab|" + c.removesuffix(" cab") + "|" + pay
        return "other|" + c + "|" + pay

    return df.apply(f, axis=1)


# --------------------------------------------------------------------------
# polars
# --------------------------------------------------------------------------


def polars_setup(co, pt):
    import polars as pl

    return pl.DataFrame({"company": co, "ptype": pt}), None


def _pl(df, e):
    return df.select(e.alias("r")).to_series()


def polars_filter(df, _):
    import polars as pl

    return _pl(df, pl.col("company").str.starts_with("Taxi") & (pl.col("ptype") != "Cash"))


def polars_transform(df, _):
    import polars as pl

    return _pl(df, pl.lit("co=") + pl.col("company") + "|pay=" + pl.col("ptype").str.to_lowercase())


def polars_kernel(df, _):
    import polars as pl

    c = pl.col("company").str.to_lowercase()
    tail = pl.lit("|") + pl.col("ptype").str.to_lowercase()
    return _pl(
        df,
        pl.when(c.str.contains(" cab", literal=True))
        .then(pl.lit("cab|") + c.str.strip_suffix(" cab") + tail)
        .otherwise(pl.lit("other|") + c + tail),
    )


# --------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------

# No removesuffix() in SQL; ends_with + a slice is the literal equivalent and
# stays away from the regex engine, which would measure something else.
_DUCK_NOSUFFIX = "CASE WHEN ends_with(c, ' cab') THEN c[1:length(c) - 4] ELSE c END"


def _duck(con, q):
    # .arrow() yields a RecordBatchReader from duckdb 1.5 on, a Table before it.
    res = con.sql(q).arrow()
    if hasattr(res, "read_all"):
        res = res.read_all()
    return res["r"]


def duckdb_setup(co, pt):
    import duckdb
    import pyarrow as pa

    con = duckdb.connect()
    con.register("t", pa.table({"company": co, "ptype": pt}))
    return con, None


def duckdb_filter(con, _):
    return _duck(con, "SELECT starts_with(company, 'Taxi') AND ptype <> 'Cash' AS r FROM t")


def duckdb_transform(con, _):
    return _duck(con, "SELECT 'co=' || company || '|pay=' || lower(ptype) AS r FROM t")


def duckdb_kernel(con, _):
    return _duck(
        con,
        f"""
    SELECT CASE WHEN contains(c, ' cab')
                THEN 'cab|' || ({_DUCK_NOSUFFIX}) || tail
                ELSE 'other|' || c || tail END AS r
    FROM (SELECT lower(company) AS c, '|' || lower(ptype) AS tail FROM t)
    """,
    )


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

WINDOW = 1 << 19  # rows per verification window; bounds peak memory of the check


def _window(x, lo, hi):
    """`x[lo:hi]` as a NumPy array, for any of the engines' native containers."""
    if type(x).__module__.startswith("pyarrow"):  # NDArray has a .slice() too
        return x.slice(lo, hi - lo).to_numpy(zero_copy_only=False)
    part = x[lo:hi]
    return np.asarray(part.to_numpy() if hasattr(part, "to_numpy") else part)


def digest(x, n):
    """Memory-bounded fingerprint of a result, for cross-engine agreement.

    A 24 M-row `<U101` result is ~10 GB, so nothing can hold a second copy to
    diff against.  Instead: hash every row's *length*, plus the exact content of
    every 97th row.  Lengths alone catch any misalignment (they are what the
    4096-element eval-block bug perturbed); the sample catches wrong values.
    """
    h = hashlib.blake2b(digest_size=16)
    for lo in range(0, n, WINDOW):
        w = _window(x, lo, min(lo + WINDOW, n))
        if w.dtype.kind in "bi":
            h.update(np.ascontiguousarray(w).tobytes())
        else:
            # A Utf8Array window is StringDType, which has no fixed-width cast
            # (and needs none -- np.strings works on it directly).
            s = w if w.dtype.kind == "T" else w.astype(str)
            h.update(np.strings.str_len(s).astype(np.int32).tobytes())
            h.update("\x00".join(s[::97].tolist()).encode())
    return h.hexdigest()


def footprint(x):
    """MB the result occupies in its engine's own container."""
    if isinstance(x, blosc2.NDArray):
        return x.schunk.cbytes / 2**20
    if isinstance(x, blosc2.Utf8Array):
        return x.cbytes / 2**20
    if isinstance(x, np.ndarray):
        return x.nbytes / 2**20
    if hasattr(x, "memory_usage"):  # pandas Series; matches its Arrow buffers
        return x.memory_usage(deep=True) / 2**20
    if hasattr(x, "to_arrow"):
        # polars Series.  NOT estimated_size(): that reports the data buffer
        # only and omits the 8 B/row int64 offsets, ~20% low on this workload.
        x = x.to_arrow()
    chunks = getattr(x, "chunks", [x])  # pyarrow ChunkedArray / Array
    return sum(b.size for c in chunks for b in c.buffers() if b is not None) / 2**20


def bench(fn, args, nruns):
    # One untimed pass first: miniexpr compiles the expression on its first
    # evaluation, and polars/duckdb build a plan.  Every engine gets the same
    # courtesy.
    out = fn(*args)
    best = float("inf")
    for _ in range(nruns):
        del out  # never hold two full-size results at once
        gc.collect()
        t0 = time.perf_counter()
        out = fn(*args)
        best = min(best, time.perf_counter() - t0)
    return best, out


def fname(engine, task):
    return f"{engine.replace(' (', '_').replace(')', '')}_{task}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nrows", type=int, default=0, help="0 = the whole table (24.3 M rows)")
    p.add_argument("--nruns", type=int, default=3)
    p.add_argument("--plot", default="string-ops.png")
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--engines", default=",".join(ENGINES), help=f"any of {ALL_ENGINES}")
    p.add_argument(
        "--apply",
        action="store_true",
        help="also time pandas' row-wise .apply() (~20 s per million rows)",
    )
    args = p.parse_args()
    engines = [e.strip() for e in args.engines.split(",")]

    co, pt = load(args.parquet, args.nrows)
    n = len(co)
    print(f"{n:,} rows  company={co.dtype}  payment.type={pt.dtype}")

    times, sizes, seen = {}, {}, {}
    # blosc2 (raw) runs last so the NumPy source can be dropped before it
    # allocates its own uncompressed copy of everything.
    for engine in sorted(engines, key=lambda e: e == "blosc2 (raw)"):
        operands = globals()[f"{fname(engine, 'setup')}"](co, pt)
        if engine == "blosc2":
            print(f"  operand cratio: company {operands[0].cratio:.0f}x, ptype {operands[1].cratio:.0f}x")
        if engine == "blosc2 (raw)":
            del co, pt
            gc.collect()
        for task in TASKS:
            secs, out = bench(globals()[fname(engine, task)], operands, args.nruns)
            times[engine, task] = secs
            sizes[engine, task] = footprint(out)
            d = digest(out, n)
            # Every engine must agree with the first one to report a result.
            assert seen.setdefault(task, d) == d, f"{engine} disagrees on {task}"
            print(f"  {engine:<13} {task:<10} {secs * 1000:9.1f} ms  {sizes[engine, task]:9.1f} MB")
            del out
            gc.collect()
        del operands
        gc.collect()

    if args.apply:
        secs, out = bench(pandas_kernel_apply, pandas_setup(co, pt), 1)
        assert digest(out, n) == seen["kernel"]
        print(f"  {'pandas .apply':<13} {'kernel':<10} {secs * 1000:9.1f} ms  (row-wise)")

    print("\nall engines agree on all three results")
    plot(times, sizes, engines, args.plot, n)


def plot(times, sizes, engines, path, nrows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Two blosc2 shades against one neutral: every bar is also value-labelled
    # and named on the x axis, so identity never rests on colour alone.
    shade = {"blosc2": "#1f77b4", "blosc2 (raw)": "#8fbde0"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)

    def panel(ax, vals, title, unit, fmt):
        bars = ax.bar(engines, vals, color=[shade.get(e, "#b3b9c4") for e in engines])
        ref = vals[engines.index("blosc2")] if "blosc2" in engines else 0
        for e, bar, v in zip(engines, bars, vals, strict=True):
            ratio = "" if e == "blosc2" or not ref else f"\n({v / ref:.1f}x)"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v,
                fmt.format(v) + ratio,
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#333333",
            )
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit, fontsize=9)
        ax.set_ylim(top=max(vals) * 1.3)
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e6e8ec", lw=0.8)
        ax.set_axisbelow(True)

    t = [times[e, PLOT_TASK] for e in engines]
    s = [sizes[e, PLOT_TASK] for e in engines]
    unit, scale = ("s", 1) if max(t) > 2 else ("ms", 1000)
    panel(axes[0], [v * scale for v in t], "kernel: time", f"{unit}, lower is better", "{:.2f} " + unit)
    panel(axes[1], s, "kernel: result footprint", "MB held in memory", "{:,.0f} MB")

    fig.suptitle(
        f"Chicago Taxi row-wise string kernel, {nrows:,} rows  (Nx = vs blosc2)\n"
        "'blosc2 (raw)' is the identical path at clevel=0: compression is the only variable",
        fontsize=11,
    )
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
