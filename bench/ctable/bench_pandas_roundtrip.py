#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: pandas ↔ CTable round-trip (with on-disk persistence)
#
# Pipeline measured in four isolated steps:
#
#   1. pandas → CTable   : DataFrame.to_arrow() + CTable.from_arrow()
#   2. CTable.save()     : write in-memory CTable to disk
#   3. CTable.load()     : read disk table back into RAM
#   4. CTable → pandas   : CTable.to_arrow().to_pandas()
#
# Plus the combined full round-trip (steps 1-4) is shown at the end.
#
# Each measurement is the minimum of NRUNS repetitions to reduce noise.
# Schema: id (int64), score (float64), active (bool), label (string ≤16).

import os
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa

from blosc2 import CTable

NRUNS = 3
TABLE_DIR = "saved_ctable/bench_pandas"
SIZES = [1_000, 10_000, 100_000, 1_000_000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sep(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def tmin(fn, n: int = NRUNS) -> float:
    """Minimum elapsed time (s) over *n* calls of *fn*."""
    best = float("inf")
    for _ in range(n):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


def clean(path: str = TABLE_DIR) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def make_dataframe(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "id":     np.arange(n, dtype=np.int64),
        "score":  rng.uniform(0, 100, n).astype(np.float64),
        "active": rng.integers(0, 2, n, dtype=bool),
        "label":  [f"r{i % 10000:05d}" for i in range(n)],
    })


# ---------------------------------------------------------------------------
# Section 1: pandas → CTable  (in-memory)
# ---------------------------------------------------------------------------

sep("1. pandas → CTable  (from_arrow, in-memory)")
print(f"{'rows':>12}  {'pandas→arrow (s)':>18}  {'arrow→ctable (s)':>18}  {'total (s)':>12}")
print(f"{'----':>12}  {'----------------':>18}  {'----------------':>18}  {'---------':>12}")

ctables: dict[int, CTable] = {}  # keep for steps 2 & 4

for N in SIZES:
    df = make_dataframe(N)

    def bench_to_arrow(df=df):
        return pa.Table.from_pandas(df, preserve_index=False)

    def bench_from_arrow(df=df):
        at = pa.Table.from_pandas(df, preserve_index=False)
        return CTable.from_arrow(at)

    t_pa  = tmin(bench_to_arrow)
    t_ct  = tmin(bench_from_arrow) - t_pa   # from_arrow only
    t_tot = t_pa + t_ct

    # Keep one CTable for later steps
    at = pa.Table.from_pandas(df, preserve_index=False)
    ctables[N] = CTable.from_arrow(at)

    print(f"{N:>12,}  {t_pa:>18.4f}  {t_ct:>18.4f}  {t_tot:>12.4f}")


# ---------------------------------------------------------------------------
# Section 2: CTable.save()  (in-memory → disk)
# ---------------------------------------------------------------------------

sep("2. CTable.save()  (in-memory → disk)")
print(f"{'rows':>12}  {'save (s)':>14}  {'compressed':>12}  {'ratio':>8}")
print(f"{'----':>12}  {'--------':>14}  {'----------':>12}  {'-----':>8}")

for N in SIZES:
    t = ctables[N]
    path = os.path.join(TABLE_DIR, f"ct_{N}")

    def bench_save(t=t, path=path):
        if os.path.exists(path):
            shutil.rmtree(path)
        t.save(path, overwrite=True)

    elapsed = tmin(bench_save)
    # Final state for size info
    t.save(path, overwrite=True)
    cbytes = t.cbytes
    nbytes = t.nbytes
    ratio  = nbytes / cbytes if cbytes > 0 else float("nan")

    def _fmt(n):
        if n < 1024**2:
            return f"{n / 1024:.1f} KB"
        return f"{n / 1024**2:.1f} MB"

    print(f"{N:>12,}  {elapsed:>14.4f}  {_fmt(cbytes):>12}  {ratio:>7.2f}x")


# ---------------------------------------------------------------------------
# Section 3: CTable.load()  (disk → in-memory)
# ---------------------------------------------------------------------------

sep("3. CTable.load()  (disk → in-memory)")
print(f"{'rows':>12}  {'load (s)':>14}")
print(f"{'----':>12}  {'--------':>14}")

for N in SIZES:
    path = os.path.join(TABLE_DIR, f"ct_{N}")

    def bench_load(path=path):
        return CTable.load(path)

    elapsed = tmin(bench_load)
    print(f"{N:>12,}  {elapsed:>14.4f}")


# ---------------------------------------------------------------------------
# Section 4: CTable → pandas  (to_arrow → to_pandas)
# ---------------------------------------------------------------------------

sep("4. CTable → pandas  (to_arrow + to_pandas)")
print(f"{'rows':>12}  {'ctable→arrow (s)':>18}  {'arrow→pandas (s)':>18}  {'total (s)':>12}")
print(f"{'----':>12}  {'----------------':>18}  {'----------------':>18}  {'---------':>12}")

for N in SIZES:
    t = ctables[N]
    at_cache = t.to_arrow()  # pre-convert once so we can time each step cleanly

    def bench_to_arrow_ct(t=t):
        return t.to_arrow()

    def bench_to_pandas(at=at_cache):
        return at.to_pandas()

    t_arr = tmin(bench_to_arrow_ct)
    t_pd  = tmin(bench_to_pandas)
    t_tot = t_arr + t_pd

    print(f"{N:>12,}  {t_arr:>18.4f}  {t_pd:>18.4f}  {t_tot:>12.4f}")


# ---------------------------------------------------------------------------
# Section 5: Full round-trip  (pandas → CTable → disk → load → pandas)
# ---------------------------------------------------------------------------

sep("5. Full round-trip  (pandas → CTable → save → load → pandas)")
print(f"{'rows':>12}  {'round-trip (s)':>16}")
print(f"{'----':>12}  {'---------------':>16}")

for N in SIZES:
    df = make_dataframe(N)
    path = os.path.join(TABLE_DIR, f"rt_{N}")

    def bench_roundtrip(df=df, path=path):
        # pandas → CTable
        at = pa.Table.from_pandas(df, preserve_index=False)
        t = CTable.from_arrow(at)
        # save to disk
        t.save(path, overwrite=True)
        # load back
        t2 = CTable.load(path)
        # CTable → pandas
        return t2.to_arrow().to_pandas()

    elapsed = tmin(bench_roundtrip)
    print(f"{N:>12,}  {elapsed:>16.4f}")


# Cleanup
clean()
print()
