#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Using blosc2.jit as a pandas execution engine.
#
# pandas' DataFrame.apply and Series.map accept an engine= argument: a
# callable exposing a __pandas_udf__ attribute that pandas dispatches to
# instead of running its own per-row/per-element Python loop. blosc2.jit is
# such an engine — but the function passed to it must be *vectorized*: it
# is called once with a full NumPy array (a column, or a row, depending on
# axis), not once per element.
#
# This example shows:
#   1. DataFrame.apply(f, engine=blosc2.jit) with axis=0 (the default) — the
#      case that actually benefits from the engine, and why axis=1 does not.
#   2. Series.map(f, engine=blosc2.jit).
#   3. The clear error raised for non-numeric columns instead of a deep
#      numexpr failure.
#   4. Measured timings for axis=0 on a large DataFrame — the actual win.

try:
    import pandas as pd
except ImportError:
    raise SystemExit("This example requires pandas: pip install pandas") from None

from time import perf_counter

import numpy as np

import blosc2

# ---------------------------------------------------------------------------
# 1. DataFrame.apply — axis=0 (column-wise) is where the engine wins
# ---------------------------------------------------------------------------

df = pd.DataFrame(
    {
        "a": np.linspace(0, 10, 8),
        "b": np.linspace(10, 20, 8),
    }
)
print("Input DataFrame:")
print(df)


def transform(col):
    return np.sin(col) * np.cos(col) + col**2


expected = df.apply(transform)
result = df.apply(transform, engine=blosc2.jit)
print("\ndf.apply(transform, engine=blosc2.jit):")
print(result)
assert isinstance(result, pd.DataFrame)  # not a raw ndarray
pd.testing.assert_frame_equal(result, expected)
print("matches plain df.apply(transform): True")

# axis=1 (row-wise) still calls the function once per row, same as plain
# pandas — for a handful of columns the per-call wrapping overhead of the
# compute engine is *larger* than the win, so engine=blosc2.jit is typically
# slower than plain apply(axis=1) there. Prefer axis=0, as above.
result_axis1 = df.apply(transform, engine=blosc2.jit, axis=1)
pd.testing.assert_frame_equal(result_axis1, df.apply(transform, axis=1))
print("\ndf.apply(transform, engine=blosc2.jit, axis=1) also matches (just not faster).")

# ---------------------------------------------------------------------------
# 1b. Timings: axis=0 is a real, measured win
# ---------------------------------------------------------------------------

N_ROWS, N_COLS = 1_000_000, 8
big_df = pd.DataFrame(
    {f"c{i}": np.random.default_rng(i).random(N_ROWS) for i in range(N_COLS)},
)


def heavier_transform(col):
    return np.sin(col) * np.cos(col) + col**2 - np.sqrt(np.abs(col)) + np.exp(-col)


def timeit(fn, reps=3):
    best = float("inf")
    for _ in range(reps):
        t0 = perf_counter()
        fn()
        best = min(best, perf_counter() - t0)
    return best


print(f"\nTimings on a {N_ROWS:,}-row, {N_COLS}-column DataFrame (min of 3 runs):")

t_plain0 = timeit(lambda: big_df.apply(heavier_transform))
t_engine0 = timeit(lambda: big_df.apply(heavier_transform, engine=blosc2.jit))
print(f"  {'plain df.apply(f):':<34s} {t_plain0 * 1000:7.1f} ms")
print(
    f"  {'df.apply(f, engine=blosc2.jit):':<34s} {t_engine0 * 1000:7.1f} ms   speedup: {t_plain0 / t_engine0:.1f}x"
)

# ---------------------------------------------------------------------------
# 2. Series.map
# ---------------------------------------------------------------------------

s = pd.Series(np.linspace(-5, 5, 6))
mapped = s.map(lambda x: x**2 - 1, engine=blosc2.jit)
print("\nSeries.map(f, engine=blosc2.jit):")
print(mapped)
pd.testing.assert_series_equal(mapped, s.map(lambda x: x**2 - 1))
print("matches plain Series.map: True")

# ---------------------------------------------------------------------------
# 3. Non-numeric columns raise a clear error
# ---------------------------------------------------------------------------

df_text = pd.DataFrame({"label": ["x", "y", "z"]})
try:
    df_text.apply(lambda x: x + 1, engine=blosc2.jit)
except ValueError as exc:
    print("\nNon-numeric column raises ValueError:")
    print(" ", exc)
