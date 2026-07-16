#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: DataFrame.apply(f, engine=blosc2.jit) vs plain DataFrame.apply(f)
#
# engine=blosc2.jit calls the vectorized function once per column (the
# default axis=0), so the win comes from the Blosc2/numexpr compute engine
# (operator fusion, multi-threading) beating plain NumPy on a
# multi-operation elementwise expression over a large 1D array. This script
# measures that on a 1,000,000-row, 8-column frame.
#
# Note: axis=1 (row-wise) is NOT a good fit for this engine. It still calls
# the function once per row in a Python loop either way, and for a handful
# of columns the wrapping overhead per call (building a compute-engine proxy
# for a tiny array) is larger than the win, so engine=blosc2.jit is actually
# *slower* than plain apply(axis=1) in that case. Use axis=0 (or restructure
# the computation to operate on whole columns) to get the engine's benefit.
#
# Each measurement is the minimum of NRUNS repetitions to reduce noise.

from time import perf_counter

import numpy as np
import pandas as pd

import blosc2

NRUNS = 3
NROWS = 1_000_000
NCOLS = 8


def make_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {f"c{i}": rng.random(NROWS) for i in range(NCOLS)},
    )


def transform(col):
    return np.sin(col) * np.cos(col) + col**2 - np.sqrt(np.abs(col)) + np.exp(-col)


def timeit(fn):
    best = float("inf")
    result = None
    for _ in range(NRUNS):
        t0 = perf_counter()
        result = fn()
        best = min(best, perf_counter() - t0)
    return best, result


def main():
    df = make_df()

    t_plain, result_plain = timeit(lambda: df.apply(transform))
    t_engine, result_engine = timeit(lambda: df.apply(transform, engine=blosc2.jit))

    pd.testing.assert_frame_equal(result_engine, result_plain)

    print(f"rows={NROWS}, cols={NCOLS}")
    print(f"plain df.apply(f):              {t_plain:.4f} s")
    print(f"df.apply(f, engine=blosc2.jit):  {t_engine:.4f} s")
    print(f"speedup:                         {t_plain / t_engine:.1f}x")


if __name__ == "__main__":
    main()
