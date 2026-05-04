#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: CTable str() / head() rendering time vs pandas.
#
# Measures how long it takes to render the first 10 rows as a table
# for both CTable (head()) and pandas (DataFrame.to_string()),
# plus ingestion time and memory footprint comparison.

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

import blosc2


@dataclass
class Row:
    id:    int   = blosc2.field(blosc2.int64())
    name:  str   = blosc2.field(blosc2.string(max_length=9), default="")
    score: float = blosc2.field(blosc2.float64(ge=0), default=0.0)


NAMES = ["benchmark", "alpha", "beta", "gamma", "delta",
         "epsilon", "zeta", "eta", "theta", "iota"]

N   = 100_000
rng = np.random.default_rng(42)

np_dtype = np.dtype([("id", np.int64), ("name", "<U9"), ("score", np.float64)])


def make_data(n: int) -> np.ndarray:
    arr = np.empty(n, dtype=np_dtype)
    arr["id"]    = np.arange(n, dtype=np.int64)
    arr["name"]  = np.array([rng.choice(NAMES) for _ in range(n)], dtype="<U9")
    arr["score"] = rng.uniform(0, 100, n)
    return arr


data = make_data(N)

t0 = perf_counter()
df = pd.DataFrame(data)
t_pandas = perf_counter() - t0

t0 = perf_counter()
ct = blosc2.CTable(Row, expected_size=N)
ct.extend(data)
t_blosc = perf_counter() - t0

t0 = perf_counter()
_ = df.head(10).to_string()
t_print_pandas = perf_counter() - t0

t0 = perf_counter()
_ = ct.head(10)
t_print_blosc = perf_counter() - t0

mem_pandas   = df.memory_usage(deep=True).sum() / 1024**2
mem_blosc_c  = (sum(c.cbytes for c in ct._cols.values()) + ct._valid_rows.cbytes) / 1024**2
mem_blosc_uc = (sum(c.nbytes for c in ct._cols.values()) + ct._valid_rows.nbytes) / 1024**2

print(f"CTable vs pandas — ingestion + print  |  N = {N:,}")
print()
print(f"  {'METRIC':<30}  {'pandas':>10}  {'CTable':>10}")
print(f"  {'─'*30}  {'─'*10}  {'─'*10}")
print(f"  {'Ingestion time (s)':<30}  {t_pandas:>10.4f}  {t_blosc:>10.4f}")
print(f"  {'Memory compressed (MB)':<30}  {mem_pandas:>10.2f}  {mem_blosc_c:>10.2f}")
print(f"  {'Memory uncompressed (MB)':<30}  {mem_pandas:>10.2f}  {mem_blosc_uc:>10.2f}")
print(f"  {'head(10) render time (s)':<30}  {t_print_pandas:>10.6f}  {t_print_blosc:>10.6f}")
print()
print(f"  Ingestion speedup CTable vs pandas:  {t_pandas / t_blosc:.2f}×")
print(f"  Compression ratio CTable:            {mem_blosc_uc / mem_blosc_c:.2f}×")
print(f"  CTable compressed vs pandas RAM:     {mem_blosc_c / mem_pandas * 100:.1f}%")
