#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: iterative ingestion comparison — Pandas vs CTable
#   Data source: randomly generated numpy structured array

import time
from typing import Annotated

import numpy as np
import pandas as pd
import blosc2
from pydantic import BaseModel, Field


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


class RowModel(BaseModel):
    id:    Annotated[int,   NumpyDtype(np.int64)]
    name:  Annotated[str,   NumpyDtype(np.dtype("<U9"))]
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0)


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


print(f"=== BENCHMARK: Iterative Ingestion ({N:,} rows) ===\n")

# ─────────────────────────────────────────────────────────────
# 1. PANDAS
# ─────────────────────────────────────────────────────────────
print("--- 1. PANDAS (structured array -> DataFrame) ---")
data = make_data(N)

t0 = time.perf_counter()
df = pd.DataFrame(data)
t_pandas = time.perf_counter() - t0

mem_pandas = df.memory_usage(deep=True).sum() / (1024 ** 2)
print(f"Total time:   {t_pandas:.4f} s")
print(f"Memory (RAM): {mem_pandas:.2f} MB")

print("\n--- PANDAS: First 10 rows ---")
t0_print = time.perf_counter()
print(df.head(10).to_string())
t_print_pandas = time.perf_counter() - t0_print
print(f"\nPrint time: {t_print_pandas:.6f} s")

# ─────────────────────────────────────────────────────────────
# 2. BLOSC2 CTable
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("--- 2. BLOSC2 CTable (structured array -> extend) ---")
data = make_data(N)

t0 = time.perf_counter()
ct = blosc2.CTable(RowModel, expected_size=N)
ct.extend(data)
t_blosc = time.perf_counter() - t0

fields       = list(RowModel.model_fields.keys())
mem_blosc_c  = (sum(col.cbytes for col in ct._cols.values()) + ct._valid_rows.cbytes) / (1024 ** 2)
mem_blosc_uc = (sum(col.nbytes for col in ct._cols.values()) + ct._valid_rows.nbytes) / (1024 ** 2)

print(f"Total time:            {t_blosc:.4f} s")
print(f"Memory (uncompressed): {mem_blosc_uc:.2f} MB")
print(f"Memory (compressed):   {mem_blosc_c:.2f} MB")

print("\n--- BLOSC2: First 10 rows ---")
t0_print = time.perf_counter()
print(ct.head(10))
t_print_blosc = time.perf_counter() - t0_print
print(f"\nPrint time: {t_print_blosc:.6f} s")

# ─────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("--- SUMMARY ---")
speedup   = t_pandas / t_blosc
direction = "faster" if t_blosc < t_pandas else "slower"

print(f"{'METRIC':<30} {'Pandas':>12} {'Blosc2':>12}")
print("-" * 55)
print(f"{'Ingestion time (s)':<30} {t_pandas:>12.4f} {t_blosc:>12.4f}")
print(f"{'Memory (MB)':<30} {mem_pandas:>12.2f} {mem_blosc_c:>12.2f}")
print(f"{'Print time (s)':<30} {t_print_pandas:>12.6f} {t_print_blosc:>12.6f}")
print("-" * 55)
print(f"\nSpeedup:               {speedup:.2f}x {direction}")
print(f"Compression ratio:     {mem_blosc_uc / mem_blosc_c:.2f}x")
print(f"Blosc2 vs Pandas size: {mem_blosc_c / mem_pandas * 100:.1f}%")
