#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for comparing chained where() calls vs a single combined filter.
# Filters: 250k < id < 750k, active == False, 25.0 < score < 75.0

from dataclasses import dataclass
from time import perf_counter as time

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


N = 1_000_000

print(f"where() chained vs combined benchmark  |  N = {N:,}")

# Build CTable once
np_dtype = np.dtype([
    ("id",     np.int64),
    ("c_val",  np.complex128),
    ("score",  np.float64),
    ("active", np.bool_),
])
DATA = np.array(
    [
        (i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0)
        for i in range(N)
    ],
    dtype=np_dtype,
)

ct = blosc2.CTable(Row, expected_size=N)
ct.extend(DATA)

print(f"CTable built with {len(ct):,} rows\n")
print("=" * 70)

# 1. Three chained where() calls
t0 = time()
r1 = ct.where(ct["id"] > 250_000)
r2 = r1.where(ct["id"] < 750_000)
r3 = r2.where(ct["score"] > 25.0)
r4 = r3.where(ct["score"] < 75.0)
r5 = r4.where(ct["active"] == False)
t_chained = time() - t0
print(f"Chained where() (5 calls):  {t_chained:.6f} s   rows: {len(r5):,}")

# 2. Single combined where() call
t0 = time()
result = ct.where(
    (ct["id"] > 250_000) & (ct["id"] < 750_000) &
    (ct["active"] == False) &
    (ct["score"] > 25.0) & (ct["score"] < 75.0)
)
t_combined = time() - t0
print(f"Combined where() (1 call):  {t_combined:.6f} s   rows: {len(result):,}")

print("=" * 70)
print(f"Speedup combined vs chained: {t_chained / t_combined:.2f}x")
