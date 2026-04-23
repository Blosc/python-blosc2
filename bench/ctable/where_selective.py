#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring where() performance with varying selectivity.
# Filter: id < threshold, with thresholds covering 1%, 10%, 50%, 90%, 100%

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
thresholds = [10,10_000, 100_000,250_000, 500_000,750_000 ,900_000, 999_990, 1_000_000]

print(f"where() selectivity benchmark  |  N = {N:,}")

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
print(f"{'THRESHOLD':<15} {'ROWS RETURNED':>15} {'SELECTIVITY':>13} {'TIME (s)':>12}")
print("-" * 70)

for threshold in thresholds:
    t0 = time()
    result = ct.where(ct["id"] < threshold)
    t_where = time() - t0
    selectivity = threshold / N * 100
    print(f"id < {threshold:<10,} {len(result):>15,} {selectivity:>12.1f}% {t_where:>12.6f}")

print("-" * 70)
