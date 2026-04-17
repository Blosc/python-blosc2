#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for comparing append() (row by row) vs extend() (bulk),
# to find the crossover point where extend() becomes worth it.

from dataclasses import dataclass
from time import time

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


# Parameter — change N to test different crossover points
N = 2
print("append() vs extend() benchmark")
for i in range(6):
    print("\n")
    print("%" * 100)


    # Base data generation
    data_list = [
        [i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0] for i in range(N)
    ]

    # 1. N individual append() calls
    print(f"{N} individual append() calls")
    ct_append = blosc2.CTable(Row, expected_size=N)
    t0 = time()
    for row in data_list:
        ct_append.append(row)
    t_append = time() - t0
    print(f"   Time: {t_append:.6f} s")
    print(f"   Rows: {len(ct_append):,}")

    # 2. N individual extend() calls (one row at a time)
    print(f"{N} individual extend() calls (one row at a time)")
    ct_extend_one = blosc2.CTable(Row, expected_size=N)
    t0 = time()
    for row in data_list:
        ct_extend_one.extend([row])
    t_extend_one = time() - t0
    print(f"   Time: {t_extend_one:.6f} s")
    print(f"   Rows: {len(ct_extend_one):,}")

    # 3. Single extend() call with all N rows at once
    print(f"Single extend() call with all {N} rows at once")
    ct_extend_bulk = blosc2.CTable(Row, expected_size=N)
    t0 = time()
    ct_extend_bulk.extend(data_list)
    t_extend_bulk = time() - t0
    print(f"   Time: {t_extend_bulk:.6f} s")
    print(f"   Rows: {len(ct_extend_bulk):,}")

    # Summary
    print("=" * 70)
    print(f"{'METHOD':<35} {'TIME (s)':>12} {'SPEEDUP vs append':>20}")
    print("-" * 70)
    print(f"{'append() x N':<35} {t_append:>12.6f} {'1.00x':>20}")
    print(f"{'extend() x N (one row each)':<35} {t_extend_one:>12.6f} {t_append / t_extend_one:>19.2f}x")
    print(f"{'extend() x 1 (all at once)':<35} {t_extend_bulk:>12.6f} {t_append / t_extend_bulk:>19.2f}x")
    print("-" * 70)

    N=N*2
