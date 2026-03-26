#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for comparing append() (row by row) vs extend() (bulk),
# to find the crossover point where extend() becomes worth it.

from time import time
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field

import blosc2


class NumpyDtype:
    def __init__(self, dtype):
        self.dtype = dtype


# Row model
class RowModel(BaseModel):
    id: Annotated[int, NumpyDtype(np.int64)] = Field(ge=0)
    c_val: Annotated[complex, NumpyDtype(np.complex128)] = Field(default=0j)
    score: Annotated[float, NumpyDtype(np.float64)] = Field(ge=0, le=100)
    active: Annotated[bool, NumpyDtype(np.bool_)] = True


# Parameter — change N to test different crossover points
N = 2
print(f"append() vs extend() benchmark")
for i in range(6):
    print("\n")
    print("%" * 100)


    # Base data generation
    data_list = [
        [i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0] for i in range(N)
    ]

    # 1. N individual append() calls
    print(f"{N} individual append() calls")
    ct_append = blosc2.CTable(RowModel, expected_size=N)
    t0 = time()
    for row in data_list:
        ct_append.append(row)
    t_append = time() - t0
    print(f"   Time: {t_append:.6f} s")
    print(f"   Rows: {len(ct_append):,}")

    # 2. N individual extend() calls (one row at a time)
    print(f"{N} individual extend() calls (one row at a time)")
    ct_extend_one = blosc2.CTable(RowModel, expected_size=N)
    t0 = time()
    for row in data_list:
        ct_extend_one.extend([row])
    t_extend_one = time() - t0
    print(f"   Time: {t_extend_one:.6f} s")
    print(f"   Rows: {len(ct_extend_one):,}")

    # 3. Single extend() call with all N rows at once
    print(f"Single extend() call with all {N} rows at once")
    ct_extend_bulk = blosc2.CTable(RowModel, expected_size=N)
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
