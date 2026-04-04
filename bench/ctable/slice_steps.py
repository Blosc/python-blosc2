#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring Column[::step].to_array() with varying step sizes.

from time import time
from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


N = 1_000_000
steps = [1, 2, 4, 8, 16, 100, 1000]

print(f"Column[::step].to_array() benchmark  |  N = {N:,}\n")

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
print("=" * 60)
print(f"{'STEP':<10} {'ROWS RETURNED':>15} {'TIME (s)':>12}")
print("-" * 60)

col = ct["score"]
for step in steps:
    t0 = time()
    arr = col[::step].to_numpy()
    t_total = time() - t0
    print(f"::{ step:<8} {len(arr):>15,} {t_total:>12.6f}")

print("-" * 60)
