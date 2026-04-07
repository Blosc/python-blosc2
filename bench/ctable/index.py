#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring Column[int] access (single row by logical index),
# which exercises _find_physical_index() traversal over chunk metadata.

from dataclasses import dataclass
from time import time

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


N = 1_000_000
indices = [0, N // 4, N // 2, (3 * N) // 4, N - 1]

print(f"Column[int] access benchmark  |  N = {N:,}\n")

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
print(f"{'INDEX':<15} {'POSITION':>12} {'TIME (s)':>12}")
print("-" * 60)

col = ct["score"]
for idx in indices:
    t0 = time()
    val = col[idx]
    t_access = time() - t0
    position = f"{idx / N * 100:.0f}% into array"
    print(f"{idx:<15,} {position:>12}   {t_access:.6f}")

print("-" * 60)
