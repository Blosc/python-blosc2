#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring row[int] access (full row via _RowIndexer),
# testing access at different positions across the array.

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


N = 1_000_000
indices = [0, N // 4, N // 2, (3 * N) // 4, N - 1]

print(f"row[int] access benchmark  |  N = {N:,}\n")

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

ct = blosc2.CTable(RowModel, expected_size=N)
ct.extend(DATA)

print(f"CTable built with {len(ct):,} rows\n")
print("=" * 60)
print(f"{'INDEX':<15} {'POSITION':>12} {'TIME (s)':>12}")
print("-" * 60)

for idx in indices:
    t0 = time()
    row = ct.row[idx]
    t_access = time() - t0
    position = f"{idx / N * 100:.0f}% into array"
    print(f"{idx:<15,} {position:>12}   {t_access:.6f}")

print("-" * 60)
