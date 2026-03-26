#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring Column[slice] + to_array() with slices of
# different sizes and positions: small, large, and middle of the array.

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
slices = [
    ("small  — start",  slice(0, 100)),
    ("small  — middle", slice(N // 2, N // 2 + 100)),
    ("small  — end",    slice(N - 100, N)),
    ("large  — start",  slice(0, 100_000)),
    ("large  — middle", slice(N // 2 - 50_000, N // 2 + 50_000)),
    ("large  — end",    slice(N - 100_000, N)),
    ("full   — all",    slice(0, N)),
]

print(f"Column[slice].to_array() benchmark  |  N = {N:,}\n")

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
print("=" * 65)
print(f"{'SLICE':<25} {'ROWS':>8} {'TIME (s)':>12}")
print("-" * 65)

col = ct["score"]
for label, s in slices:
    t0 = time()
    arr = col[s].to_numpy()
    t_total = time() - t0
    n_rows = s.stop - s.start
    print(f"{label:<25} {n_rows:>8,} {t_total:>12.6f}")

print("-" * 65)
