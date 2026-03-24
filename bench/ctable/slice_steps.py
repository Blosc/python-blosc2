#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring Column[::step].to_array() with varying step sizes.

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

ct = blosc2.CTable(RowModel, expected_size=N)
ct.extend(DATA)

print(f"CTable built with {len(ct):,} rows\n")
print("=" * 60)
print(f"{'STEP':<10} {'ROWS RETURNED':>15} {'TIME (s)':>12}")
print("-" * 60)

col = ct["score"]
for step in steps:
    t0 = time()
    arr = col[::step].to_array()
    t_total = time() - t0
    print(f"::{ step:<8} {len(arr):>15,} {t_total:>12.6f}")

print("-" * 60)
