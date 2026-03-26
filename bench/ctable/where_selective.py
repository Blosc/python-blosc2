#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring where() performance with varying selectivity.
# Filter: id < threshold, with thresholds covering 1%, 10%, 50%, 90%, 100%

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

ct = blosc2.CTable(RowModel, expected_size=N)
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
