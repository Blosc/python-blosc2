#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring delete() performance with different index types:
# int, slice, and list — with varying sizes.

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

print(f"delete() benchmark  |  N = {N:,}\n")

# Build base data once
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

delete_cases = [
    ("int",          0),
    ("slice small",  slice(0, 100)),
    ("slice large",  slice(0, 100_000)),
    ("slice full",   slice(0, N)),
    ("list small",   list(range(100))),
    ("list large",   list(range(100_000))),
    ("list full",    list(range(N))),
]

print("=" * 60)
print(f"{'CASE':<20} {'ROWS DELETED':>14} {'TIME (s)':>12}")
print("-" * 60)

for label, key in delete_cases:
    ct = blosc2.CTable(RowModel, expected_size=N)
    ct.extend(DATA)

    if isinstance(key, int):
        n_deleted = 1
    elif isinstance(key, slice):
        n_deleted = len(range(*key.indices(N)))
    else:
        n_deleted = len(key)

    t0 = time()
    ct.delete(key)
    t_delete = time() - t0
    print(f"{label:<20} {n_deleted:>14,} {t_delete:>12.6f}")

print("-" * 60)
