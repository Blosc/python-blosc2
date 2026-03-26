#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring compact() time and memory gain after deletions
# of varying fractions of the table.

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

print(f"compact() benchmark  |  N = {N:,}\n")

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

delete_fractions = [0.1, 0.25, 0.5, 0.75, 0.9]

print("=" * 75)
print(f"{'DELETED':>10} {'ROWS LEFT':>10} {'TIME (s)':>12} {'CBYTES BEFORE':>15} {'CBYTES AFTER':>14}")
print("-" * 75)

for frac in delete_fractions:
    ct = blosc2.CTable(RowModel, expected_size=N)
    ct.extend(DATA)

    n_delete = int(N * frac)
    ct.delete(list(range(n_delete)))

    cbytes_before = sum(col.cbytes for col in ct._cols.values()) + ct._valid_rows.cbytes

    t0 = time()
    ct.compact()
    t_compact = time() - t0

    cbytes_after = sum(col.cbytes for col in ct._cols.values()) + ct._valid_rows.cbytes

    print(
        f"{frac*100:>9.0f}%"
        f" {N - n_delete:>10,}"
        f" {t_compact:>12.4f}"
        f" {cbytes_before / 1024**2:>13.2f} MB"
        f" {cbytes_after / 1024**2:>12.2f} MB"
    )

print("-" * 75)
