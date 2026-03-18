#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring the overhead of resize() when expected_size
# is too small (M rows) vs correctly sized (N rows) during extend().

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



M = 50
N = 62_500
MAX_N = 1_000_000
print(f"expected_size benchmark  |  wrong expected_size = {M}")

# Pre-generate full dataset once
np_dtype = np.dtype([
    ("id",     np.int64),
    ("c_val",  np.complex128),
    ("score",  np.float64),
    ("active", np.bool_),
])
DATA = np.array(
    [
        (i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0)
        for i in range(MAX_N)
    ],
    dtype=np_dtype,
)

while N <= MAX_N:
    print("-" * 80)
    print(f"N = {N:,} rows")

    # 1. extend() with correct expected_size = N
    ct_correct = blosc2.CTable(RowModel, expected_size=N)
    t0 = time()
    ct_correct.extend(DATA[:N])
    t_correct = time() - t0
    print(f"extend() expected_size=N  ({N:>8,}):  {t_correct:.4f} s   rows: {len(ct_correct):,}")

    # 2. extend() with wrong expected_size = M (forces resize)
    ct_wrong = blosc2.CTable(RowModel, expected_size=M)
    t0 = time()
    ct_wrong.extend(DATA[:N])
    t_wrong = time() - t0
    print(f"extend() expected_size=M  ({M:>8,}):  {t_wrong:.4f} s   rows: {len(ct_wrong):,}")

    # Summary
    print(f"  Slowdown from wrong expected_size: {t_wrong / t_correct:.2f}x")

    N *= 2
