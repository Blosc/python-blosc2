#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring the overhead of resize() when expected_size
# is too small (M rows) vs correctly sized (N rows) during extend().

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



M = 779
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
    ct_correct = blosc2.CTable(Row, expected_size=N)
    t0 = time()
    ct_correct.extend(DATA[:N])
    t_correct = time() - t0
    print(f"extend() expected_size=N  ({N:>8,}):  {t_correct:.4f} s   rows: {len(ct_correct):,}")

    # 2. extend() with wrong expected_size = M (forces resize)
    ct_wrong = blosc2.CTable(Row, expected_size=M)
    t0 = time()
    ct_wrong.extend(DATA[:N])
    t_wrong = time() - t0
    print(f"extend() expected_size=M  ({M:>8,}):  {t_wrong:.4f} s   rows: {len(ct_wrong):,}")

    # Summary
    print(f"  Slowdown from wrong expected_size: {t_wrong / t_correct:.2f}x")

    N *= 2
