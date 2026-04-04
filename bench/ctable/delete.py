#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring delete() performance with different index types:
# int, slice, and list — with varying sizes.

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
    ct = blosc2.CTable(Row, expected_size=N)
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
