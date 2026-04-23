#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring compact() time and memory gain after deletions
# of varying fractions of the table.

from dataclasses import dataclass
from time import perf_counter as time

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    c_val: complex = blosc2.field(blosc2.complex128(), default=0j)
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


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
    ct = blosc2.CTable(Row, expected_size=N)
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
