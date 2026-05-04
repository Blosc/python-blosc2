#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: row iteration cost with different access patterns.
#
# Measures:
#   1. Iteration without column access (loop overhead only)
#   2. Iteration accessing 1 column per row
#   3. Iteration accessing 3 columns per row
#   4. Iteration with deleted rows (holes in _valid_rows)

from dataclasses import dataclass
from time import perf_counter

import blosc2


@dataclass
class Row:
    id:     int   = blosc2.field(blosc2.int64(ge=0))
    score:  float = blosc2.field(blosc2.float64(ge=0, le=100))
    active: bool  = blosc2.field(blosc2.bool(), default=True)


N = 100_000

data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]
ct = blosc2.CTable(Row, expected_size=N)
ct.extend(data)

ct_holes = blosc2.CTable(Row, expected_size=N)
ct_holes.extend(data)
ct_holes.delete(list(range(0, N, 2)))   # keep only odd rows

print(f"Row iteration benchmark  |  N = {N:,}  |  holes table: {len(ct_holes):,} rows")
print()
print(f"  {'PATTERN':<35}  {'TIME (ms)':>10}  {'µs/row':>8}")
print(f"  {'─'*35}  {'─'*10}  {'─'*8}")

cases = [
    ("no column access",          ct,       lambda row: None),
    ("access 1 col (id)",         ct,       lambda row: row["id"]),
    ("access 3 cols",             ct,       lambda row: (row["id"], row["score"], row["active"])),
    ("access 1 col — with holes", ct_holes, lambda row: row["id"]),
]

for label, table, accessor in cases:
    n = len(table)
    t0 = perf_counter()
    for row in table:
        accessor(row)
    elapsed = perf_counter() - t0
    us = elapsed / n * 1e6
    print(f"  {label:<35}  {elapsed*1e3:>10.2f}  {us:>8.2f}")
