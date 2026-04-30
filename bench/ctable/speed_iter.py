#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: row iteration speed — how fast is iterating over CTable rows
# when most rows are skipped (only accessing every K-th row's value).
#
# Models a real-world "sample scan" pattern where not every row needs
# a column read, but you still iterate the whole table.

from dataclasses import dataclass
from time import perf_counter

import blosc2


@dataclass
class Row:
    id:     int   = blosc2.field(blosc2.int64(ge=0))
    score:  float = blosc2.field(blosc2.float64(ge=0, le=100))
    active: bool  = blosc2.field(blosc2.bool(), default=True)


N = 1_000_000
SAMPLE_EVERY = [1, 10, 100, 1_000, 10_000]

data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]
ct = blosc2.CTable(Row, expected_size=N)
ct.extend(data)

print(f"Row iteration sample-scan benchmark  |  N = {N:,}")
print()
print(f"  {'SAMPLE_EVERY':>13}  {'READS':>9}  {'TIME (s)':>10}  {'µs/read':>9}")
print(f"  {'─'*13}  {'─'*9}  {'─'*10}  {'─'*9}")

for k in SAMPLE_EVERY:
    n_reads = N // k
    i = 0
    t0 = perf_counter()
    for row in ct:
        i = (i + 1) % k
        if i == 0:
            _ = row["score"]
    elapsed = perf_counter() - t0
    us_per_read = elapsed / n_reads * 1e6 if n_reads > 0 else 0
    print(f"  {k:>13,}  {n_reads:>9,}  {elapsed:>10.4f}  {us_per_read:>9.2f}")
