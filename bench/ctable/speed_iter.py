#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from dataclasses import dataclass
from time import time

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100))
    active: bool = blosc2.field(blosc2.bool(), default=True)


N = 1_000_000  # start small, increase when confident

data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]
tabla = CTable(Row, new_data=data)

print(f"Table created with {len(tabla)} rows\n")

# -------------------------------------------------------------------
# Test 1: iterate without accessing any column (minimum cost)
# -------------------------------------------------------------------
i=0
t0 = time()
for row in tabla:
    i=(i+1)%10000
    if i==0:
        _ = row["score"]

t1 = time()
print(f"[Test 1] Iter without accessing columns:    {(t1 - t0):.3f} s")
