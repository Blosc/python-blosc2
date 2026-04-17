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


N = 1_000  # start small, increase when confident

data = [(i, float(i % 100), i % 2 == 0) for i in range(N)]
tabla = CTable(Row, new_data=data)

print(f"Table created with {len(tabla)} rows\n")

# -------------------------------------------------------------------
# Test 1: iterate without accessing any column (minimum cost)
# -------------------------------------------------------------------
t0 = time()
for _row in tabla:
    pass
t1 = time()
print(f"[Test 1] Iter without accessing columns:    {(t1 - t0)*1000:.3f} ms")

# -------------------------------------------------------------------
# Test 2: iterate accessing a single column (real_pos cached once)
# -------------------------------------------------------------------
t0 = time()
for row in tabla:
    _ = row["id"]
t1 = time()
print(f"[Test 2] Iter accessing 'id':               {(t1 - t0)*1000:.3f} ms")

# -------------------------------------------------------------------
# Test 3: iterate accessing all columns (real_pos cached once per row)
# -------------------------------------------------------------------
t0 = time()
for row in tabla:
    _ = row["id"]
    _ = row["score"]
    _ = row["active"]
t1 = time()
print(f"[Test 3] Iter accessing 3 columns:          {(t1 - t0)*1000:.3f} ms")

# -------------------------------------------------------------------
# Test 4: correctness — values match expected
# -------------------------------------------------------------------
errors = 0
for row in tabla:
    if row["id"] != row._nrow:
        errors += 1
    if row["score"] != float(row._nrow % 100):
        errors += 1
    if row["active"] != (row._nrow % 2 == 0):
        errors += 1

print(f"\n[Test 4] Correctness errors: {errors} (expected: 0)")

# -------------------------------------------------------------------
# Test 5: with holes (deleted rows)
# -------------------------------------------------------------------
tabla2 = CTable(Row, new_data=data)
tabla2.delete(list(range(0, N, 2)))  # delete even rows, keep odd ones

print(f"\nTable with holes: {len(tabla2)} rows (expected: {N // 2})")

t0 = time()
ids = []
for row in tabla2:
    ids.append(row["id"])
t1 = time()

expected_ids = [i for i in range(N) if i % 2 != 0]
ok = ids == expected_ids
print(f"[Test 5] Iter with holes ({N//2} rows):        {(t1 - t0)*1000:.3f} ms  |  correctness: {ok}")

# -------------------------------------------------------------------
# Test 6: real_pos is cached correctly (not recomputed)
# -------------------------------------------------------------------
row0 = next(iter(tabla))
assert row0._real_pos is None, "real_pos should be None before first access"
_ = row0["id"]
assert row0._real_pos is not None, "real_pos should be cached after first access"
print(f"\n[Test 6] real_pos caching: OK (real_pos={row0._real_pos})")
