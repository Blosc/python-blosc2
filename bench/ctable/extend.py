#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for measuring CTable creation time from three different sources:
#   1. Python list of lists (1M rows)
#   2. NumPy structured array (1M rows) — list of named tuples
#   3. An existing CTable (previously created from Python lists, 1M rows)

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


N = 1_000_000
print(f"CTable creation benchmark with {N:,} rows\n")

# ---------------------------------------------------------------------------
# Base data generation (not part of the benchmark timing)
# ---------------------------------------------------------------------------
print("Generating base data...")

t0 = time()
data_list = [
    [i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0]
    for i in range(N)
]
t_gen_list = time() - t0
print(f"  Python list generated in:         {t_gen_list:.4f} s")

t0 = time()
np_dtype = np.dtype([
    ("id",     np.int64),
    ("c_val",  np.complex128),
    ("score",  np.float64),
    ("active", np.bool_),
])
data_np = np.array(
    [
        (i, complex(i * 0.1, i * 0.01), 10.0 + (i % 100) * 0.4, i % 3 == 0)
        for i in range(N)
    ],
    dtype=np_dtype,
)
t_gen_np = time() - t0
print(f"  NumPy structured array generated: {t_gen_np:.4f} s\n")

# ---------------------------------------------------------------------------
# 1. Creation from a Python list of lists
# ---------------------------------------------------------------------------
print("CTable from Python list of lists")
t0 = time()
ct_from_list = blosc2.CTable(Row, expected_size=N)
ct_from_list.extend(data_list)
t_from_list = time() - t0
print(f"   extend() time (Python list):  {t_from_list:.4f} s")
print(f"   Rows: {len(ct_from_list):,}")

# ---------------------------------------------------------------------------
# 2. Creation from a NumPy structured array (list of named tuples)
# ---------------------------------------------------------------------------
print("CTable from NumPy structured array")
t0 = time()
ct_from_np = blosc2.CTable(Row, expected_size=N)
ct_from_np.extend(data_np)
t_from_np = time() - t0
print(f"   extend() time (NumPy struct): {t_from_np:.4f} s")
print(f"   Rows: {len(ct_from_np):,}")


# ---------------------------------------------------------------------------
# 3. Creation from an existing CTable (ct_from_list, already built above)
# ---------------------------------------------------------------------------
print("CTable from an existing CTable")
t0 = time()
ct_from_ctable = blosc2.CTable(Row, expected_size=N)
ct_from_ctable.extend(ct_from_list)
t_from_ctable = time() - t0
print(f"   extend() time (CTable):       {t_from_ctable:.4f} s")
print(f"   Rows: {len(ct_from_ctable):,}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n")
print("=" * 60)
print(f"{'SOURCE':<30} {'TIME (s)':>12} {'SPEEDUP vs list':>18}")
print("-" * 60)
print(f"{'Python list of lists':<30} {t_from_list:>12.4f} {'1.00x':>18}")
print(f"{'NumPy structured array':<30} {t_from_np:>12.4f} {t_from_list / t_from_np:>17.2f}x")
print(f"{'Existing CTable':<30} {t_from_ctable:>12.4f} {t_from_list / t_from_ctable:>17.2f}x")
