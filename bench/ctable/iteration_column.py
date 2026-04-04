#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark for comparing full column iteration strategies:
#   1. for val in ct["score"]  — Python iterator via __iter__
#   2. np.array(list(ct["score"]))  — materialize via list then convert
#   3. ct["score"][0:N].to_array()  — slice view + to_array()

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

print(f"Column iteration benchmark  |  N = {N:,}\n")

# Build CTable once
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

ct = blosc2.CTable(Row, expected_size=N)
ct.extend(DATA)

print(f"CTable built with {len(ct):,} rows\n")
print("=" * 60)

col = ct["score"]

# 1. Python iterator
t0 = time()
for val in col:
    pass
t_iter = time() - t0
print(f"for val in col:              {t_iter:.4f} s")

# 2. list() + np.array()
t0 = time()
arr = np.array(list(col))
t_list = time() - t0
print(f"np.array(list(col)):         {t_list:.4f} s")

# 3. slice view + to_array()
t0 = time()
arr = col[0:N].to_numpy()
for val in arr:
    pass
t_toarray = time() - t0
print(f"col[0:N].to_array():         {t_toarray:.4f} s")

print("=" * 60)
print(f"Speedup to_array vs iter:    {t_iter / t_toarray:.2f}x")
print(f"Speedup to_array vs list:    {t_list / t_toarray:.2f}x")
