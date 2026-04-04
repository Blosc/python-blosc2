#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark comparing CTable vs pandas DataFrame for:
#   1. Creation from a NumPy structured array
#   2. Column access (full column)
#   3. Filtering (where/query)
#   4. Row iteration

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
rng = np.random.default_rng(42)

print(f"CTable vs pandas benchmark  |  N = {N:,}\n")

# Build base data once
np_dtype = np.dtype([
    ("id",     np.int64),
    ("c_val",  np.complex128),
    ("score",  np.float64),
    ("active", np.bool_),
])
DATA = np.empty(N, dtype=np_dtype)
DATA["id"]     = np.arange(N, dtype=np.int64)
DATA["c_val"]  = rng.standard_normal(N) + 1j * rng.standard_normal(N)
DATA["score"]  = rng.uniform(0, 100, N)
DATA["active"] = rng.integers(0, 2, N, dtype=np.bool_)

print("=" * 65)
print(f"{'OPERATION':<30} {'CTable':>12} {'pandas':>12} {'SPEEDUP':>10}")
print("-" * 65)

# 1. Creation
t0 = time()
ct = blosc2.CTable(Row, expected_size=N)
ct.extend(DATA)
t_ct_create = time() - t0

t0 = time()
df = pd.DataFrame(DATA)
t_pd_create = time() - t0

print(f"{'Creation':<30} {t_ct_create:>12.4f} {t_pd_create:>12.4f} {t_pd_create/t_ct_create:>9.2f}x")

# 2. Column access (full column)
t0 = time()
arr = ct["score"]
t_ct_col = time() - t0

t0 = time()
arr = df["score"]
t_pd_col = time() - t0

print(f"{'Column access (full)':<30} {t_ct_col:>12.4f} {t_pd_col:>12.4f} {t_pd_col/t_ct_col:>9.2f}x")

# 2.5 Column access (full column)
t0 = time()
arr = ct["score"].to_numpy()
t_ct_col = time() - t0

t0 = time()
arr = df["score"].to_numpy()
t_pd_col = time() - t0

print(f"{'Column access to numpy (full)':<30} {t_ct_col:>12.4f} {t_pd_col:>12.4f} {t_pd_col/t_ct_col:>9.3f}x")

# 3. Filtering
t0 = time()
result_ct = ct.where((ct["id"] > 250_000) & (ct["id"] < 750_000))
t_ct_filter = time() - t0

t0 = time()
result_pd = df.query("250000 < id < 750000")
t_pd_filter = time() - t0

print(f"{'Filter (id 250k-750k)':<30} {t_ct_filter:>12.4f} {t_pd_filter:>12.4f} {t_pd_filter/t_ct_filter:>9.2f}x")

# 4. Row iteration
t0 = time()
for val in ct["score"]:
    pass
t_ct_iter = time() - t0

t0 = time()
for val in df["score"]:
    pass
t_pd_iter = time() - t0

print(f"{'Row iteration':<30} {t_ct_iter:>12.4f} {t_pd_iter:>12.4f} {t_pd_iter/t_ct_iter:>9.2f}x")

print("-" * 65)

# Memory
ct_cbytes = sum(col.cbytes for col in ct._cols.values()) + ct._valid_rows.cbytes
ct_nbytes = sum(col.nbytes for col in ct._cols.values()) + ct._valid_rows.nbytes
pd_nbytes  = df.memory_usage(deep=True).sum()

print(f"\nMemory — CTable compressed:   {ct_cbytes / 1024**2:.2f} MB")
print(f"Memory — CTable uncompressed: {ct_nbytes / 1024**2:.2f} MB")
print(f"Memory — pandas:              {pd_nbytes  / 1024**2:.2f} MB")
print(f"Compression ratio CTable:     {ct_nbytes / ct_cbytes:.2f}x")
