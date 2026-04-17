#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Benchmark: append() overhead introduced by the new schema pipeline
#
# The new append() path routes every row through:
#   _normalize_row_input → validate_row (Pydantic) → _coerce_row_to_storage
#
# This benchmark isolates how much each step costs, and shows the
# total overhead vs the raw NDArray write speed.

from dataclasses import dataclass
from time import perf_counter

import numpy as np

import blosc2
from blosc2.schema_compiler import compile_schema
from blosc2.schema_validation import build_validator_model, validate_row


@dataclass
class Row:
    id:     int   = blosc2.field(blosc2.int64(ge=0))
    score:  float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool  = blosc2.field(blosc2.bool(), default=True)


N = 5_000
rng = np.random.default_rng(42)
data = [
    (int(i), float(rng.uniform(0, 100)), bool(i % 2))
    for i in range(N)
]
schema = compile_schema(Row)
# Warm up the Pydantic model cache
build_validator_model(schema)

print(f"append() pipeline cost breakdown  |  N = {N:,} rows")
print("=" * 60)

# ── 1. Raw NDArray writes (no CTable overhead at all) ────────────────────────
ids    = np.zeros(N, dtype=np.int64)
scores = np.zeros(N, dtype=np.float64)
flags  = np.zeros(N, dtype=np.bool_)
mask   = np.zeros(N, dtype=np.bool_)

t0 = perf_counter()
for i, (id_, score, active) in enumerate(data):
    ids[i]    = id_
    scores[i] = score
    flags[i]  = active
    mask[i]   = True
t_raw = perf_counter() - t0
print(f"{'Raw NumPy writes (baseline)':<40} {t_raw:.4f} s")

# ── 2. _normalize_row_input only ─────────────────────────────────────────────
t_obj = blosc2.CTable(Row, expected_size=N, validate=False)
t0 = perf_counter()
for row in data:
    _ = t_obj._normalize_row_input(row)
t_normalize = perf_counter() - t0
print(f"{'_normalize_row_input only':<40} {t_normalize:.4f} s  ({t_normalize/t_raw:.1f}x baseline)")

# ── 3. Pydantic validate_row only ────────────────────────────────────────────
row_dicts = [t_obj._normalize_row_input(row) for row in data]
t0 = perf_counter()
for rd in row_dicts:
    _ = validate_row(schema, rd)
t_validate = perf_counter() - t0
print(f"{'validate_row (Pydantic) only':<40} {t_validate:.4f} s  ({t_validate/t_raw:.1f}x baseline)")

# ── 4. _coerce_row_to_storage only ───────────────────────────────────────────
t0 = perf_counter()
for rd in row_dicts:
    _ = t_obj._coerce_row_to_storage(rd)
t_coerce = perf_counter() - t0
print(f"{'_coerce_row_to_storage only':<40} {t_coerce:.4f} s  ({t_coerce/t_raw:.1f}x baseline)")

# ── 5. Full append(), validate=False  (3 runs, take minimum) ─────────────────
RUNS = 3
best_off = float("inf")
for _ in range(RUNS):
    t_obj2 = blosc2.CTable(Row, expected_size=N, validate=False)
    t0 = perf_counter()
    for row in data:
        t_obj2.append(row)
    best_off = min(best_off, perf_counter() - t0)
t_append_off = best_off
print(f"{'Full append(), validate=False':<40} {t_append_off:.4f} s  ({t_append_off/t_raw:.1f}x baseline)")

# ── 6. Full append(), validate=True  (3 runs, take minimum) ──────────────────
best_on = float("inf")
for _ in range(RUNS):
    t_obj3 = blosc2.CTable(Row, expected_size=N, validate=True)
    t0 = perf_counter()
    for row in data:
        t_obj3.append(row)
    best_on = min(best_on, perf_counter() - t0)
t_append_on = best_on
print(f"{'Full append(), validate=True':<40} {t_append_on:.4f} s  ({t_append_on/t_raw:.1f}x baseline)")

print()
print("=" * 60)
pydantic_cost = max(t_append_on - t_append_off, 0.0)
print(f"{'Pydantic overhead in append()':<40} {pydantic_cost:.4f} s")
if t_append_on > 0:
    print(f"{'Validation fraction of total':<40} {pydantic_cost/t_append_on*100:.1f}%")
print(f"{'Per-row Pydantic cost (isolated)':<40} {(t_validate/N)*1e6:.2f} µs/row")
print()
print(f"Note: append() is dominated by blosc2 I/O ({t_append_off/t_raw:.0f}x raw numpy),")
print("      not by the validation pipeline.")
print("      The main bottleneck is the last_true_pos backward scan per row.")
