#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Copying a CTable: in-memory, persistent, reblocking, and recompression.
# The table has a mix of column types — scalar, dictionary, varlen string,
# and list — to show that all are handled correctly in every copy variant.

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass

import numpy as np

import blosc2 as b2


@dataclass
class Ride:
    distance_km: float = b2.field(b2.float32(ge=0.0))
    duration_sec: int = b2.field(b2.int32(ge=0))
    payment: str = b2.field(b2.string(max_length=12))  # dictionary column
    waypoints: list[float] = b2.field(  # noqa: RUF009         # list column (no default)
        b2.list(b2.float32(), nullable=True, batch_rows=16)
    )
    fare: float = b2.field(b2.float64(ge=0.0, null_value=-1.0), default=-1.0)
    verified: bool = b2.field(b2.bool(), default=True)


# ── build a table with 2 000 rows ───────────────────────────────────────────
rng = np.random.default_rng(0)
N = 2_000

PAYMENT_TYPES = ["cash", "card", "app", "voucher"]

distances = rng.uniform(0.5, 40.0, N).astype(np.float32)
durations = rng.integers(60, 7200, N).astype(np.int32)
fares = rng.uniform(3.0, 120.0, N)
fares[rng.random(N) < 0.05] = -1.0  # ~5 % fare unknown
payments = rng.choice(PAYMENT_TYPES, N).tolist()
verified = (rng.random(N) > 0.1).tolist()
waypoints = [
    None if rng.random() < 0.1 else rng.uniform(-90, 90, rng.integers(2, 8)).tolist() for _ in range(N)
]

data = list(
    zip(
        distances.tolist(),
        durations.tolist(),
        payments,
        waypoints,
        fares.tolist(),
        verified,
        strict=False,
    )
)

t = b2.CTable(Ride, new_data=data)
print(f"Source table: {t.nrows} rows, {t.ncols} columns")
print(f"  blocks : {t.blocks}  (items per block)")
print(f"  cbytes : {t.cbytes / 1e6:.2f} MB\n")


# ── 1. in-memory copy ───────────────────────────────────────────────────────
t0 = time.perf_counter()
mem_copy = t.copy()
print(f"1. In-memory copy:  {time.perf_counter() - t0:.4f}s")
assert mem_copy.nrows == t.nrows
assert mem_copy.waypoints[:5] == t.waypoints[:5]


# ── 2. persistent copy to .b2z ──────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    b2z_path = os.path.join(tmp, "rides.b2z")

    t0 = time.perf_counter()
    disk_copy = t.copy(urlpath=b2z_path)
    print(f"2. Persistent .b2z: {time.perf_counter() - t0:.4f}s")

    assert disk_copy.nrows == t.nrows
    assert disk_copy[0].payment == t[0].payment
    assert disk_copy.waypoints[-1] == t.waypoints[-1]
    disk_copy.close()

    # Reopen and spot-check
    with b2.open(b2z_path) as reopened:
        assert reopened.nrows == N
        assert reopened.fare[50] == t.fare[50]
    print("   round-trip check: ok")


# ── 3. reblocking: copy with a 4× smaller block size ────────────────────────
# blocks controls how many *items* are packed per blosc2 block inside each
# chunk.  Smaller blocks give the query engine finer-grained index skipping
# but add more per-block overhead.  Indexes are automatically rebuilt at the
# new granularity.
original_blocks = t.blocks[0]  # e.g. 32 768 items
new_blocks = original_blocks // 4  # 4× smaller

with tempfile.TemporaryDirectory() as tmp:
    reblocked_path = os.path.join(tmp, "rides_reblocked.b2z")

    t0 = time.perf_counter()
    rb = t.copy(urlpath=reblocked_path, blocks=new_blocks)
    elapsed = time.perf_counter() - t0

    print(f"\n3. Reblocked copy (blocks {original_blocks} → {new_blocks}):")
    print(f"   time   : {elapsed:.4f}s")
    print(f"   blocks : {rb.blocks}")
    print(f"   cbytes : {rb.cbytes / 1e6:.2f} MB")

    # All column types must round-trip correctly
    assert rb.nrows == t.nrows
    assert rb[0].distance_km == t[0].distance_km
    assert (rb.payment[:5] == t.payment[:5]).all()  # dictionary column
    assert rb.waypoints[10] == t.waypoints[10]  # list column
    # fare null sentinels must be preserved
    null_orig = [i for i in range(N) if t.fare[i] == -1.0]
    null_copy = [i for i in range(N) if rb.fare[i] == -1.0]
    assert null_orig == null_copy
    print(f"   correctness: ok (fare nulls={len(null_orig)}, waypoints sample ok)")
    rb.close()


# ── 4. recompression: copy with a different codec ───────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    lz4_path = os.path.join(tmp, "rides_lz4.b2z")

    t0 = time.perf_counter()
    lz4_copy = t.copy(
        urlpath=lz4_path,
        cparams={"codec": b2.Codec.LZ4, "clevel": 5},
    )
    elapsed = time.perf_counter() - t0

    print("\n4. Recompressed copy (ZSTD → LZ4-5):")
    print(f"   time   : {elapsed:.4f}s")
    print(f"   cbytes : {lz4_copy.cbytes / 1e6:.2f} MB")

    assert lz4_copy.nrows == t.nrows
    assert np.array_equal(lz4_copy.fare[:10], t.fare[:10])
    print("   correctness: ok")
    lz4_copy.close()


# ── 5. selective copy: only a subset of columns ─────────────────────────────
# select() returns a view; copy() saves it to a new file.
with tempfile.TemporaryDirectory() as tmp:
    sel_path = os.path.join(tmp, "rides_selected.b2z")
    keep = ["distance_km", "duration_sec", "fare"]

    t0 = time.perf_counter()
    sel_copy = t.select(keep).copy(urlpath=sel_path)
    elapsed = time.perf_counter() - t0

    print(f"\n5. Selective copy ({keep}):")
    print(f"   time   : {elapsed:.4f}s")
    print(f"   columns: {sel_copy.col_names}")
    print(f"   cbytes : {sel_copy.cbytes / 1e6:.2f} MB")

    assert sel_copy.col_names == keep
    assert sel_copy.nrows == t.nrows
    sel_copy.close()


print("\nAll copies verified.")
