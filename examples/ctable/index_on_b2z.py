"""Demonstrate that CTable indexes survive a .b2z round-trip.

Steps
-----
1. Build a small CTable with synthetic sensor data and save it as .b2z.
2. Measure query speed with full scan (no index).
3. Reopen in append mode, create FULL indexes, close (triggers rezip).
4. Reopen read-only — indexes are present and queries are faster.
"""

import os
import shutil
import time
from dataclasses import dataclass

import numpy as np

import blosc2

# ---------------------------------------------------------------------------
# 1. Schema and synthetic data
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    sensor_id: int = blosc2.field(blosc2.int32())
    timestamp: int = blosc2.field(blosc2.int64())
    value: float = blosc2.field(blosc2.float64())
    active: bool = blosc2.field(blosc2.bool())


N = 5_000_000
rng = np.random.default_rng(42)
B2D = "/tmp/sensors.b2d"
B2Z = "/tmp/sensors.b2z"

for p in (B2D, B2Z):
    if os.path.exists(p):
        (shutil.rmtree if os.path.isdir(p) else os.remove)(p)

# ---------------------------------------------------------------------------
# 2. Create and zip
# ---------------------------------------------------------------------------

print(f"Creating CTable with {N:,} rows ...")
ct = blosc2.CTable(Reading, urlpath=B2D, mode="w", expected_size=N)
ct.extend(
    {
        "sensor_id": rng.integers(0, 100, N, dtype=np.int32),
        "timestamp": np.arange(N, dtype=np.int64),
        "value": rng.uniform(-50.0, 150.0, N),
        "active": rng.integers(0, 2, N).astype(bool),
    }
)
ct.close()

store = blosc2.TreeStore(B2D, mode="r")
store.to_b2z(filename=B2Z, overwrite=True)
store.discard()
shutil.rmtree(B2D)
print(f"  saved → {B2Z}  ({os.path.getsize(B2Z) / 1e6:.1f} MB)")

# ---------------------------------------------------------------------------
# 3. Baseline: full scan (no index)
# ---------------------------------------------------------------------------

QUERIES = [
    ("value > 100.0", lambda ct: ct.where(ct["value"] > 100.0)),
    ("value > 120.0", lambda ct: ct.where(ct["value"] > 120.0)),
    ("value between 0 and 10", lambda ct: ct.where((ct["value"] >= 0.0) & (ct["value"] <= 10.0))),
    ("timestamp > 450_000", lambda ct: ct.where(ct["timestamp"] > 4_500_000)),
    ("timestamp > 4_999_000", lambda ct: ct.where(ct["timestamp"] > 4_999_000)),
]


def bench(fn, reps=5):
    times = [0.0] * reps
    for i in range(reps):
        t = time.perf_counter()
        result = fn()
        times[i] = (time.perf_counter() - t) * 1000
    return min(times), result


print("\n--- Full scan (no index) ---")
baseline = {}
ct = blosc2.CTable.open(B2Z, mode="r")
assert not ct.indexes, "expected no indexes yet"
for label, fn in QUERIES:
    ms, view = bench(lambda fn=fn: fn(ct))
    baseline[label] = (ms, len(view))
    print(f"  {label:<38}  {ms:7.1f} ms   {len(view):>8,} rows")
ct.close()

# ---------------------------------------------------------------------------
# 4. Build indexes (append mode → rezip on close)
# ---------------------------------------------------------------------------

print("\nBuilding indexes (mode='a') ...")
ct = blosc2.CTable.open(B2Z, mode="a")

t0 = time.perf_counter()
ct.create_index("value", kind=blosc2.IndexKind.FULL)
print(f"  value     FULL index  {(time.perf_counter() - t0) * 1000:.0f} ms")

t0 = time.perf_counter()
ct.create_index("timestamp", kind=blosc2.IndexKind.FULL)
print(f"  timestamp FULL index  {(time.perf_counter() - t0) * 1000:.0f} ms")

t0 = time.perf_counter()
ct.close()
print(
    f"  closed + rezipped     {(time.perf_counter() - t0) * 1000:.0f} ms  "
    f"({os.path.getsize(B2Z) / 1e6:.1f} MB)"
)

# ---------------------------------------------------------------------------
# 5. Read-only: verify indexes survived, benchmark
# ---------------------------------------------------------------------------

print("\nReopening .b2z read-only ...")
ct = blosc2.CTable.open(B2Z, mode="r")
found = [idx.col_name for idx in ct.indexes]
print(f"  indexes present: {found}")
assert "value" in found, "index for 'value' missing after round-trip!"
assert "timestamp" in found, "index for 'timestamp' missing after round-trip!"

print()
print(f"{'query':<38}  {'no index':>9}  {'indexed':>9}  {'speedup':>8}  {'rows':>8}")
print("-" * 78)

for label, fn in QUERIES:
    i_ms, view = bench(lambda fn=fn: fn(ct))
    b_ms, b_n = baseline[label]
    sp = b_ms / i_ms if i_ms > 0 else float("inf")
    assert len(view) == b_n, f"row count mismatch for {label!r}"
    print(f"  {label:<38}  {b_ms:8.1f}ms  {i_ms:8.1f}ms  {sp:7.1f}x  {len(view):>8,}")

ct.close()

# ---------------------------------------------------------------------------
# 6. Cleanup
# ---------------------------------------------------------------------------

os.remove(B2Z)
print("\nDone.")
