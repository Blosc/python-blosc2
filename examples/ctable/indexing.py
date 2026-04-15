#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable indexing: create, query, rebuild, and drop indexes on columns.

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import blosc2


@dataclass
class Measurement:
    sensor_id: int = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float64())
    region: int = blosc2.field(blosc2.int32())


# -------------------------------------------------------------------------
# In-memory table
# -------------------------------------------------------------------------

t = blosc2.CTable(Measurement)
for i in range(200):
    t.append([i, 15.0 + (i % 30) * 0.5, i % 4])

# Create a bucket index on 'sensor_id' (default kind).
idx = t.create_index("sensor_id")
print("Index created:", idx)
print("Stale?", idx.stale)

# The where() call automatically uses the index when possible.
result = t.where(t["sensor_id"] > 180)
print("Rows with sensor_id > 180 (via index):", len(result))
print("sensor_ids:", sorted(int(v) for v in result["sensor_id"].to_numpy()))

# List all indexes on the table.
print("\nAll indexes:", t.indexes)

# After mutating the table the index is marked stale and where() falls
# back to a full scan automatically — results are still correct.
t.append([999, 42.0, 2])
idx = t.index("sensor_id")
print("\nAfter append, stale?", idx.stale)

result_stale = t.where(t["sensor_id"] == 999)
print("Row with sensor_id=999 (scan fallback):", len(result_stale))

# Rebuild the index to make it current again.
idx = t.rebuild_index("sensor_id")
print("\nAfter rebuild, stale?", idx.stale)

result_rebuilt = t.where(t["sensor_id"] == 999)
print("Row with sensor_id=999 (via rebuilt index):", len(result_rebuilt))

# Drop the index.
t.drop_index("sensor_id")
print("\nIndexes after drop:", t.indexes)

# -------------------------------------------------------------------------
# Persistent table
# -------------------------------------------------------------------------

tmpdir = Path(tempfile.mkdtemp())
path = str(tmpdir / "measurements.b2d")

try:
    pt = blosc2.CTable(Measurement, urlpath=path, mode="w")
    for i in range(200):
        pt.append([i, 15.0 + (i % 30) * 0.5, i % 4])

    pidx = pt.create_index("sensor_id")
    print("\nPersistent index created:", pidx)

    # Sidecar files are stored under <table.b2d>/_indexes/sensor_id/
    index_dir = Path(path) / "_indexes" / "sensor_id"
    sidecars = list(index_dir.glob("**/*.b2nd"))
    print(f"Sidecar files: {len(sidecars)}")

    result_p = pt.where(pt["sensor_id"] > 190)
    print("Rows sensor_id > 190:", len(result_p))

    # Close and reopen — the index catalog is persisted.
    del pt
    pt2 = blosc2.open(path)
    print("\nAfter reopen, indexes:", pt2.indexes)
    result_p2 = pt2.where(pt2["sensor_id"] > 190)
    print("Rows sensor_id > 190 (after reopen):", len(result_p2))

finally:
    shutil.rmtree(tmpdir, ignore_errors=True)
