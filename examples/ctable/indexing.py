#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable indexing with mixed dtypes, persistent sidecars, and a packed .b2z bundle.

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import blosc2


@dataclass
class Measurement:
    sensor_id: int = blosc2.field(blosc2.int32())
    temperature: float = blosc2.field(blosc2.float64())
    region: str = blosc2.field(blosc2.string(max_length=12), default="")
    active: bool = blosc2.field(blosc2.bool(), default=True)
    status: str = blosc2.field(blosc2.string(max_length=12), default="")


def load_rows(table: blosc2.CTable, nrows: int = 240) -> None:
    regions = ["north", "south", "east", "west"]
    for i in range(nrows):
        region = regions[i % len(regions)]
        active = (i % 7) not in (0, 6)
        status = "alert" if i % 23 == 0 else ("warm" if i % 11 == 0 else "ok")
        table.append([i, 12.5 + (i % 40) * 0.35, region, active, status])


bundle_path = Path("indexed_measurements.b2z").resolve()
workspace = Path(tempfile.mkdtemp())
table_path = workspace / "indexed_measurements.b2d"

pt = None
packed = None

try:
    print("Creating a CTable with mixed dtypes...")
    pt = blosc2.CTable(Measurement, urlpath=str(table_path), mode="w")
    load_rows(pt)

    # Create a couple of indexes on columns with different dtypes.
    print("\nCreating indexes...")
    idx_sensor = pt.create_index("sensor_id", kind=blosc2.IndexKind.FULL)
    idx_active = pt.create_index("active")
    print("Indexes created:", pt.indexes)
    print("sensor_id stale?", idx_sensor.stale)
    print("active stale?", idx_active.stale)

    # Queries can combine indexed and non-indexed predicates.
    recent_active = pt.where((pt["sensor_id"] >= 180) & pt["active"] & (pt["region"] == "north"))
    print("\nLive rows with sensor_id >= 180, active=True, region='north':", len(recent_active))
    print("sensor_ids:", recent_active["sensor_id"])
    print("statuses:", recent_active["status"].to_numpy())

    # Close the table, pack the TreeStore into a single .b2z file, and reopen it.
    del pt
    pt = None

    if bundle_path.exists():
        bundle_path.unlink()

    store = blosc2.TreeStore(str(table_path), mode="r")
    try:
        packed_path = store.to_b2z(filename=str(bundle_path), overwrite=True)
    finally:
        store.close()

    print(f"\nPacked bundle created at: {packed_path}")

    packed = blosc2.open(str(bundle_path), mode="r")
    print("Reopened object type:", type(packed).__name__)
    print("Indexes after reopen from .b2z:", packed.indexes)

    # Query directly against the .b2z bundle; no unpack step is needed.
    warm_active = packed.where(packed["active"] & (packed["status"] == "warm") & (packed["sensor_id"] > 100))
    print("\nRows from .b2z with active=True, status='warm', sensor_id > 100:", len(warm_active))
    print("sensor_ids:", warm_active["sensor_id"])
    print("regions:", warm_active["region"].to_numpy())

    print("\nThe packed file is kept on disk.")
    print(f"Inspect it later with: f = blosc2.open({bundle_path.name!r}, mode='r')")
    print("Then call: f.info()")
    print("For a quick check of the available info entry point, print: f.info")

finally:
    if packed is not None:
        del packed
    if pt is not None:
        del pt
    shutil.rmtree(workspace, ignore_errors=True)
