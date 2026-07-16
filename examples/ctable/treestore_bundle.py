#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Bundling CTables and NDArrays together in a single TreeStore file.
#
# A TreeStore can hold both plain NDArrays and CTable objects in the same
# .b2d directory or .b2z zip archive.  Each CTable is stored inline as a
# named subtree — its columns, metadata, and index sidecars all live as
# normal Blosc2 leaves inside the bundle.  From the outside the CTable
# appears as a single key, just like any other leaf.

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Sensor:
    sensor_id: int = blosc2.field(blosc2.int32(ge=0))
    temperature: float = blosc2.field(blosc2.float64(), default=0.0)
    day: int = blosc2.field(blosc2.int16(ge=1, le=365), default=1)


@dataclass
class Event:
    event_id: int = 0
    label: str = ""


tmpdir = tempfile.mkdtemp(prefix="blosc2_bundle_")
b2d_path = f"{tmpdir}/dataset.b2d"
b2z_path = f"{tmpdir}/dataset.b2z"

try:
    # ------------------------------------------------------------------
    # 1. Build some in-memory data
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    N = 500

    sensors = blosc2.CTable(Sensor)
    for _ in range(N):
        sensors.append(
            Sensor(
                sensor_id=int(rng.integers(0, 5)),
                temperature=float(rng.normal(20.0, 5.0)),
                day=int(rng.integers(1, 366)),
            )
        )

    events = blosc2.CTable(Event)
    for i in range(20):
        events.append(Event(event_id=i, label=f"evt-{i:03d}"))

    calibration = blosc2.linspace(0.0, 1.0, 10)
    timestamps = np.arange(N, dtype=np.int64)

    # ------------------------------------------------------------------
    # 2. Write a .b2d bundle mixing CTables and NDArrays
    # ------------------------------------------------------------------
    with blosc2.TreeStore(b2d_path, mode="w") as ts:
        # NDArrays sit alongside CTables — same assignment syntax
        ts["/raw/calibration"] = calibration
        ts["/raw/timestamps"] = timestamps

        # CTables are stored inline; internals are hidden from normal traversal
        ts["/tables/sensors"] = sensors
        ts["/tables/events"] = events

        print("Keys written to .b2d bundle:")
        for k in sorted(ts.keys()):
            print(f"  {k}")

    # ------------------------------------------------------------------
    # 3. Read the .b2d bundle back
    # ------------------------------------------------------------------
    print("\nReading .b2d bundle:")
    with blosc2.open(b2d_path, mode="r") as ts:
        # CTable is returned transparently
        s = ts["/tables/sensors"]
        print(f"  /tables/sensors  : {type(s).__name__}, {len(s):,} rows")
        print(f"    mean temp      : {s['temperature'].mean():.2f}")

        e = ts["/tables/events"]
        print(f"  /tables/events   : {type(e).__name__}, {len(e)} rows")

        cal = ts["/raw/calibration"]
        print(f"  /raw/calibration : {type(cal).__name__}, shape={cal.shape}")

        # Object internals are not exposed in keys()
        assert "/tables/sensors/_meta" not in ts
        assert "/tables/sensors/_cols" not in ts

    # ------------------------------------------------------------------
    # 4. Pack the .b2d bundle into a single .b2z archive
    # ------------------------------------------------------------------
    with blosc2.TreeStore(b2d_path, mode="r") as ts:
        ts.to_b2z(filename=b2z_path)

    print(f"\nPacked to .b2z: {b2z_path}")

    # ------------------------------------------------------------------
    # 5. Read directly from the .b2z archive (offset-based, no extraction)
    # ------------------------------------------------------------------
    print("\nReading .b2z archive:")
    with blosc2.open(b2z_path, mode="r") as ts:
        s2 = ts["/tables/sensors"]
        print(f"  /tables/sensors : {type(s2).__name__}, {len(s2):,} rows")

        e2 = ts["/tables/events"]
        print(f"  /tables/events  : {type(e2).__name__}, {len(e2)} rows")
        print(f"  first event     : event_id={e2[0]['event_id']}, label='{e2[0]['label']}'")

    # ------------------------------------------------------------------
    # 6. Append a new row to a CTable inside the bundle
    # ------------------------------------------------------------------
    with blosc2.TreeStore(b2d_path, mode="a") as ts:
        s3 = ts["/tables/sensors"]
        s3.append(Sensor(sensor_id=99, temperature=-10.0, day=1))
        print(f"\nAfter append: /tables/sensors has {len(s3):,} rows")
        # Closing the table explicitly is optional; TreeStore.__exit__ handles it
        s3.close()

    # ------------------------------------------------------------------
    # 7. Delete a CTable from the bundle
    # ------------------------------------------------------------------
    with blosc2.TreeStore(b2d_path, mode="a") as ts:
        del ts["/tables/events"]
        print(f"\nAfter deleting /tables/events, keys: {sorted(ts.keys())}")

finally:
    shutil.rmtree(tmpdir)
    print("\nTemporary files removed.")
