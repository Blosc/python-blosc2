#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Persistence: write to disk, open read-only/read-write, generic open(), save, load.

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Measurement:
    sensor_id: int = blosc2.field(blosc2.int32(ge=0))
    temperature: float = blosc2.field(blosc2.float64(), default=0.0)
    day: int = blosc2.field(blosc2.int16(ge=1, le=365), default=1)


rng = np.random.default_rng(0)
N = 10_000
data = [
    (int(rng.integers(0, 20)), float(rng.normal(15.0, 10.0)), int(rng.integers(1, 366))) for _ in range(N)
]

tmpdir = tempfile.mkdtemp(prefix="blosc2_ctable_")
disk_path = f"{tmpdir}/measurements.b2d"
zip_path = f"{tmpdir}/measurements.b2z"
unpacked_path = f"{tmpdir}/measurements_unpacked.b2d"
compact_zip_path = f"{tmpdir}/measurements_compact.b2z"
copy_path = f"{tmpdir}/measurements_copy.b2d"

try:
    # -- Create directly on disk (mode="w") ---------------------------------
    # Paths ending in .b2z create compact zip-backed stores; all other paths
    # create directory-backed stores.  A .b2d suffix is recommended for
    # directory-backed CTable stores.
    t = blosc2.CTable(Measurement, new_data=data, urlpath=disk_path, mode="w")
    print(f"Created on disk: {len(t):,} rows at '{disk_path}'")
    t.info()
    t.close()

    # -- Open read-only (default) -------------------------------------------
    ro = blosc2.CTable.open(disk_path)  # mode="r" by default
    print(f"Opened read-only: {len(ro):,} rows")
    print(f"  mean temperature: {ro['temperature'].mean():.2f}")

    try:
        ro.append(Measurement(sensor_id=0, temperature=20.0, day=1))
    except ValueError as e:
        print(f"  Write blocked (read-only): {e}")
    ro.close()

    # -- Generic open() materializes the CTable -----------------------------
    opened = blosc2.open(disk_path, mode="r")
    print(f"Generic open(): {type(opened).__name__} with {len(opened):,} rows")
    opened.close()

    # -- Open read-write and mutate -----------------------------------------
    rw = blosc2.CTable.open(disk_path, mode="a")
    rw.append(Measurement(sensor_id=99, temperature=99.0, day=100))
    print(f"\nAfter append (read-write): {len(rw):,} rows")
    rw.close()

    # -- save(): copy in-memory table to disk -------------------------------
    mem = blosc2.CTable(Measurement, new_data=data[:100])
    mem.save(copy_path)
    print(f"In-memory table saved to '{copy_path}'")

    # -- .b2d vs .b2z ------------------------------------------------------
    # .b2d is the recommended suffix for a directory-backed store: mutable,
    # easy to inspect, and a good default for local read/write workflows. .b2z is a single zip-backed file:
    # compact and convenient for moving/sharing, typically opened read-only.
    #
    # to_b2z()/to_b2d() use fast physical pack/unpack paths when possible: the
    # already-compressed leaves are copied as-is, without recompressing columns.
    rw = blosc2.CTable.open(disk_path, mode="r")
    rw.to_b2z(zip_path, overwrite=True)
    rw.close()
    print(f"Fast-packed .b2d -> .b2z: '{zip_path}'")

    zipped = blosc2.CTable.open(zip_path, mode="r")
    print(f"Opened .b2z read-only: {len(zipped):,} rows")
    zipped.to_b2d(unpacked_path, overwrite=True)
    zipped.close()
    print(f"Fast-unpacked .b2z -> .b2d: '{unpacked_path}'")

    # For a logical compacted copy (only live/visible rows), use compact=True
    # or save(). This may rewrite columns and is slower than physical packing.
    compacted = blosc2.CTable.open(disk_path, mode="r")
    compacted.to_b2z(compact_zip_path, overwrite=True, compact=True)
    compacted.close()
    print(f"Logical compacted copy: '{compact_zip_path}'")

    # -- load(): pull a disk table fully into RAM ---------------------------
    ram = blosc2.CTable.load(disk_path)
    print(f"Loaded into RAM: {len(ram):,} rows  (cbytes={ram.cbytes:,})")
    with blosc2.CTable.open(disk_path) as check:
        assert len(ram) == len(check)

finally:
    shutil.rmtree(tmpdir)
    print("\nTemporary files removed.")
