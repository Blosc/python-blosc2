#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Persistence: write to disk, open read-only/read-write, save, load.

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
disk_path = f"{tmpdir}/measurements"
copy_path = f"{tmpdir}/measurements_copy"

try:
    # -- Create directly on disk (mode="w") ---------------------------------
    t = blosc2.CTable(Measurement, new_data=data, urlpath=disk_path, mode="w")
    print(f"Created on disk: {len(t):,} rows at '{disk_path}'")
    t.info()

    # -- Open read-only (default) -------------------------------------------
    ro = blosc2.CTable.open(disk_path)  # mode="r" by default
    print(f"Opened read-only: {len(ro):,} rows")
    print(f"  mean temperature: {ro['temperature'].mean():.2f}")

    try:
        ro.append(Measurement(sensor_id=0, temperature=20.0, day=1))
    except ValueError as e:
        print(f"  Write blocked (read-only): {e}")

    # -- Open read-write and mutate -----------------------------------------
    rw = blosc2.CTable.open(disk_path, mode="a")
    rw.append(Measurement(sensor_id=99, temperature=99.0, day=100))
    print(f"\nAfter append (read-write): {len(rw):,} rows")

    # -- save(): copy in-memory table to disk -------------------------------
    mem = blosc2.CTable(Measurement, new_data=data[:100])
    mem.save(copy_path)
    print(f"In-memory table saved to '{copy_path}'")

    # -- load(): pull a disk table fully into RAM ---------------------------
    ram = blosc2.CTable.load(disk_path)
    print(f"Loaded into RAM: {len(ram):,} rows  (cbytes={ram.cbytes:,})")
    assert len(ram) == len(rw)

finally:
    shutil.rmtree(tmpdir)
    print("\nTemporary files removed.")
