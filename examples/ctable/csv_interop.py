#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CSV interop: generate a weather CSV, load it into a CTable, write it back.

import csv
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import blosc2


@dataclass
class WeatherReading:
    station_id: int = blosc2.field(blosc2.int32(ge=0, le=9999))
    temperature: float = blosc2.field(blosc2.float32(ge=-80.0, le=60.0), default=20.0)
    humidity: float = blosc2.field(blosc2.float32(ge=0.0, le=100.0), default=50.0)
    wind_speed: float = blosc2.field(blosc2.float32(ge=0.0, le=200.0), default=0.0)
    pressure: float = blosc2.field(blosc2.float32(ge=800.0, le=1100.0), default=1013.0)
    day_of_year: int = blosc2.field(blosc2.int16(ge=1, le=365), default=1)


# -- Generate a weather CSV -------------------------------------------------
rng = np.random.default_rng(42)
N = 1_000

station_ids = rng.integers(0, 100, size=N).tolist()
temperatures = [round(v, 2) for v in rng.normal(15.0, 12.0, N).clip(-80, 60).tolist()]
humidities = [round(v, 2) for v in rng.uniform(20.0, 95.0, N).tolist()]
wind_speeds = [round(v, 2) for v in rng.exponential(10.0, N).clip(0, 200).tolist()]
pressures = [round(v, 2) for v in rng.normal(1013.0, 8.0, N).clip(800, 1100).tolist()]
days = rng.integers(1, 366, size=N).tolist()

rows = list(zip(station_ids, temperatures, humidities, wind_speeds, pressures, days, strict=False))

tmpdir = Path(tempfile.mkdtemp(prefix="blosc2_csv_"))
csv_in = tmpdir / "weather.csv"
csv_out = tmpdir / "weather_out.csv"

# Write the CSV manually so the example is self-contained
with open(csv_in, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["station_id", "temperature", "humidity", "wind_speed", "pressure", "day_of_year"])
    writer.writerows(rows)

print(f"Generated {N} rows → {csv_in}")

# -- from_csv(): load into CTable -------------------------------------------
t = blosc2.CTable.from_csv(str(csv_in), WeatherReading)
print(f"Loaded into CTable: {len(t)} rows")
print(t.head())

# -- apply a filter before exporting ----------------------------------------
cold_days = t.where(t["temperature"] < 0)
print(f"\nCold days (temp < 0°C): {len(cold_days)} rows")
print(cold_days.head())

# -- to_csv(): write back to CSV --------------------------------------------
t.to_csv(str(csv_out))
print(f"\nFull table written to {csv_out}")

# Verify round-trip row count
with open(csv_out) as f:
    lines = f.readlines()
assert len(lines) == N + 1  # header + data rows
print(f"Round-trip verified: {len(lines) - 1} data rows in output CSV.")

# -- TSV variant ------------------------------------------------------------
tsv_out = tmpdir / "weather.tsv"
t.to_csv(str(tsv_out), sep="\t")
print(f"TSV variant written to {tsv_out}")

shutil.rmtree(tmpdir)
print("Temporary files removed.")
