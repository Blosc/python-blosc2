#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Real-world example: weather station data.
#
# Simulates a year of readings from 10 stations, then:
#   - filters to a single station
#   - finds the 5 hottest days
#   - computes correlations between meteorological variables
#   - saves the filtered data to disk and reloads it

import shutil
import tempfile
from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class WeatherReading:
    station_id: int = blosc2.field(blosc2.int32(ge=0, le=9))
    temperature: float = blosc2.field(blosc2.float32(ge=-80.0, le=60.0), default=20.0)
    humidity: float = blosc2.field(blosc2.float32(ge=0.0, le=100.0), default=50.0)
    wind_speed: float = blosc2.field(blosc2.float32(ge=0.0, le=200.0), default=0.0)
    pressure: float = blosc2.field(blosc2.float32(ge=800.0, le=1100.0), default=1013.0)
    day_of_year: int = blosc2.field(blosc2.int16(ge=1, le=365), default=1)


# -- Generate a full year of readings for 10 stations ----------------------
rng = np.random.default_rng(42)
N_STATIONS = 10
N_DAYS = 365
N = N_STATIONS * N_DAYS  # 3 650 rows

station_ids = np.tile(np.arange(N_STATIONS, dtype=np.int32), N_DAYS)
temperatures = rng.normal(15.0, 12.0, N).clip(-80, 60).astype(np.float32)
humidities = rng.uniform(20.0, 95.0, N).astype(np.float32)
wind_speeds = rng.exponential(10.0, N).clip(0, 200).astype(np.float32)
pressures = rng.normal(1013.0, 8.0, N).clip(800, 1100).astype(np.float32)
days = np.repeat(np.arange(1, N_DAYS + 1, dtype=np.int16), N_STATIONS)

arr = np.zeros(
    N,
    dtype=[
        ("station_id", np.int32),
        ("temperature", np.float32),
        ("humidity", np.float32),
        ("wind_speed", np.float32),
        ("pressure", np.float32),
        ("day_of_year", np.int16),
    ],
)
for col, val in [
    ("station_id", station_ids),
    ("temperature", temperatures),
    ("humidity", humidities),
    ("wind_speed", wind_speeds),
    ("pressure", pressures),
    ("day_of_year", days),
]:
    arr[col] = val

t = blosc2.CTable(WeatherReading, new_data=arr, validate=False)
print(f"Full dataset: {len(t):,} rows ({N_STATIONS} stations × {N_DAYS} days)")
t.info()

# -- Filter to station 3 ----------------------------------------------------
station3 = t.where(t["station_id"] == 3)
print(f"Station 3: {len(station3)} readings")
print(f"  mean temperature : {station3['temperature'].mean():.1f} °C")
print(f"  mean humidity    : {station3['humidity'].mean():.1f} %")
print(f"  mean wind speed  : {station3['wind_speed'].mean():.1f} km/h\n")

# -- 5 hottest days at station 3 (sort full table, then filter) ------------
sorted_by_temp = t.sort_by("temperature", ascending=False)
hottest_s3 = sorted_by_temp.where(sorted_by_temp["station_id"] == 3)
print("5 hottest days at station 3:")
print(hottest_s3.head(5))

# -- Covariance of numeric variables (all stations) -------------------------
numeric = t.select(["temperature", "humidity", "wind_speed", "pressure"])
cov = numeric.cov()
labels = ["temp", "humidity", "wind", "pressure"]
col_w = 11
print("Covariance matrix (all stations):")
print(" " * 10 + "".join(f"{lbl:>{col_w}}" for lbl in labels))
for i, lbl in enumerate(labels):
    print(f"{lbl:<10}" + "".join(f"{cov[i, j]:>{col_w}.3f}" for j in range(4)))

# -- Save station 3 data to disk and reload ---------------------------------
tmpdir = tempfile.mkdtemp(prefix="blosc2_weather_")
path = f"{tmpdir}/station3"
try:
    # Views cannot be sorted or saved directly — materialise via Arrow first
    s3_copy = blosc2.CTable.from_arrow(station3.to_arrow())
    s3_copy.sort_by("day_of_year", inplace=True)
    sorted_s3 = s3_copy
    sorted_s3.save(path, overwrite=True)
    print(f"\nStation 3 data saved to '{path}'")

    reloaded = blosc2.CTable.load(path)
    print(
        f"Reloaded: {len(reloaded)} rows, "
        f"days {reloaded['day_of_year'].min()}–{reloaded['day_of_year'].max()}"
    )
finally:
    shutil.rmtree(tmpdir)
    print("Temporary files removed.")
