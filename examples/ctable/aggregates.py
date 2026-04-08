#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Column aggregates: sum, min, max, mean, std, unique, value_counts,
# describe, and covariance matrix.

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Reading:
    sensor_id: int = blosc2.field(blosc2.int32(ge=0, le=9))
    temperature: float = blosc2.field(blosc2.float64(ge=-50.0, le=60.0), default=20.0)
    humidity: float = blosc2.field(blosc2.float64(ge=0.0, le=100.0), default=50.0)
    alert: bool = blosc2.field(blosc2.bool(), default=False)


rng = np.random.default_rng(42)
N = 500

station_ids = rng.integers(0, 10, size=N).astype(np.int32)
temperatures = rng.normal(20.0, 8.0, size=N).clip(-50, 60).astype(np.float64)
humidities = rng.uniform(30.0, 90.0, size=N).astype(np.float64)
alerts = rng.random(N) < 0.05

data = list(
    zip(station_ids.tolist(), temperatures.tolist(), humidities.tolist(), alerts.tolist(), strict=False)
)

t = blosc2.CTable(Reading, new_data=data)
print(f"Table: {len(t)} rows\n")

# -- per-column aggregates --------------------------------------------------
temp = t["temperature"]
print(f"temperature  sum  : {temp.sum():.2f}")
print(f"temperature  mean : {temp.mean():.2f}")
print(f"temperature  std  : {temp.std():.2f}")
print(f"temperature  min  : {temp.min():.2f}")
print(f"temperature  max  : {temp.max():.2f}")

print(f"\nalert  any : {t['alert'].any()}")
print(f"alert  all : {t['alert'].all()}")

# -- unique / value_counts --------------------------------------------------
print(f"\nsensor_id unique values : {t['sensor_id'].unique()}")
print(f"sensor_id value_counts  : {t['sensor_id'].value_counts()}")

# -- describe(): per-column summary printed to stdout -----------------------
print()
t.describe()

# -- cov(): covariance matrix of numeric columns ----------------------------
numeric = t.select(["sensor_id", "temperature", "humidity"])
cov = numeric.cov()
labels = ["sensor_id", "temperature", "humidity"]
col_w = 14
print("\nCovariance matrix:")
print(" " * 14 + "".join(f"{lbl:>{col_w}}" for lbl in labels))
for i, row_label in enumerate(labels):
    print(f"{row_label:<14}" + "".join(f"{cov[i, j]:>{col_w}.4f}" for j in range(3)))
