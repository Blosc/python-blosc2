#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Nullable columns: null_value sentinels, null-aware aggregates,
# is_null / notnull, sort nulls-last, Arrow null masking, CSV empty cells.
#
# CTable does not have a built-in "missing" bit per row like pandas does.
# Instead it uses a *sentinel value* approach: you choose a specific value
# that represents "null" for a column, and the library treats it
# transparently in aggregates, sorting, unique(), value_counts(), and
# Arrow export.
#
# This is especially useful for integer and string columns that have no
# natural null (unlike float, which can use NaN).

import os
import tempfile
from dataclasses import dataclass

import blosc2

# ---------------------------------------------------------------------------
# Schema with nullable columns
# ---------------------------------------------------------------------------
# Use null_value= on any spec to declare the sentinel.
# The sentinel bypasses validation constraints (ge/le etc.) so you can
# store it even when it would otherwise violate them.


@dataclass
class Reading:
    sensor_id: int = blosc2.field(blosc2.int32(ge=0))
    # -999 is "no reading" for temperature (normally ge=-50, le=60)
    temperature: float = blosc2.field(blosc2.float64(ge=-50.0, le=60.0, null_value=-999.0), default=-999.0)
    # "" is "unknown" for location (string)
    location: str = blosc2.field(blosc2.string(max_length=16, null_value=""), default="")
    # -1 is "not measured" for signal strength (normally ge=0, le=100)
    signal: int = blosc2.field(blosc2.int8(ge=0, le=100, null_value=-1), default=-1)


data = [
    (0, 22.3, "roof", 87),
    (1, -999.0, "cellar", 41),  # temperature unknown
    (2, 18.7, "", -1),  # location and signal unknown
    (3, 31.5, "garage", -1),  # signal unknown
    (4, -999.0, "", 62),  # temperature and location unknown
    (5, 15.1, "roof", 95),
]

t = blosc2.CTable(Reading, new_data=data)
print("Table with nullable columns:")
print(t)

# ---------------------------------------------------------------------------
# Detecting nulls
# ---------------------------------------------------------------------------
print("\n--- is_null() / notnull() ---")
temp_null = t["temperature"].is_null()
print(f"temperature is_null : {temp_null.tolist()}")
print(f"temperature null_count: {t['temperature'].null_count()}")

loc_null = t["location"].is_null()
print(f"location is_null   : {loc_null.tolist()}")

# Use notnull() as a filter mask
valid_temps = t["temperature"].to_numpy()[t["temperature"].notnull()]
print(f"Valid temperatures  : {valid_temps}")

# ---------------------------------------------------------------------------
# Null-aware aggregates
# ---------------------------------------------------------------------------
print("\n--- Aggregates skip null sentinels ---")
print(f"temperature.mean() = {t['temperature'].mean():.2f}   (only 3 non-null readings)")
print(f"temperature.min()  = {t['temperature'].min():.2f}")
print(f"temperature.max()  = {t['temperature'].max():.2f}")
print(f"signal.sum()       = {t['signal'].sum()}   (non-null: 87+41+62+95 = 285)")

# ---------------------------------------------------------------------------
# unique() and value_counts() exclude the null sentinel
# ---------------------------------------------------------------------------
print("\n--- unique / value_counts exclude null ---")
print(f"location unique     : {t['location'].unique().tolist()}")
print(f"signal value_counts : {t['signal'].value_counts()}")

# ---------------------------------------------------------------------------
# Appending: the sentinel bypasses validation constraints
# ---------------------------------------------------------------------------
print("\n--- Append with sentinel bypasses ge/le constraints ---")
# temperature has ge=-50, le=60; normally -999 would fail — but it's the sentinel
t.append((6, -999.0, "attic", 55))
print(f"Appended sensor 6 (temperature=null). Rows: {len(t)}")
assert t["temperature"].null_count() == 3

# ---------------------------------------------------------------------------
# Sorting: nulls always go last, regardless of ascending/descending
# ---------------------------------------------------------------------------
print("\n--- sort_by: nulls go last ---")
s_asc = t.sort_by("temperature")
print("Ascending (nulls last):")
print([round(v, 1) for v in s_asc["temperature"].to_numpy().tolist()])

s_desc = t.sort_by("temperature", ascending=False)
print("Descending (nulls still last):")
print([round(v, 1) for v in s_desc["temperature"].to_numpy().tolist()])

# ---------------------------------------------------------------------------
# Arrow interop: null sentinels become proper Arrow nulls
# ---------------------------------------------------------------------------
try:
    import pyarrow as _pa  # noqa: F401

    arrow = t.to_arrow()
    temp_col = arrow.column("temperature")
    loc_col = arrow.column("location")
    print("\n--- Arrow export ---")
    print(f"Arrow temperature null_count: {temp_col.null_count}")
    print(f"Arrow location null_count   : {loc_col.null_count}")
    print(f"Arrow temperature values    : {temp_col.to_pylist()}")
except ImportError:
    print("\npyarrow not installed — skipping Arrow demo.")

# ---------------------------------------------------------------------------
# CSV round-trip: empty cells become the null sentinel on import
# ---------------------------------------------------------------------------
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
    f.write("sensor_id,temperature,location,signal\n")
    f.write("10,25.1,lab,80\n")
    f.write("11,,office,\n")  # temperature and signal missing → sentinels
    f.write("12,18.3,,70\n")  # location missing → sentinel ""
    csv_path = f.name

print("\n--- from_csv with empty cells ---")
t2 = blosc2.CTable.from_csv(csv_path, Reading)
print(t2)
print(f"temperature null_count: {t2['temperature'].null_count()}")
print(f"signal null_count     : {t2['signal'].null_count()}")
print(f"location null_count   : {t2['location'].null_count()}")
os.unlink(csv_path)

# ---------------------------------------------------------------------------
# describe() shows null count for nullable columns
# ---------------------------------------------------------------------------
print("\n--- describe() ---")
t.describe()
