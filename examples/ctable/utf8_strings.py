#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# utf8(): variable-length string columns stored Arrow-style (int64 offsets +
# UTF-8 bytes).  Each row costs exactly its encoded byte length — no
# max_length to pick, no truncation — and bulk reads materialize as NumPy
# StringDType arrays (requires NumPy >= 2.0).
#
# This is the recommended column type for general text (names, descriptions,
# high-cardinality values).  For low-cardinality categories prefer
# blosc2.dictionary(); for short near-uniform codes, fixed-width
# blosc2.string(max_length=N) still applies.

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Place:
    # utf8 is never inferred from a plain `str` annotation (that still maps
    # to fixed-width string(max_length=32)); it must be requested explicitly.
    name: str = blosc2.field(blosc2.utf8())
    note: str = blosc2.field(blosc2.utf8(nullable=True))
    visitors: int = blosc2.field(blosc2.int64())


data = {
    "name": ["café", "O'Hare", "日本語のテキスト", "zürich", "", "boulevard of broken dreams " * 2],
    "note": ["cozy", None, "multi-byte ok", None, "empty name above", "a very long name too"],
    "visitors": [120, 85_000, 42, 300, 0, 7],
}
t = blosc2.CTable(Place, new_data=data)
print(f"table: {t.nrows} rows; name dtype: {t['name'].dtype}")

# -- bulk reads return StringDType arrays ------------------------------------
names = t["name"][:]
print(f"bulk read: {type(names).__name__} of dtype {names.dtype}")
print("lengths vary freely:", [len(s) for s in names])

# -- filtering: use the operator form ----------------------------------------
# ==, !=, <, <=, >, >= against scalars or other utf8 columns are vectorized.
print("\nexact match:", t[t.name == "café"].nrows, "row(s)")
print("range (name >= 'p'):", t[t.name >= "p"].nrows, "row(s)")

# blosc2.startswith / endswith also work on utf8 columns.
started = np.asarray(blosc2.startswith(t.name, "bo").compute())
print("startswith 'bo':", int(started.sum()), "row(s)")

# Note: the string-expression form t.where("name == 'café'") is not
# supported for utf8 columns yet, and neither is create_index(); both raise
# NotImplementedError.  Use the operator form above, or a fixed-width
# string() column when an index is required.

# -- nulls are sentinel-based -------------------------------------------------
# Unlike vlstring (native None), nullable utf8 picks a sentinel string from
# the active null policy (or takes an explicit null_value=).  Reads surface
# the sentinel verbatim; use is_null()/fillna()/dropna() for null-aware work.
print("\nnull mask:", list(t["note"].is_null()))
print("fillna:", list(t["note"].fillna("<missing>"))[:3])

# -- groupby keys and sorting -------------------------------------------------
by_name = t.group_by("name", sort=True)
print("\nvisitors per name:")
print(by_name.sum("visitors"))

print("\nsorted by name (multi-byte values sort by Unicode code point):")
print(t.sort_by("name")[["name", "visitors"]])

# -- Arrow interop -------------------------------------------------------------
try:
    import pyarrow as pa  # noqa: F401

    at = t.to_arrow()
    print(f"\nArrow export: name -> {at.schema.field('name').type}")  # large_string
    # And Arrow string columns import back as utf8 columns automatically.
    t2 = blosc2.CTable.from_arrow(at.schema, at.to_batches())
    print(f"round-trip dtype: {t2['name'].dtype}")
except ImportError:
    print("\npyarrow not installed — skipping Arrow round-trip demo.")
