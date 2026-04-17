#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Querying: where() filters, select() column projection, and chaining.

from dataclasses import dataclass

import blosc2


@dataclass
class Sale:
    id: int = blosc2.field(blosc2.int64(ge=0))
    region: str = blosc2.field(blosc2.string(max_length=16), default="")
    amount: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    returned: bool = blosc2.field(blosc2.bool(), default=False)


data = [
    (0, "North", 120.0, False),
    (1, "South", 340.0, False),
    (2, "North", 85.0, True),
    (3, "East", 210.0, False),
    (4, "West", 430.0, False),
    (5, "South", 60.0, True),
    (6, "East", 300.0, False),
    (7, "North", 500.0, False),
    (8, "West", 175.0, True),
    (9, "South", 220.0, False),
]

t = blosc2.CTable(Sale, new_data=data)

# -- where(): row filter ----------------------------------------------------
high_value = t.where(t["amount"] > 200)
print(f"Sales > $200: {len(high_value)} rows")
print(high_value)

not_returned = t.where(not t["returned"])
print(f"Not returned: {len(not_returned)} rows")

# -- chained filters (views are composable) ---------------------------------
north = t.where(t["region"] == "North")
north_big = north.where(north["amount"] > 100)
print(f"North region + amount > 100: {len(north_big)} rows")
print(north_big)

# -- select(): column projection (no data copy) -----------------------------
slim = t.select(["id", "amount"])
print("id + amount only:")
print(slim)

# -- combined: select columns, then filter rows -----------------------------
result = t.select(["region", "amount"]).where(not t["returned"])
print("Region + amount for non-returned sales:")
print(result)
