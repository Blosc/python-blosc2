#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Querying: expression indexing, where() filters, projection, and chaining.

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
high_value = t.where(t.amount > 200)
print(f"Sales > $200: {len(high_value)} rows")
print(high_value)

# -- filtered aggregate pushdown -------------------------------------------
# For aggregate queries, pass the predicate directly with where= so Blosc2 can
# avoid materializing the filtered table view.
non_returned_revenue = t.amount.sum(where=~t.returned)
north_revenue = t.amount.sum(where=(t.region == "North") & ~t.returned)
print(f"Revenue for non-returned sales: ${non_returned_revenue:.2f}")
print(f"Revenue for non-returned North sales: ${north_revenue:.2f}")

not_returned = t["not returned"]
print(f"Not returned: {len(not_returned)} rows")

# -- chained filters (views are composable) ---------------------------------
north = t.where(t.region == "North")
north_big = north.where(north.amount > 100)
print(f"North region + amount > 100: {len(north_big)} rows")
print(north_big)

# -- column projection via [] (no data copy) --------------------------------
slim = t[["id", "amount"]]
print("id + amount only:")
print(slim)

# -- combined filter + projection -------------------------------------------
result = t.where("not returned", columns=["region", "amount"])
print("Region + amount for non-returned sales:")
print(result)
