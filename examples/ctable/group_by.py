#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable.group_by(): single-key and multi-key grouping, convenience
# methods (.size / .count / .sum / .mean / .min / .max), multi-column
# aggregations, and filtered (where=) grouped output.

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Order:
    city: str = blosc2.field(blosc2.string(max_length=16))
    product: str = blosc2.field(blosc2.string(max_length=16))
    qty: int = blosc2.field(blosc2.int32())
    price: float = blosc2.field(blosc2.float64(nullable=True), default=0.0)


rng = np.random.default_rng(42)
N = 200

cities = rng.choice(["Paris", "London", "Berlin", "Rome", "Madrid"], size=N)
products = rng.choice(["Widget", "Gadget", "Doodad", "Thingy"], size=N)
qty = rng.integers(1, 10, size=N).astype(np.int32)
price = rng.lognormal(mean=3.0, sigma=0.5, size=N).round(2)
# Simulate ~4 % missing prices
price[rng.random(N) < 0.04] = np.nan

t = blosc2.CTable(Order, new_data=list(zip(cities, products, qty, price, strict=False)))
print(f"Orders table: {t.nrows} rows × {t.ncols} cols\n")

# -- Single-key grouping ---------------------------------------------------

print("=== Group by city, sorted ===============================================")
by_city = t.group_by("city", sort=True)

# Row count per city
print(by_city.size())

# Non-null price count per city
print(by_city.count("price"))

# Total and average price per city
print(by_city.sum("price"))
print(by_city.mean("price"))

# -- Single-key with multiple aggregations at once --------------------------

print("\n=== Multiple aggregations in one call ===================================")
print(by_city.agg({"price": ["sum", "mean", "min", "max"], "qty": "sum"}))

# -- Multi-key grouping -----------------------------------------------------

print("\n=== Group by city + product =============================================")
by_city_product = t.group_by(["city", "product"], sort=True)
print(by_city_product.agg({"qty": "sum", "price": "mean"}))

# -- Filtered aggregation (where= pushdown) ---------------------------------

print("\n=== Filtered: only Widget orders ========================================")
by_city_widgets = t.where(t.product == "Widget").group_by("city", sort=True)
print(by_city_widgets.agg({"qty": "sum", "price": "mean"}))

# -- Unsorted output (often faster) -----------------------------------------

print("\n=== Unsorted (hash order, faster for large tables) =======================")
print(t.group_by("city").size())
