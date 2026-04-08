#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Schema layer: dataclass field specs, constraints, and validation.

from dataclasses import dataclass

import blosc2


@dataclass
class Product:
    id: int = blosc2.field(blosc2.int64(ge=0))
    price: float = blosc2.field(blosc2.float64(ge=0.0, le=10_000.0), default=0.0)
    stock: int = blosc2.field(blosc2.int32(ge=0), default=0)
    category: str = blosc2.field(blosc2.string(max_length=32), default="")
    on_sale: bool = blosc2.field(blosc2.bool(), default=False)


t = blosc2.CTable(Product)

# Valid row
t.append(Product(id=1, price=29.99, stock=100, category="electronics", on_sale=False))
t.append(Product(id=2, price=4.50, stock=200, category="food", on_sale=True))
print("Valid rows appended successfully.")
print(t)

# Inspect the compiled schema
print("Schema:")
for col in t.schema.columns:
    print(f"  {col.name:<12} dtype={col.dtype}  spec={col.spec}")

# Constraint violation: price < 0
try:
    t.append(Product(id=3, price=-1.0, stock=10, category="misc", on_sale=False))
except Exception as e:
    print(f"\nCaught validation error (price < 0): {e}")

# Constraint violation: id < 0
try:
    t.append(Product(id=-5, price=10.0, stock=10, category="misc", on_sale=False))
except Exception as e:
    print(f"Caught validation error (id < 0): {e}")

# String too long (max_length=32)
try:
    t.append(Product(id=4, price=1.0, stock=1, category="a" * 50, on_sale=False))
except Exception as e:
    print(f"Caught validation error (string too long): {e}")

print(f"\nTable still has {len(t)} valid rows.")
