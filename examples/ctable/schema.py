#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Schema layer: dataclass field specs, constraints, validation, and null_value.

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Product:
    id: int = blosc2.field(blosc2.int64(ge=0))
    price: float = blosc2.field(blosc2.float64(ge=0.0, le=10_000.0), default=0.0)
    stock: int = blosc2.field(blosc2.int32(ge=0), default=0)
    updated_at: np.datetime64 = blosc2.field(  # noqa: RUF009
        blosc2.timestamp(unit="us"), default=np.datetime64("1970-01-01", "us")
    )
    # null_value="" means an empty string represents "unknown category"
    category: str = blosc2.field(blosc2.string(max_length=32, null_value=""), default="")
    on_sale: bool = blosc2.field(blosc2.bool(), default=False)


t = blosc2.CTable(Product)

# Valid row
t.append(
    Product(
        id=1,
        price=29.99,
        stock=100,
        updated_at=np.datetime64("2025-01-01T09:00:00", "us"),
        category="electronics",
        on_sale=False,
    )
)
t.append(
    Product(
        id=2,
        price=4.50,
        stock=200,
        updated_at=np.datetime64("2025-01-02T10:30:00", "us"),
        category="food",
        on_sale=True,
    )
)
# "" is the null sentinel for category — stored as-is, not rejected
t.append(
    Product(
        id=3,
        price=0.0,
        stock=0,
        updated_at=np.datetime64("2025-01-03T12:00:00", "us"),
        category="",
        on_sale=False,
    )
)
print("Valid rows appended successfully.")
print(t)

# Inspect the compiled schema
print("Schema:")
for col in t.schema.columns:
    nv = getattr(col.spec, "null_value", None)
    nv_str = f"  null_value={nv!r}" if nv is not None else ""
    print(f"  {col.name:<12} dtype={col.dtype}  spec={col.spec}{nv_str}")

# Constraint violation: price < 0
try:
    t.append(
        Product(
            id=4,
            price=-1.0,
            stock=10,
            updated_at=np.datetime64("2025-01-04", "us"),
            category="misc",
            on_sale=False,
        )
    )
except Exception as e:
    print(f"\nCaught validation error (price < 0): {e}")

# Constraint violation: id < 0
try:
    t.append(
        Product(
            id=-5,
            price=10.0,
            stock=10,
            updated_at=np.datetime64("2025-01-05", "us"),
            category="misc",
            on_sale=False,
        )
    )
except Exception as e:
    print(f"Caught validation error (id < 0): {e}")

# String too long (max_length=32)
try:
    t.append(
        Product(
            id=5,
            price=1.0,
            stock=1,
            updated_at=np.datetime64("2025-01-06", "us"),
            category="a" * 50,
            on_sale=False,
        )
    )
except Exception as e:
    print(f"Caught validation error (string too long): {e}")

print(f"\nTable still has {len(t)} valid rows.")
print(f"category null_count: {t['category'].null_count()} (the row with category='' is null)")
