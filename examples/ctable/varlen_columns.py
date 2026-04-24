from __future__ import annotations

from dataclasses import dataclass

import blosc2 as b2


@dataclass
class Product:
    code: str = b2.field(b2.string(max_length=8))
    ingredients: list[str] = b2.field(  # noqa: RUF009
        b2.list(b2.string(max_length=16), nullable=True, batch_rows=2)
    )


products = b2.CTable(Product)
products.append(("A1", ["salt", "water"]))
products.append(("B2", []))
products.append(("C3", None))
products.extend(
    [
        ("D4", ["flour", "oil"]),
        ("E5", ["cocoa"]),
    ]
)

print("ingredients:", products.ingredients[:])
print("tail:", products.tail(2).ingredients[:])

# Whole-cell replacement is explicit.
ing = products.ingredients[0]
ing.append("pepper")
products.ingredients[0] = ing
print("updated first row:", products.ingredients[0])

standalone = b2.ListArray(item_spec=b2.string(max_length=16), nullable=True)
standalone.extend([["a", "b"], [], None])
print("standalone list array:", standalone[:])
