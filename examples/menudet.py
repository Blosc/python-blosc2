"""Menudet 0.1 beta: one kernel, array and independent table-row execution."""

from dataclasses import dataclass

import numpy as np

import blosc2


@blosc2.menudet
def affine(x):
    return x * 2.0 - 1.0


x = np.arange(6, dtype="float64")
values = affine(x)[:]
np.testing.assert_array_equal(values, [-1, 1, 3, 5, 7, 9])
print("Array:", values)


@dataclass
class Row:
    amount: float = 0.0


table = blosc2.CTable(Row, new_data={"amount": x.tolist()}, create_summary_index=False)
kernel = affine.specialize(x)
table.add_computed_column("adjusted", kernel, inputs={"x": "amount"})
np.testing.assert_array_equal(table["adjusted"][:], values)
print("Table:", table["adjusted"][:])

# A reduction is a different result contract: one scalar per ORIGINAL group.


@blosc2.menudet(cardinality="block_scalar", partitions=(3,))
def total(x):
    return block_sum(x)  # noqa: F821


totals = total(x)[:]
np.testing.assert_array_equal(totals, [3, 12])
print("Block totals:", totals)
