"""Menudet 1.0 draft: one kernel, array and independent table-row execution."""

from dataclasses import dataclass

import numpy as np

import blosc2

author = blosc2.DSLKernel.from_source("def affine(x):\n    return x * 2.0 - 1.0\n")
kernel = blosc2.PortableKernel.from_json(author.export({"x": "float64"}, "float64"))
x = np.arange(6, dtype="float64")
values = kernel.lazy({"x": x}, partitions=(3,))[:]
np.testing.assert_array_equal(values, [-1, 1, 3, 5, 7, 9])
print("Array:", values)


@dataclass
class Row:
    amount: float = 0.0


table = blosc2.CTable(Row, new_data={"amount": x.tolist()}, create_summary_index=False)
table.add_computed_column("adjusted", kernel, inputs={"x": "amount"})
np.testing.assert_array_equal(table["adjusted"][:], values)
print("Table:", table["adjusted"][:])

# A reduction is a different result contract: one scalar per ORIGINAL group.
total = blosc2.DSLKernel.from_source("def total(x):\n    return block_sum(x)\n")
record = total.export({"x": "float64"}, "float64", cardinality="block_scalar")
totals = blosc2.PortableKernel.from_json(record).lazy({"x": x}, partitions=(3,))[:]
np.testing.assert_array_equal(totals, [3, 12])
print("Block totals:", totals)
