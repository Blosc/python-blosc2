#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Mutations: delete, compact, sort_by, add/drop/rename columns, assign.

from dataclasses import dataclass

import blosc2


@dataclass
class Employee:
    id: int = blosc2.field(blosc2.int64(ge=0))
    name: str = blosc2.field(blosc2.string(max_length=16), default="")
    salary: float = blosc2.field(blosc2.float64(ge=0), default=0.0)


data = [
    (0, "Alice", 85_000.0),
    (1, "Bob", 72_000.0),
    (2, "Carol", 91_000.0),
    (3, "Dave", 65_000.0),
    (4, "Eve", 110_000.0),
    (5, "Frank", 78_000.0),
]

t = blosc2.CTable(Employee, new_data=data)
print("Original:")
print(t)

# -- delete(): logical deletion (tombstone) ---------------------------------
t.delete([1, 3])  # remove Bob and Dave
print(f"After deleting rows 1 and 3: {len(t)} live rows")
print(t)

# -- compact(): physically close the gaps -----------------------------------
t.compact()
print("After compact():")
print(t)

# -- sort_by(): returns a sorted copy by default ----------------------------
sorted_t = t.sort_by("salary", ascending=False)
print("Sorted by salary descending:")
print(sorted_t)

# -- sort_by(inplace=True) --------------------------------------------------
t.sort_by("name", inplace=True)
print("Sorted in-place by name:")
print(t)

# -- add_column(): new column filled with a default -------------------------
t.add_column("bonus", blosc2.float64(ge=0), default=0.0)
print("After add_column('bonus'):")
print(t)

# -- assign(): fill the new column with computed values ---------------------
bonuses = t["salary"][:] * 0.10
t["bonus"].assign(bonuses)
print("After assigning 10% bonuses:")
print(t)

# -- rename_column() --------------------------------------------------------
t.rename_column("bonus", "annual_bonus")
print(f"Column names after rename: {t.col_names}")

# -- drop_column() ----------------------------------------------------------
t.drop_column("annual_bonus")
print(f"Column names after drop: {t.col_names}")
