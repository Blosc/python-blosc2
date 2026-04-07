#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# CTable basics: creation, append, extend, head/tail, len.

from dataclasses import dataclass

import numpy as np

import blosc2


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    price: float = blosc2.field(blosc2.float64(ge=0), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


# -- Create an empty table --------------------------------------------------
t = blosc2.CTable(Row)
print(f"Empty table: {len(t)} rows")

# -- append(): one row at a time --------------------------------------------
t.append(Row(id=0, price=1.5, active=True))
t.append(Row(id=1, price=2.3, active=False))
print(f"After 2 appends: {len(t)} rows")

# -- extend(): bulk load from a list of tuples ------------------------------
bulk = [(i, float(i) * 0.5, i % 2 == 0) for i in range(2, 10)]
t.extend(bulk)
print(f"After extend: {len(t)} rows")

# -- extend() from a structured numpy array ---------------------------------
arr = np.zeros(5, dtype=[("id", np.int64), ("price", np.float64), ("active", np.bool_)])
arr["id"] = np.arange(10, 15)
arr["price"] = np.linspace(10.0, 14.0, 5)
arr["active"] = [True, False, True, False, True]
t.extend(arr)
print(f"After numpy extend: {len(t)} rows\n")

# -- display: head / tail / full table --------------------------------------
print("head(3):")
print(t.head(3))

print("tail(3):")
print(t.tail(3))

print("Full table:")
print(t)

# -- basic properties -------------------------------------------------------
print(f"nrows  : {t.nrows}")
print(f"ncols  : {t.ncols}")
print(f"columns: {t.col_names}")
print(f"cbytes : {t.cbytes:,} B  (compressed)")
print(f"nbytes : {t.nbytes:,} B  (uncompressed)")
