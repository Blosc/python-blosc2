#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np

import blosc2

# Intent: show how to build an index on a derived expression stream and
# reuse it for both filtering and direct ordered reads.

dtype = np.dtype([("x", np.int64), ("payload", np.int32)])
data = np.array(
    [(-8, 0), (5, 1), (-2, 2), (11, 3), (3, 4), (-3, 5), (2, 6), (-5, 7)],
    dtype=dtype,
)

arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
arr.create_index(expression="abs(x)", kind=blosc2.IndexKind.FULL, name="abs_x")

expr = blosc2.lazyexpr("(abs(x) >= 2) & (abs(x) < 8)", arr.fields).where(arr)

print("Expression-indexed filter result:")
print(expr[:])

print("\nRows ordered by abs(x) via the full expression index:")
print(arr.sort(order="abs(x)")[:])

print("\nFiltered rows ordered by abs(x):")
print(expr.sort(order="abs(x)")[:])
