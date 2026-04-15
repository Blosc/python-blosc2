#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np

import blosc2

# Intent: show that appending to a 1-D indexed array keeps the index sidecars
# usable, so indexed queries and sorted reads continue to work without an
# explicit rebuild after append().

dtype = np.dtype([("id", np.int64), ("payload", np.int32)])
data = np.array([(2, 20), (0, 0), (3, 30), (1, 10)], dtype=dtype)
arr = blosc2.asarray(data, chunks=(2,), blocks=(2,))

arr.create_index("id", kind=blosc2.IndexKind.FULL)

to_append = np.array([(6, 60), (4, 40), (5, 50)], dtype=dtype)
arr.append(to_append)

expr = blosc2.lazyexpr("(id >= 4) & (id < 7)", arr.fields).where(arr)

print("Indexed query after append:")
print(expr[:])

print("\nSorted rows after append:")
print(arr.sort(order="id")[:])
