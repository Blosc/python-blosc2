#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np

import blosc2

# Intent: show how a sorted index can be reused for direct sorted reads,
# sorted logical positions, and streaming ordered iteration.

dtype = np.dtype([("id", np.int64), ("score", np.float64)])
data = np.array(
    [
        (4, 0.3),
        (1, 1.5),
        (3, 0.8),
        (1, 0.2),
        (2, 3.1),
        (3, 0.1),
        (2, 1.2),
    ],
    dtype=dtype,
)

arr = blosc2.asarray(data, chunks=(4,), blocks=(2,))
arr.create_index("id", kind=blosc2.IndexKind.FULL)

print("Sorted rows via sorted index:")
print(arr.sort(order=["id", "score"])[:])

print("\nSorted logical positions:")
print(arr.argsort(order=["id", "score"])[:])

print("\nIterating in sorted order:")
for row in arr.iter_sorted(order=["id", "score"], batch_size=3):
    print(row)
