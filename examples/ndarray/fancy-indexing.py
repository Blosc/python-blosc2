#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Example showing fancy indexing (__getitem__) with integer arrays on
# 1-D and 3-D blosc2 NDArrays.
#
# Fancy indexing with integer arrays uses the same efficient
# b2nd_get_sparse_cbuffer() path as NDArray.take(), decompressing
# only the specific blocks holding the requested elements.
#
# This covers expressions like:
#   a[[0, 3, 5]]         — 1-D index on any dimensionality
#   a[[[0, 3], [5, 2]]]  — multi-dimensional index on any dimensionality
#
# Boolean masks and tuple fancy indexing (e.g. a[[0, 2], [1, 3]])
# still use the existing fancy-indexing machinery.

import numpy as np

import blosc2

# ============================================================
# 1-D array
# ============================================================
print("=== 1-D array ===")
a = blosc2.arange(20, dtype=np.int32)

# 1-D integer index
print(f"a = {a[:]}")
print(f"a[[0, 5, 12, 19]] = {a[[0, 5, 12, 19]]}")
print()

# Multi-dimensional integer index (2-D)
print(f"a[[[1, 3], [5, 7]]] =\n{a[[[1, 3], [5, 7]]]}")
print()

# ============================================================
# 3-D array
# ============================================================
print("=== 3-D array ===")
shape = (4, 5, 6)  # 120 elements total
a = blosc2.asarray(np.arange(120, dtype=np.float64).reshape(shape), chunks=(2, 3, 4), blocks=(2, 2, 2))

# 1-D index selects along axis 0
print(f"shape = {shape}")
print("a[[0, 2, 3]] — selects rows 0, 2, 3 along axis 0")
print(f"result shape: {a[[0, 2, 3]].shape}")
print(f"result:\n{a[[0, 2, 3]]}")
print()

# 2-D index — result shape = index shape + remaining dims
print("2-D index:")
print("a[[[0, 2], [3, 1]]]")
print(f"result shape: {a[[[0, 2], [3, 1]]].shape}")
print(f"result:\n{a[[[0, 2], [3, 1]]]}")
print()

# Negative and duplicate indices
print("Negative and duplicate indices:")
print(f"a[[-1, 0, -1, 2]] shape: {a[[-1, 0, -1, 2]].shape}")
print(f"result:\n{a[[-1, 0, -1, 2]]}")
print()

# Empty index
print("Empty index:")
print(f"a[[]] shape: {a[[]].shape}, value: {a[[]]}")
print()

# ============================================================
# Boolean masks
# ============================================================
print("=== Boolean mask ===")
mask = np.array([True, False, True, False])
print(f"mask = {mask.tolist()}")
print(f"a[mask] shape: {a[mask].shape}")
print(f"result:\n{a[mask]}")
print()

# ============================================================
# In summary
# ============================================================
print("=== Summary ===")
print("a[[0, 3, 5]]         — integer array on any dims → b2nd sparse gather")
print("a[[[0, 3], [5, 2]]]  — multi-dim integer array  → b2nd sparse gather")
print("a[[True, False, ...]] — boolean mask             → existing fancy path")
print("a[[0, 2], [1, 3]]    — tuple fancy indexing      → existing fancy path")
