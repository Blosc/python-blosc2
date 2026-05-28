#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Example showing `blosc2.take()` on 1-D and 3-D arrays.
#
# `take()` follows the Array API shape rules:
#   - axis=None    : the array is flattened conceptually and the result
#                     has the same shape as *indices*.
#   - axis=<int>   : the indexed axis is replaced by *indices.shape*.
#
# Behind the scenes `take()` uses `b2nd_get_sparse_cbuffer()`, which
# decompresses *only* the specific blocks holding the requested elements.
# This is much more efficient than decompressing entire chunks, especially
# for large multi-dimensional arrays.

import numpy as np

import blosc2

# ============================================================
# 1-D array
# ============================================================
print("=== 1-D array ===")
a = blosc2.arange(20, dtype=np.int32)

# Take specific elements by flat index (axis=None is the default)
result = blosc2.take(a, [0, 5, 12, 19])
print(f"a = {a[:]}")
print(f"blosc2.take(a, [0, 5, 12, 19]) = {result[:]}")
print()

# Multi-dimensional index array: result shape = indices shape
result = blosc2.take(a, [[1, 3], [5, 7]])
print(f"blosc2.take(a, [[1, 3], [5, 7]]) =\n{result[:]}")
print()

# ============================================================
# 3-D array, axis=None (flattened)
# ============================================================
print("=== 3-D array, axis=None (flattened) ===")
shape = (4, 5, 6)  # 120 elements total
a = blosc2.asarray(np.arange(120, dtype=np.float64).reshape(shape), chunks=(2, 3, 4), blocks=(2, 2, 2))

# Flat indices into the 120-element buffer
result = blosc2.take(a, [0, 50, 119])
print(f"shape = {shape}")
print(f"blosc2.take(a, [0, 50, 119]) = {result[:]}")
print()

# ============================================================
# 3-D array, axis=0 (gather along first dimension)
# ============================================================
print("=== 3-D array, axis=0 ===")
result = blosc2.take(a, [0, 2, 3], axis=0)
print(f"shape = {shape}, axis=0, indices = [0, 2, 3]")
print(f"result shape: {result.shape}")
print(f"result:\n{result[:]}")
print()

# ============================================================
# 3-D array, axis=1 (gather along second dimension)
# ============================================================
print("=== 3-D array, axis=1 ===")
result = blosc2.take(a, [0, 3, 4], axis=1)
print(f"shape = {shape}, axis=1, indices = [0, 3, 4]")
print(f"result shape: {result.shape}")
print(f"result:\n{result[:]}")
print()

# ============================================================
# 3-D array, axis=2 (gather along third dimension)
# ============================================================
print("=== 3-D array, axis=2 ===")
result = blosc2.take(a, [0, 3, 5], axis=2)
print(f"shape = {shape}, axis=2, indices = [0, 3, 5]")
print(f"result shape: {result.shape}")
print(f"result:\n{result[:]}")
print()

# ============================================================
# Multi-dimensional indices and negative/duplicate indices
# ============================================================
print("=== Multi-dimensional indices (axis=1) ===")
result = blosc2.take(a, [[0, 3], [2, 4]], axis=1)
print(f"shape = {shape}, axis=1")
print("indices (2-D) = [[0, 3], [2, 4]]")
print(f"result shape: {result.shape}")
print(f"result:\n{result[:]}")
print()

print("=== Negative and duplicate indices (axis=1) ===")
result = blosc2.take(a, [-1, 0, -1, 2, 2], axis=1)
print("indices = [-1, 0, -1, 2, 2]")
print(f"result shape: {result.shape}")
print(f"result:\n{result[:]}")
