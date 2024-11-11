#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate and store expressions with reductions,
# using NDArray instances as operands.
#
# For this to work correctly, we must use a string for the expression,
# as the reductions are normally evaluated eagerly.
# String-expressions also allow to be stored for later evaluation.
#
# Note how:
# 0) The expression can be evaluated and stored for later evaluation.
# 1) Re-opening a stored expression can adapt to changes in operands.
# 2) The expression can be evaluated lazily, only when needed.
# 3) Broadcasting is supported.

import numpy as np

import blosc2

# Create arrays with specific dimensions
a = blosc2.full((2, 3, 4), 1, dtype=np.int8, urlpath="a.b2nd", mode="w")
b = blosc2.full((2, 4), 2, dtype=np.uint16, urlpath="b.b2nd", mode="w")
c = blosc2.full((4,), 3, dtype=np.int8, urlpath="c.b2nd", mode="w")

# print("Array a:", a[:])
# print("Array b:", b[:])
# print("Array c:", c[:])

# Define an expression using the arrays above
# We can use a rich variety of functions, like sum, mean, std, sin, cos, etc.
# expr = "a.sum() + b * c"
# expr = "a.sum(axis=1) + b * c"
expr = "sum(a, axis=1) + b * sin(c)"
# Create a lazy expression
print("expr:", expr)
lazy_expr = blosc2.lazyexpr(expr)
print(f"expr shape: {lazy_expr.shape}; dtype: {lazy_expr.dtype}")
# Evaluate and print the result of the lazy expression (should be a 2x4 arr)
print(lazy_expr[:])

# Store and reload the expressions
url_path = "my_expr.b2nd"
lazy_expr.save(urlpath=url_path, mode="w")

url_path = "my_expr.b2nd"
# Open the saved file
lazy_expr = blosc2.open(urlpath=url_path)
print(lazy_expr)
print(f"expr (after open) shape: {lazy_expr.shape}; dtype: {lazy_expr.dtype}")
# Evaluate and print the result of the lazy expression (should be a 2x4 arr)
print(lazy_expr[:])

# Enlarge the arrays and re-evaluate the expression
a.resize((3, 3, 4))
a[2] = 3
b.resize((3, 4))
b[2] = 5
lazy_expr = blosc2.open(urlpath=url_path)  # Open the saved file
print(f"expr (after resize & reopen) shape: {lazy_expr.shape}; dtype: {lazy_expr.dtype}")
print(lazy_expr[:])
