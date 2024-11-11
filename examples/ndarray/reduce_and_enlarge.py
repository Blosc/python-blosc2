#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate and store expressions with reductions,
# using NDArray instances as operands.  Note how we must use a string
# for the expression, as the reductions are normally evaluated eagerly.
# Using strings prevents eager computation, and allows to store the
# expression for later evaluation.
# In particular, note how re-opening a stored expression can adapt to
# changes in the operands.

import numpy as np

import blosc2

# Create arrays with specific dimensions
a = blosc2.full((2, 3, 4), 1, urlpath="a.b2nd", mode="w")  # 3D array with dimensions (2, 3, 4)
b = blosc2.full((2, 4), 2, urlpath="b.b2nd", mode="w")  # 2D array with dimensions (2, 4)
c = blosc2.full(4, 3, dtype=np.int8, urlpath="c.b2nd", mode="w")  # 1D array with dimensions (4,)

print("Array a:", a[:])
print("Array b:", b[:])
print("Array c:", c[:])

# Define an expression using the arrays above
# expression = "a.sum() + b * c"
# expression = "a.sum(axis=1) + b * c"
# expression = "sum(a, axis=1) + b * c + 0"
expression = "c + np.int8(0)"
# expression = "sum(a, axis=1) + b * sin(c)"
# Define the operands for the expression
# operands = {"a": a, "b": b, "c": c}
operands = {"c": c}
# Create a lazy expression
print("expression:", expression, type(expression), operands["c"].dtype)
lazy_expression = blosc2.lazyexpr(expression, operands)
print(lazy_expression.shape, lazy_expression.dtype)  # Print the shape of the lazy expression
# lazy_expression = blosc2.lazyexpr(expression)
print(lazy_expression[:])  # Evaluate and print the result of the lazy expression (should be a 2x4 arr)

# Store and reload the expressions
url_path = "my_expr.b2nd"  # Define the path for the file
lazy_expression.save(urlpath=url_path, mode="w")  # Save the lazy expression to the specified path

url_path = "my_expr.b2nd"  # Define the path for the file
lazy_expression = blosc2.open(urlpath=url_path)  # Open the saved file
print(lazy_expression)  # Print the lazy expression
print(lazy_expression.shape, lazy_expression.dtype)  # Print the shape of the lazy expression
print(lazy_expression[:])  # Evaluate and print the result of the lazy expression (should be a 2x4 arr)
