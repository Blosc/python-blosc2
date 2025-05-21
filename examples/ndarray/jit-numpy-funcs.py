#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Examples of using the jit decorator with arbitrary NumPy functions.
# These functions are not optimized for performance, but they show how
# to use the jit decorator with NumPy functions.
# You can find benchmarks for this example in the bench/ndarray directory

import numpy as np

import blosc2

# Create some sample data
a = blosc2.linspace(0, 1, 10 * 100, dtype="float32", shape=(10, 100))
b = blosc2.linspace(1, 2, 10 * 100, dtype="float32", shape=(10, 100))
c = blosc2.linspace(-10, 10, 100, dtype="float32", shape=(100,))


# Example 1: Basic usage of the jit decorator with reduction
@blosc2.jit
def expr_jit(a, b, c):
    # This function computes a cumulative sum reduction along axis 0
    return np.cumsum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)


# Call the function with the jit decorator
result = expr_jit(a, b, c)
print(f"Example 1 result[0, 0:10]: {result[0, 0:10]}")


# Example 2: Using the jit decorator with an out parameter for reduction
out = np.zeros(result.shape, dtype=np.int64)


@blosc2.jit
def expr_jit_out(a, b, c):
    return np.cumulative_prod(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0, out=out, include_initial=False)


# Call the function with the jit decorator and out parameter
result = expr_jit_out(a, b, c)
print(f"Example 2 result[0, 0:10]: {result[0, 0:10]}")
print("Example 2 out[0, 0:10] array:", out[0, 0:10])  # the 'out' array should now contain the same result


# Example 3: Using the jit decorator with a combination of NumPy functions
@blosc2.jit
def expr_jit_diff(a, b, c):
    return np.diff((a**3 + np.cumsum(b * 2, axis=1) + c), axis=1)


# Call the function with the jit decorator and custom parameters
result = expr_jit_diff(a, b, c)
print(f"Example 3 result[0, 0:5]: {result[0, 0:5]}")
