#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Examples of using the jit decorator with arbitrary NumPy functions.
# You can find benchmarks for this example in the bench/ndarray directory

import numpy as np
from time import time

import blosc2

N = 3000   # working size of ~200 MB

# Create some sample data
a = blosc2.linspace(0, 1, N * N, dtype="float32", shape=(N, N))
b = blosc2.linspace(1, 2, N * N, dtype="float32", shape=(N, N))
c = blosc2.linspace(-10, 10, 10, dtype="float32", shape=(N,))


# Example 1: Basic usage of the jit decorator with reduction
@blosc2.jit
def expr_jit(a, b, c):
    # This function computes a cumulative sum reduction along axis 0
    return np.cumsum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)

# Call the function with the jit decorator
t0 = time()
result = expr_jit(a, b, c)
print(f"Time for example 1: {time() - t0:.6f} s")


# Example 2: Using the jit decorator with an out parameter for reduction
out = np.zeros(result.shape, dtype=np.int64)
@blosc2.jit
def expr_jit_out(a, b, c):
    return np.cumulative_prod(((a**3 + np.sin(a * 2)) < c) & (b > 0),
                              axis=0, out=out, include_initial=False)

# Call the function with the jit decorator and out parameter
t0 = time()
result_out = expr_jit_out(a, b, c)
print(f"Time for example 2: {time() - t0:.6f} s")


# Example 3: Using the jit decorator with a combination of NumPy functions
@blosc2.jit
def expr_jit_diff(a, b, c):
    return np.diff((a**3 + np.cumsum(b * 2, axis=1) + c), axis=1)

# Call the function with the jit decorator and custom parameters
t0 = time()
result = expr_jit_diff(a, b, c)
print(f"Time for example 3: {time() - t0:.6f} s")
