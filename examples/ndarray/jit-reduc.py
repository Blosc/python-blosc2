#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Examples of using the jit decorator with reductions
# You can find benchmarks for this example in the bench/ndarray directory

import numpy as np

import blosc2


# Example 1: Basic usage of the jit decorator with reduction
@blosc2.jit
def expr_jit(a, b, c):
    # This function computes a sum reduction along axis 1
    return np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1)


# Create some sample data
a = blosc2.linspace(0, 1, 10 * 100, dtype="float32", shape=(10, 100))
b = blosc2.linspace(1, 2, 10 * 100, dtype="float32", shape=(10, 100))
c = blosc2.linspace(-10, 10, 100, dtype="float32", shape=(100,))

# Call the function with the jit decorator
result = expr_jit(a, b, c)
print("Example 1 result:", result)

# Example 2: Using the jit decorator with an out parameter for reduction
out = np.zeros((10,), dtype=np.int64)


@blosc2.jit
def expr_jit_out(a, b, c):
    # This function computes a sum reduction along axis 1 and stores the result in the 'out' array
    return np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1, out=out)


# Call the function with the jit decorator and out parameter
result_out = expr_jit_out(a, b, c)
print("Example 2 result:", result_out)
print("Example 2 out array:", out)  # The 'out' array should now contain the same result

# Example 3: Using the jit decorator with additional keyword arguments for reduction
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE])
out_cparams = blosc2.zeros((10,), dtype=np.int64, cparams=cparams)


@blosc2.jit
def expr_jit_cparams(a, b, c):
    # This function computes a sum reduction along axis 1 with custom compression parameters
    return np.sum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=1, out=out_cparams)


# Call the function with the jit decorator and custom parameters
result_cparams = expr_jit_cparams(a, b, c)
print("Example 3 result:", result_cparams[...])
print("Example 3 out array:", out_cparams[...])  # The 'out_cparams' array should now contain the same result
