#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Examples of using the jit decorator with expressions
# You can find benchmarks for this example in the bench/ndarray directory

import numpy as np

import blosc2


# Example 1: Basic usage of the jit decorator
@blosc2.jit
def expr_jit(a, b, c):
    # This function computes a boolean array where the condition is met
    return ((a**3 + np.sin(a * 2)) < c) & (b > 0)


# Create some sample data
a = blosc2.linspace(0, 1, 10 * 100, dtype="float32", shape=(10, 100))
b = blosc2.linspace(1, 2, 10 * 100, dtype="float32", shape=(10, 100))
c = blosc2.linspace(-10, 10, 100, dtype="float32", shape=(100,))

# Call the function with the jit decorator
result = expr_jit(a, b, c)
print(result[1, :10])

# Example 2: Using the jit decorator with an out parameter
out = blosc2.zeros((10, 100), dtype=np.bool_)


@blosc2.jit(out=out)
def expr_jit_out(a, b, c):
    # This function computes a boolean array and stores the result in the 'out' array
    return ((a**3 + np.sin(a * 2)) < c) & (b > 0)


# Call the function with the jit decorator and out parameter
result_out = expr_jit_out(a, b, c)
print(result_out[1, :10])
print(out[1, :10])  # The 'out' array should now contain the same result

# Example 3: Using the jit decorator with additional keyword arguments
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE])


@blosc2.jit(cparams=cparams)
def expr_jit_cparams(a, b, c):
    # This function computes a boolean array with custom compression parameters
    return ((a**3 + np.sin(a * 2)) < c) & (b > 0)


# Call the function with the jit decorator and custom parameters
result_cparams = expr_jit_cparams(a, b, c)
print(result_cparams[1, :10])
