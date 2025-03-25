#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmarks of using the jit decorator with arbitrary NumPy functions.

import numpy as np
from time import time

import blosc2

N = 5_000   # working size of ~200 MB

# Create some sample data
t0 = time()
na = np.linspace(0, 1, N * N, dtype="float32").reshape(N, N)
nb = np.linspace(1, 2, N * N, dtype="float32").reshape(N, N)
nc = np.linspace(-10, 10, N, dtype="float32")
print(f"Time to create data (np.ndarray): {time() - t0:.3f} s")

t0 = time()
a = blosc2.linspace(0, 1, N * N, dtype="float32", shape=(N, N))
b = blosc2.linspace(1, 2, N * N, dtype="float32", shape=(N, N))
c = blosc2.linspace(-10, 10, 10, dtype="float32", shape=(N,))
print(f"Time to create data (NDArray): {time() - t0:.3f} s")
#print("a.chunks: ", a.chunks, "a.blocks: ", a.blocks)

# Compare with NumPy
def expr_numpy(a, b, c):
    # return np.cumsum(((na**3 + np.sin(na * 2)) < nc) & (nb > 0), axis=0)
    # The next is equally illustrative, but can achieve better speedups
    return np.sum(((na**3 + np.sin(na * 2)) < np.cumulative_sum(nc)) & (nb > 0), axis=1)

@blosc2.jit
def expr_jit(a, b, c):
    # return np.cumsum(((a**3 + np.sin(a * 2)) < c) & (b > 0), axis=0)
    return np.sum(((a**3 + np.sin(a * 2)) < np.cumulative_sum(c)) & (b > 0), axis=1)

# Call the NumPy function natively on NumPy containers
t0 = time()
result = expr_numpy(a, b, c)
tref = time() - t0
print(f"Time for native NumPy: {tref:.3f} s")

# Call the function with the jit decorator, using NumPy containers
t0 = time()
result = expr_jit(na, nb, nc)
print(f"Time for blosc2.jit (np.ndarray): {time() - t0:.3f} s, speedup: {tref / (time() - t0):.2f}x")

# Call the function with the jit decorator, using Blosc2 containers
t0 = time()
result = expr_jit(a, b, c)
print(f"Time for blosc2.jit (NDArray): {time() - t0:.3f} s, speedup: {tref / (time() - t0):.2f}x")
