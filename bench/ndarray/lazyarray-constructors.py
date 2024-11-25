#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This example shows how to use the `linspace()` constructor to create a blosc2 array.

from time import time

import numpy as np

import blosc2

N = 10_000_000

# Use a constructor inside a lazy expression
print("*** Using a constructor inside a lazy expression ***")
t0 = time()
o1 = blosc2.linspace(0, 10, N, shape=(5, N // 5))
la = blosc2.lazyexpr("o1 + 1")
print(f"Build time: {time() - t0:.3f} s")
t0 = time()
for i in range(5):
    _ = la[i]
print(f"Access time: {time() - t0:.3f} s")

t0 = time()
la = (o1 + 1).sum()
print(f"Build time (sum): {time() - t0:.3f} s")
t0 = time()
print("sum:", la)
print(f"Reduction time (sum): {time() - t0:.3f} s")

# Use a constructor inside a lazy expression (string form)
print("*** Using a constructor inside a lazy expression (string form) ***")
o1 = f"linspace(0, 10, {N}, shape=(5, {N} // 5))"
t0 = time()
la = blosc2.lazyexpr(f"{o1} + 1")
print(f"Build time: {time() - t0:.3f} s")
t0 = time()
for i in range(5):
    _ = la[i]
print(f"Access time: {time() - t0:.3f} s")

t0 = time()
la = blosc2.lazyexpr(f"sum({o1} + 1)")
print(f"Build time (sum): {time() - t0:.3f} s")
t0 = time()
print("sum:", la[()])
print(f"Reduction time (sum): {time() - t0:.3f} s")

# Compare with numpy
print("*** Comparison with numpy ***")
t0 = time()
o1 = np.linspace(0, 10, N).reshape(5, N // 5) + 1
print(f"Build time: {time() - t0:.3f} s")
t0 = time()
for i in range(5):
    _ = o1[i]
print(f"Access time: {time() - t0:.3f} s")

t0 = time()
print("sum:", o1.sum())
print(f"Reduction time (sum): {time() - t0:.3f} s")
