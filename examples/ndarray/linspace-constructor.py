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

shape = (N,)
print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}) ***")
t0 = time()
a = blosc2.linspace(0, 10, N)
cratio = a.schunk.nbytes / a.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {a.schunk.cbytes / 1e6:.2f} MB (cratio: {cratio:.2f}x)"
)
print(f"Last 3 elements: {a[-3:]}")

# You can create ndim arrays too
shape = (5, N // 5)
chunks = None
# chunks = (5, N // 10)   # Uncomment this line to experiment with chunks
print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}, c_order: True) ***")
t0 = time()
b = blosc2.linspace(0, 10, N, shape=(5, N // 5), chunks=chunks, c_order=True)
cratio = b.schunk.nbytes / b.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {b.schunk.cbytes / 1e6:.2f} MB (cratio: {cratio:.2f}x)"
)

# You can go faster by not requesting the array to be C ordered (fun for users)
shape = (5, N // 5)
chunks = None
# chunks = (5, N // 10)   # Uncomment this line to experiment with chunks
print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}, c_order: False) ***")
t0 = time()
b = blosc2.linspace(0, 10, N, shape=(5, N // 5), chunks=chunks, c_order=False)
cratio = b.schunk.nbytes / b.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {b.schunk.cbytes / 1e6:.2f} MB (cratio: {cratio:.2f}x)"
)


# For reference, let's compare with numpy
print(f"*** Creating a numpy array with {N:_} elements (shape: {shape}) ***")
t0 = time()
na = np.linspace(0, 10, N).reshape(shape)
print(
    f"Time: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {na.nbytes / 1e6:.2f} MB"
)
# np.testing.assert_allclose(b[:], na)

# Create an NDArray from a numpy array
print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}) from numpy ***")
t0 = time()
c = blosc2.asarray(na)
cratio = c.schunk.nbytes / c.schunk.cbytes
print(
    f"Time: {time() - t0:.3f} s ({N / (time() - t0) / 1e6:.2f} M/s)"
    f"\tStorage required: {c.schunk.cbytes / 1e6:.2f} MB ({cratio:.2f}x)"
)
# np.testing.assert_allclose(c[:], na)

# In conclusion, you can use blosc2 linspace() to create blosc2 arrays requiring much less storage
# than numpy arrays.  If speed is important, and you can afford the extra memory, you can create
# blosc2 arrays faster straight from numpy arrays as well.
