#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Storing data in sparse vs contiguous mode

from time import time

import numpy as np

import blosc2

urlpath_sparse = "ex_formats_sparse.b2nd"
urlpath_contiguous = "ex_formats_contiguous.b2nd"

shape = (1000 * 1000,)
chunks = (1000,)
blocks = (100,)
dtype = np.dtype(np.float64)

t0 = time()
a = blosc2.empty(
    shape,
    dtype=dtype,
    chunks=chunks,
    blocks=blocks,
    urlpath=urlpath_sparse,
    contiguous=False,
    mode="w",
)
for nchunk in range(a.schunk.nchunks):
    a[nchunk * chunks[0] : (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
an = a[...]

t0 = time()
b = blosc2.empty(
    shape,
    dtype=dtype,
    chunks=chunks,
    blocks=blocks,
    urlpath=urlpath_contiguous,
    contiguous=True,
    mode="w",
)

for nchunk in range(shape[0] // chunks[0]):
    b[nchunk * chunks[0] : (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
bn = b[...]

np.testing.assert_allclose(an, bn)
