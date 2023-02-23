#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from time import time

import numpy as np

import blosc2

urlpath_sparse = "ex_formats_sparse.b2nd"
# urlpath_sparse = None
urlpath_contiguous = "ex_formats_contiguous.b2nd"
# urlpath_contiguous = None

blosc2.remove_urlpath(urlpath_sparse)
blosc2.remove_urlpath(urlpath_contiguous)

shape = (1000 * 1000,)
chunks = (100,)
blocks = (100,)
dtype = np.dtype(np.float64)

t0 = time()
a = blosc2.empty(shape, dtype=dtype, chunks=chunks, blocks=blocks, urlpath=urlpath_sparse, contiguous=False)
for nchunk in range(a.schunk.nchunks):
    a[nchunk * chunks[0] : (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
print(a.schunk.nchunks)
an = a[...]

t0 = time()
b = blosc2.empty(
    shape, dtype=dtype, chunks=chunks, blocks=blocks, urlpath=urlpath_contiguous, contiguous=True
)

print(b.schunk.nchunks)
for nchunk in range(shape[0] // chunks[0]):
    b[nchunk * chunks[0] : (nchunk + 1) * chunks[0]] = np.arange(chunks[0], dtype=dtype)
t1 = time()

print(f"Time: {(t1 - t0):.4f} s")
print(b.schunk.nchunks)
bn = b[...]

np.testing.assert_allclose(an, bn)
