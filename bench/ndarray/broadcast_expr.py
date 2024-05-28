#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Small benchmark for evaluating outer products using the broadcast feature

from time import time

import numpy as np

import blosc2

N = 10_000
# N = 1_000
# chunks = 11
# blocks = 9
shape1, shape2 = (N, 1), (N,)

# Create a NDArray from a NumPy array
npa = np.arange(np.prod(shape1), dtype=np.int64).reshape(shape1)
npb = np.arange(np.prod(shape2), dtype=np.int64).reshape(shape2)
# a = blosc2.asarray(npa, chunks=(chunks, 1), blocks=(blocks, 1))
# b = blosc2.asarray(npb, chunks=chunks, blocks=blocks)
a = blosc2.asarray(npa)
b = blosc2.asarray(npb)

for codec in blosc2.Codec:
    if codec.value > blosc2.Codec.ZSTD.value:
        break
    print(f"Codec: {codec}")
    t0 = time()
    c = a * b
    # print(f"Elapsed time (expr): {time() - t0:.6f} s")
    t0 = time()
    # d = c.eval(cparams=dict(codec=codec, clevel=5), chunks=(chunks, chunks), blocks=(blocks, blocks))
    d = c.eval(cparams=dict(codec=codec, clevel=5))
    print(f"Elapsed time (eval): {time() - t0:.6f} s")
    # print(d[:])
    print(f"cratio: {d.schunk.cratio:.2f}x")
    # print(d.info)
