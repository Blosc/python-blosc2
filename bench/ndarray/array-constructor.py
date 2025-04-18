#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from time import time

import numpy as np

import blosc2

N = 100_000_000

def info(a, t1):
    size = a.schunk.nbytes
    csize = a.schunk.cbytes
    print(
        f"Time: {t1:.3f} s - size: {size / 2 ** 30:.2f} GB ({size / t1 / 2 ** 30:.2f} GB/s)"
        f"\tStorage required: {csize / 2 ** 20:.2f} MB (cratio: {size / csize:.1f}x)"
    )


shape = (N,)
shape = (100, 1000, 1000)
print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}) ***")
t0 = time()
# a = blosc2.arange(N, shape=shape, dtype=np.int32, urlpath="a.b2nd", mode="w")
a = blosc2.linspace(0, 1, N, shape=shape, dtype=np.float64, urlpath="a.b2nd", mode="w")
info(a, time() - t0)
