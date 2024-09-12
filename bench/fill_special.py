#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import sys
from time import time

import numpy as np

import blosc2

# Dimensions, type and persistence properties for the arrays
shape = 1_000 * 1_000
chunksize = 10_000
blocksize = 1_000

dtype = np.float64

nchunks = shape // chunksize
# Set the compression and decompression parameters
cparams = {"codec": blosc2.Codec.BLOSCLZ, "typesize": 8, "blocksize": blocksize * 8}
dparams = {}
contiguous = True
persistent = bool(sys.argv[1]) if len(sys.argv) > 1 else False

if persistent:
    urlpath = "bench_fill_special.b2frame"
else:
    urlpath = None


def create_schunk(data=None):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)
    # Create the empty SChunk
    return blosc2.SChunk(chunksize=chunksize * cparams["typesize"], data=data, **storage)

t0 = time()
schunk = create_schunk(data=np.full(shape, np.pi, dtype))
t1 = time()
print("Time for filling the schunk with `data` argument in the constructor: {:.3f}s".format(t1 - t0))

schunk = create_schunk()
t0 = time()
schunk.fill_special(shape, blosc2.SpecialValue.UNINIT)
schunk[:] = np.full(shape, np.pi, dtype)
t1 = time()
print("Time for filling the schunk without passing directly the value: {:.3f}s".format(t1 - t0))

schunk = create_schunk()
t0 = time()
schunk.fill_special(shape, blosc2.SpecialValue.VALUE, np.pi)
t1 = time()
print("Time for filling the schunk passing directly the value to `fill_special`: {:.3f}s".format(t1 - t0))

blosc2.remove_urlpath(urlpath)
