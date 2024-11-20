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
nelem = 1_00_000_000
dtype = np.dtype(np.float64)
print(f"Filling a SChunk with {nelem / 1e6} Melements of {dtype=}")

persistent = bool(sys.argv[1]) if len(sys.argv) > 1 else False
if persistent:
    urlpath = "bench_fill_special.b2frame"
    print(f"Writing output to {urlpath}...")
else:
    urlpath = None


def create_schunk(data=None):
    blosc2.remove_urlpath(urlpath)
    # Create the empty SChunk
    return blosc2.SChunk(data=data, urlpath=urlpath, cparams={"typesize": dtype.itemsize})


t0 = time()
schunk = create_schunk(data=np.full(nelem, np.pi, dtype))
t = (time() - t0) * 1000.
print(f"Time with `data` argument in constructor: {t:19.3f} ms")

schunk = create_schunk()
t0 = time()
schunk.fill_special(nelem, blosc2.SpecialValue.UNINIT)
schunk[:] = np.full(nelem, np.pi, dtype)
t = (time() - t0) * 1000.
print(f"Time without passing directly the value: {t:20.3f} ms")

schunk = create_schunk()
t0 = time()
schunk.fill_special(nelem, blosc2.SpecialValue.VALUE, np.pi)
t = (time() - t0) * 1000.
print(f"Time passing directly the value to `fill_special`: {t:10.3f} ms")

blosc2.remove_urlpath(urlpath)
