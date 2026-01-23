#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from time import time

import numexpr as ne
import numpy as np

import blosc2

shape = (4_000, 5_000)
chunks = (10, 5_000)
blocks = (1, 1000)
# Comment out the next line to force chunks and blocks above
chunks, blocks = None, None
# Check with fast compression
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.BLOSCLZ)

print(f"*** Working with an struct array with shape: {shape}")
# Create a structured NumPy array
npa_ = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(shape)
npb_ = np.linspace(1, 2, np.prod(shape), dtype=np.float64).reshape(shape)
nps = np.empty(shape, dtype=[('a', npa_.dtype), ('b', npb_.dtype)])
nps['a'] = npa_
nps['b'] = npb_
npa = nps['a']
npb = nps['b']
t0 = time()
npc = npa**2 + npb**2 > 2 * npa * npb + 1
t = time() - t0
print(f"Time to compute field expression (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

t0 = time()
npc = ne.evaluate('a**2 + b**2 > 2 * a * b + 1', local_dict={'a': npa, 'b': npb})
t = time() - t0
print(f"Time to compute field expression (NumExpr): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

s = blosc2.asarray(nps, chunks=chunks, blocks=blocks, cparams=cparams)
print(f"*** Working with NDArray with shape: {s.shape}, chunks: {s.chunks}, blocks: {s.blocks},"
      f" cratio: {s.schunk.cratio:.2f}x")
a = s['a']
b = s['b']

# Get a LazyExpr instance
c = a**2 + b**2 > 2 * a * b + 1
# Compute: output is a NDArray
t0 = time()
d = c.compute(cparams=cparams)
t = time() - t0
print(f"Time to compute field expression (compute): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

# Compute the whole slice: output is a NumPy array
t0 = time()
npd = c[:]
t = time() - t0
print(f"Time to compute field expression (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

# Compute a partial slice: output is a NumPy array
t0 = time()
npd = c[1:10]
t = time() - t0
print(f"Time to compute field expression (partial getitem): {t:.3f} s; {npd.nbytes/2**20/t:.2f} MB/s")
