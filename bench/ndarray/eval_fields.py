#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

from time import time

import blosc2
import numexpr as ne
import numpy as np

shape = (4_000, 5_000)
chunks = (10, 5_000)
blocks = (1, 1000)
# Comment out the next line to force chunks and blocks above
chunks, blocks = None, None
# Check with fast compression
cparams = dict(clevel=1, codec=blosc2.Codec.BLOSCLZ)

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
print(f"Time to evaluate field expression (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

t0 = time()
npc = ne.evaluate('a**2 + b**2 > 2 * a * b + 1', local_dict={'a': npa, 'b': npb})
t = time() - t0
print(f"Time to evaluate field expression (NumExpr): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

s = blosc2.asarray(nps, chunks=chunks, blocks=blocks, cparams=cparams)
print(f"shape: {s.shape}, chunks: {s.chunks}, blocks: {s.blocks}, cratio: {s.schunk.cratio:.2f}")
a = s.fields['a']
# a = s['a']  # TODO: implement this (should be an expression)
b = s.fields['b']

# Get a LazyExpr instance
c = a**2 + b**2 > 2 * a * b + 1
# Evaluate: output is a NDArray
t0 = time()
d = c.compute(cparams=cparams)
t = time() - t0
print(f"Time to evaluate field expression (eval): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

# Evaluate the whole slice: output is a NumPy array
t0 = time()
npd = c[:]
t = time() - t0
print(f"Time to evaluate field expression (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.2f} GB/s")

# Evaluate a partial slice: output is a NumPy array
t0 = time()
npd = c[1:10]
t = time() - t0
print(f"Time to evaluate field expression (partial getitem): {t:.3f} s; {npd.nbytes/2**20/t:.2f} MB/s")
