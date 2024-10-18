#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2
from time import time

import numexpr as ne

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
npd = np.where(npc, 0, 1)
t = time() - t0
print(f"Time to evaluate where expression (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

t0 = time()
npc = ne.evaluate('where(a**2 + b**2 > 2 * a * b + 1, 0, 1)', local_dict={'a': npa, 'b': npb})
t = time() - t0
print(f"Time to evaluate where expression (NumExpr): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

s = blosc2.asarray(nps, chunks=chunks, blocks=blocks, cparams=cparams)
print(f"shape: {s.shape}, chunks: {s.chunks}, blocks: {s.blocks}")
a = s.fields['a']
b = s.fields['b']

# Get a LazyExpr instance
# Evaluate: output is a NDArray
t0 = time()
c = a**2 + b**2 > 2 * a * b + 1
d = c.where(0, 1).compute(cparams=cparams)
t = time() - t0
print(f"Time to evaluate where expression (eval): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate the whole slice: output is a NumPy array
t0 = time()
c = a**2 + b**2 > 2 * a * b + 1
npd = c.where(0, 1)[:]
t = time() - t0
print(f"Time to evaluate where expression (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate and get row values: NumPy
t0 = time()
npc = npa**2 + npb**2 > 2 * npa * npb + 1
npd = nps[npc]
t = time() - t0
print(f"Time to get row values (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate and get row values: output is a NDArray
t0 = time()
npd = s[a**2 + b**2 > 2 * a * b + 1].compute(cparams=cparams)
t = time() - t0
print(f"Time to get row values (eval): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate and get row values: output is a NDArray
t0 = time()
npd = s['a**2 + b**2 > 2 * a * b + 1'].compute(cparams=cparams)
t = time() - t0
print(f"Time to get row values (eval, string): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate and get row values: output is a NumPy array
t0 = time()
npd = s[a**2 + b**2 > 2 * a * b + 1][:]
t = time() - t0
print(f"Time to get row values (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Evaluate and get row values: output is a NumPy array
t0 = time()
npd = s['a**2 + b**2 > 2 * a * b + 1'][:]
t = time() - t0
print(f"Time to get row values (getitem, string): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")
