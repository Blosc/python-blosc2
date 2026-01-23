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

shape = (40_000, 5_000)
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
npd = np.where(npc, 0, 1)
tref = t = time() - t0
print(f"Time to compute where expression (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

t0 = time()
npc = ne.evaluate('where(a**2 + b**2 > 2 * a * b + 1, 0, 1)', local_dict={'a': npa, 'b': npb})
t = time() - t0
print(f"Time to compute where expression (NumExpr): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

s = blosc2.asarray(nps, chunks=chunks, blocks=blocks, cparams=cparams)
print(f"*** Working with NDArray with shape: {s.shape}, chunks: {s.chunks}, blocks: {s.blocks},"
      f" cratio: {s.schunk.cratio:.2f}x")
a = s['a']
b = s['b']

# Get a LazyExpr instance
# Compute: output is a NDArray
t0 = time()
c = a**2 + b**2 > 2 * a * b + 1
d = c.where(0, 1).compute(cparams=cparams)
t = time() - t0
print(f"Time to compute where expression (compute): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

# Compute the whole slice: output is a NumPy array
t0 = time()
c = a**2 + b**2 > 2 * a * b + 1
npd = c.where(0, 1)[:]
t = time() - t0
print(f"Time to compute where expression (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

print("*** Extracting rows")
# Compute and get row values: NumPy
t0 = time()
npc = npa**2 + npb**2 > 2 * npa * npb + 1
npd = nps[npc]
tref = t = time() - t0
print(f"Time to get row values (NumPy): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s")

# Compute and get row values: output is a NDArray
t0 = time()
npd = s[a**2 + b**2 > 2 * a * b + 1].compute(cparams=cparams)
t = time() - t0
print(f"Time to get row values (compute): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

# Compute and get row values: output is a NDArray
t0 = time()
npd = s['a**2 + b**2 > 2 * a * b + 1'].compute(cparams=cparams)
t = time() - t0
print(f"Time to get row values (compute, string): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

# Compute and get row values: output is a NumPy array
t0 = time()
npd = s[a**2 + b**2 > 2 * a * b + 1][:]
t = time() - t0
print(f"Time to get row values (getitem): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")

# Compute and get row values: output is a NumPy array
t0 = time()
npd = s['a**2 + b**2 > 2 * a * b + 1'][:]
t = time() - t0
print(f"Time to get row values (getitem, string): {t:.3f} s; {nps.nbytes/2**30/t:.3f} GB/s; {tref / t:.1f}x wrt NumPy")
