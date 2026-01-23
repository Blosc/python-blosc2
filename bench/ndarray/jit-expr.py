#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compute expressions for different array sizes, using the jit decorator.

from time import time
import blosc2
import numpy as np
import numexpr as ne

niter = 5
# Create some data operands
N = 10_000   # working size of ~1 GB
dtype = "float32"
chunks = (100, N)
blocks = (1, N)
chunks, blocks= None, None   # enforce automatic chunk and block sizes
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4)
cparams_out = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4)
print("Using cparams: ", cparams)
check_result = False
# Lossy compression
# filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
# filters_meta = [8, 0]  # keep 8 bits of precision in mantissa
# cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=filters, filters_meta=filters_meta)
# check_result = False


t0 = time()
na = np.linspace(0, 1, N * N, dtype=dtype).reshape(N, N)
nb = np.linspace(1, 2, N * N, dtype=dtype).reshape(N, N)
nc = np.linspace(-10, 10, N, dtype=dtype)  # broadcasting is supported
# nc = np.linspace(-10, 10, N * N, dtype=dtype).reshape(N, N)
print("Time to create data: ", time() - t0)

def compute_expression_numpy(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

t0 = time()
nout = compute_expression_numpy(na, nb, nc)
tref = time() - t0
print(f"Time to compute with NumPy engine: {tref:.5f}")

nout = ne.evaluate("((na ** 3 + sin(na * 2)) < nc) & (nb > 0)")
t0 = time()
for i in range(niter):
    nout = ne.evaluate("((na ** 3 + sin(na * 2)) < nc) & (nb > 0)")
t1 = (time() - t0) / niter
print(f"Time to compute with NumExpr: {t1:.5f}")
print(f"Speedup: {tref / t1:.2f}x")

@blosc2.jit
def compute_expression_nocompr(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

print("\nUsing NumPy operands...")

@blosc2.jit(cparams=cparams_out)
def compute_expression_compr(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

out = compute_expression_compr(na, nb, nc)
t0 = time()
for i in range(niter):
    out = compute_expression_compr(na, nb, nc)
t1 = (time() - t0) / niter
print(f"Time to compute with NumPy operands and NDArray as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)

out = compute_expression_nocompr(na, nb, nc)
t0 = time()
for i in range(niter):
    out = compute_expression_nocompr(na, nb, nc)
t1 = (time() - t0) / niter
print(f"Time to compute with NumPy operands and NumPy as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)

print("\nUsing NDArray operands *with* compression...")
# Create Blosc2 operands
a = blosc2.asarray(na, cparams=cparams, chunks=chunks, blocks=blocks)
b = blosc2.asarray(nb, cparams=cparams, chunks=chunks, blocks=blocks)
c = blosc2.asarray(nc, cparams=cparams)
# c = blosc2.asarray(nc, cparams=cparams, chunks=chunks, blocks=blocks)
print(f"{a.chunks=}, {a.blocks=}, {a.schunk.cratio=:.2f}x")

out = compute_expression_compr(a, b, c)
t0 = time()
for i in range(niter):
    out = compute_expression_compr(a, b, c)
t1 = (time() - t0) / niter
print(f"[COMPR] Time to compute with NDArray operands and NDArray as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)

out = compute_expression_nocompr(a, b, c)
t0 = time()
for i in range(niter):
    out = compute_expression_nocompr(a, b, c)
t1 = (time() - t0) / niter
print(f"[COMPR] Time to compute with NDArray operands and NumPy as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)

print("\nUsing NDArray operands without compression...")
# Create NDArray operands without compression
cparams = cparams_out = blosc2.CParams(clevel=0)
a = blosc2.asarray(na, cparams=cparams, chunks=chunks, blocks=blocks)
b = blosc2.asarray(nb, cparams=cparams, chunks=chunks, blocks=blocks)
c = blosc2.asarray(nc, cparams=cparams)
# c = blosc2.asarray(nc, cparams=cparams, chunks=chunks, blocks=blocks)
print(f"{a.chunks=}, {a.blocks=}, {a.schunk.cratio=:.2f}x")

out = compute_expression_compr(a, b, c)
t0 = time()
for i in range(niter):
    out = compute_expression_compr(a, b, c)
t1 = (time() - t0) / niter
print(f"[NOCOMPR] Time to compute with NDArray operands and NDArray as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)

out = compute_expression_nocompr(a, b, c)
t0 = time()
for i in range(niter):
    out = compute_expression_nocompr(a, b, c)
t1 = (time() - t0) / niter
print(f"[NOCOMPR] Time to compute with NDArray operands and NumPy as result: {t1:.5f}")
cratio = out.schunk.cratio if isinstance(out, blosc2.NDArray) else 1.0
print(f"Speedup: {tref / t1:.2f}x, out cratio: {cratio:.2f}x")
if check_result:
    np.testing.assert_allclose(out, nout)
    print("All results are equal!")
