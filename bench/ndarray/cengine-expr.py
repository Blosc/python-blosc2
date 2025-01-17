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
chunks, blocks= None, None
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4)
# Lossy compression
# filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
# filters_meta = [8, 0]  # keep 8 bits of precision in mantissa
# cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=filters, filters_meta=filters_meta)

t0 = time()
na = np.linspace(0, 1, N * N, dtype=dtype).reshape(N, N)
nb = np.linspace(1, 2, N * N, dtype=dtype).reshape(N, N)
nc = np.linspace(-10, 10, N, dtype=dtype)  # broadcasting is supported
# nc = np.linspace(-10, 10, N * N, dtype=dtype).reshape(N, N)
print("Time to create data: ", time() - t0)

def compute_expression_numpy(na, nb, nc):
    return ((na ** 3 + np.sin(na * 2)) < nc) & (nb > 0)

t0 = time()
nout = compute_expression_numpy(na, nb, nc)
t2 = time() - t0
print(f"Time to compute with NumPy engine: {t2:.5f}")

nout = ne.evaluate("((na ** 3 + sin(na * 2)) < nc) & (nb > 0)")
t0 = time()
for i in range(niter):
    nout = ne.evaluate("((na ** 3 + sin(na * 2)) < nc) & (nb > 0)")
t1 = (time() - t0) / niter
print(f"Time to compute with NumExpr: {t1:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")

@blosc2.cengine
def compute_expression_nocompr(na, nb, nc):
    return ((na ** 3 + np.sin(na * 2)) < nc) & (nb > 0)

out = compute_expression_nocompr(na, nb, nc)
t0 = time()
for i in range(niter):
    out = compute_expression_nocompr(na, nb, nc)
t1 = (time() - t0) / niter
print(f"Time to compute with NumPy operands and Blosc2 engine: {t1:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")
np.testing.assert_allclose(out, nout)

@blosc2.cengine(cparams=cparams)
def compute_expression_compr(a, b, c):
    return ((a ** 3 + np.sin(a * 2)) < c) & (b > 0)

# Create Blosc2 operands
a = blosc2.asarray(na, cparams=cparams, chunks=chunks, blocks=blocks)
b = blosc2.asarray(nb, cparams=cparams, chunks=chunks, blocks=blocks)
c = blosc2.asarray(nc, cparams=cparams)
# c = blosc2.asarray(nc, cparams=cparams, chunks=chunks, blocks=blocks)
print("a.chunks, a.blocks, a.schunk.cratio: ", a.chunks, a.blocks, a.schunk.cratio)

out2 = compute_expression_compr(a, b, c)
t0 = time()
for i in range(niter):
    out2 = compute_expression_compr(a, b, c)
t1 = (time() - t0) / niter
print(f"Time to compute with NDArray operands and NDArray as result: {t1:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")
np.testing.assert_allclose(out2, nout)

out3 = compute_expression_compr(na, nb, nc)
t0 = time()
for i in range(niter):
    out3 = compute_expression_compr(na, nb, nc)
t1 = (time() - t0) / niter
print(f"Time to compute with NumPy operands and NDArray as result: {t1:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")
np.testing.assert_allclose(out3, nout)

out4 = compute_expression_nocompr(a, b, c)
t0 = time()
for i in range(niter):
    out4 = compute_expression_nocompr(a, b, c)
t1 = (time() - t0) / niter
print(f"Time to compute with NDArray operands and NumPy as result: {t1:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")
np.testing.assert_allclose(out4, nout)

print("All results are equal!")
