from time import time
import blosc2
import numpy as np
import numexpr as ne

N = 10_000
dtype= np.float32
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

t0 = time()
#a = blosc2.ones((N, N), dtype=dtype, cparams=cparams)
#a = blosc2.arange(np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
a = blosc2.linspace(0., 1., np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
#rng = np.random.default_rng(1234)
#a = rng.integers(0, 2, size=(N, N), dtype=dtype)
#a = blosc2.asarray(a, cparams=cparams, urlpath="a.b2nd", mode="w")
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")
#print(a[:])
t0 = time()
b = a.copy()
c = a.copy()
print(f"Time to copy data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = blosc2.sum(2 * a**2 - 3 * b + c + 1.2, cparams=cparams)
t = time() - t0
print(f"Time to evaluate: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(a.nbytes * 3 / 1e9) / t:.2f}")
print("Result:", res, "Mean:", res / (N * N))

na = a[:]
nb = b[:]
nc = c[:]
#np.testing.assert_allclose(res, np.sum(2 * a**2 - 3 * b + c + 1.2))

t0 = time()
res = np.sum(2 * na**2 - 3 * nb + nc + 1.2)
t = time() - t0
print(f"Time to evaluate with NumPy: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes * 3 / 1e9) / t:.2f}")

t0 = time()
res = ne.evaluate("sum(2 * na**2 - 3 * nb + nc + 1.2)")
t = time() - t0
print(f"Time to evaluate with NumExpr: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes / 1e9) / t:.2f}")
