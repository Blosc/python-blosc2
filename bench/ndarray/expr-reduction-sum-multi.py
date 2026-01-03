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
res = blosc2.sum(a + b + c, cparams=cparams)
print(f"Time to evaluate: {(time() - t0) * 1000 :.4f} ms")
print("Result:", res, "Mean:", res / (N * N))

na = a[:]
nb = b[:]
nc = c[:]
#np.testing.assert_allclose(res, np.sum(na + nb + nc))
#
#t0 = time()
#res = ne.evaluate("sum(na)")
#print(f"Time to evaluate with NumExpr: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = np.sum(na + nb + nc)
print(f"Time to evaluate with NumPy: {(time() - t0) * 1000 :.4f} ms")
