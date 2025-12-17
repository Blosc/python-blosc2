from time import time
import blosc2
import numpy as np
import numexpr as ne

N = 10_000
dtype= np.float32
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

t0 = time()
#a = blosc2.ones((N, N), dtype=dtype)
a = blosc2.arange(np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
# a = blosc2.linspace(0., 1., np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")
t0 = time()
b = a.copy()
c = a.copy()
print(f"Time to copy data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = ((a + b) * c).compute(cparams=cparams)
print(f"Time to evaluate: {(time() - t0) * 1000 :.4f} ms")
# print(res.info)

na = a[:]
nb = b[:]
nc = c[:]
np.testing.assert_allclose(res, (na + nb) * nc)

t0 = time()
res = ne.evaluate("(na + nb) * nc")
print(f"Time to evaluate with NumExpr: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = na + nb * nc
print(f"Time to evaluate with NumPy: {(time() - t0) * 1000 :.4f} ms")
