from time import time
import blosc2
import numpy as np
import numexpr as ne

N = 10_000
dtype= np.int32
#dtype= np.float32
#dtype= np.float64
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

t0 = time()
a = blosc2.ones((N, N), dtype=dtype, cparams=cparams)
#a = blosc2.arange(np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
# a = blosc2.linspace(0., 1., np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")
t0 = time()
b = a.copy()
c = a.copy()
print(f"Time to copy data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = ((a + b) * c).compute(cparams=cparams)
t = time() - t0
print(f"Time to evaluate: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(a.nbytes * 4 / 1e9) / t:.2f}")
# print(res.info)

na = a[:]
nb = b[:]
nc = c[:]
#np.testing.assert_allclose(res, (na + nb) * nc)

t0 = time()
res = (na + nb) * nc
t = time() - t0
print(f"Time to evaluate with NumPy: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes * 4 / 1e9) / t:.2f}")

t0 = time()
res = ne.evaluate("(na + nb) * nc")
t = time() - t0
print(f"Time to evaluate with NumExpr: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes * 4 / 1e9) / t:.2f}")
