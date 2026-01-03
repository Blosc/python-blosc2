from time import time
import blosc2
import numpy as np
import numexpr as ne

N = 10_000
dtype= np.float32
#dtype= np.float64
#dtype= np.int32
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)
cparams_out = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=0,
                             blocksize=cparams.blocksize, splitmode=blosc2.SplitMode.NEVER_SPLIT)

t0 = time()
#a = blosc2.ones((N, N), dtype=dtype, cparams=cparams)
#a = blosc2.arange(np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
a = blosc2.linspace(0., 1., np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")
t0 = time()
b = a.copy()
c = a.copy()
print(f"Time to copy data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = blosc2.sum(a, cparams=cparams)
t = time() - t0
print(f"Time to evaluate: {t * 1000 :.4f} ms")
print(f"Speed (GB/s): {(a.nbytes / 1e9) / t:.2f}")
print("res:", res)

na = a[:]
nb = b[:]
nc = c[:]
# np.testing.assert_allclose(res, np.sum(na), rtol=1e-5)

t0 = time()
res = np.sum(na)
t = time() - t0
print(f"Time to evaluate with NumPy: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes / 1e9) / t:.2f}")

t0 = time()
res = ne.evaluate("sum(na)")
t = time() - t0
print(f"Time to evaluate with NumExpr: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes / 1e9) / t:.2f}")
