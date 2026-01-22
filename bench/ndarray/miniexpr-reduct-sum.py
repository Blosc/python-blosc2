from time import time
import blosc2
import numpy as np
import numexpr as ne

N = 10_000
# dtype= np.int32
dtype= np.float32
# dtype= np.float64
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

t0 = time()
# a = blosc2.ones((N, N), dtype=dtype, cparams=cparams)
# a = blosc2.arange(np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
a = blosc2.linspace(0., 1., np.prod((N, N)), shape=(N, N), dtype=dtype, cparams=cparams)
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = blosc2.sum(a)
t = time() - t0
print(f"Time to evaluate: {t * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(a.nbytes / 1e9) / t:.2f}")
print("Result:", res, "Mean:", res / (N * N))

na = a[:]

t0 = time()
nres = np.sum(na)
nt = time() - t0
print(f"Time to evaluate with NumPy: {nt * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes / 1e9) / nt:.2f}")
print("Result:", nres, "Mean:", nres / (N * N))
print(f"Speedup Blosc2 vs NumPy: {nt / t:.2f}x")
assert np.allclose(res, nres)

t0 = time()
neres = ne.evaluate("sum(na)")
net = time() - t0
print(f"Time to evaluate with NumExpr: {net * 1000 :.4f} ms", end=" ")
print(f"Speed (GB/s): {(na.nbytes / 1e9) / net:.2f}")
print("Result:", neres, "Mean:", neres / (N * N))
print(f"Speedup Blosc2 vs NumExpr: {net / t:.2f}x")
