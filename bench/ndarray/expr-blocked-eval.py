from time import time
import blosc2
import numpy as np

N = 10_000
dtype= np.float32

t0 = time()
a = blosc2.ones((N, N), dtype=dtype)
print(f"Time to create data: {(time() - t0) * 1000 :.4f} ms")
t0 = time()
b = a.copy()
c = a.copy()
print(f"Time to copy data: {(time() - t0) * 1000 :.4f} ms")

t0 = time()
res = ((a + b) * c).compute()
print(f"Time to evaluate: {(time() - t0) * 1000 :.4f} ms")
# print(res.info)

np.testing.assert_allclose(res, a[:] * 2)
