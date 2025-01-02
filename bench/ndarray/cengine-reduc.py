from time import time
import blosc2
import numpy as np

# Create some data operands
N = 20_000   # working size of 6 GB
chunks = (100, N)
blocks = (1, N)
chunks, blocks= None, None
cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4)
t0 = time()
# a = blosc2.linspace(0, 1, N * N, dtype="float32", shape=(N, N), cparams=cparams)
a = blosc2.linspace(0, 1, N * N, shape=(N, N), cparams=cparams, chunks=chunks, blocks=blocks)
b = blosc2.linspace(1, 2, N * N, shape=(N, N), cparams=cparams, chunks=chunks, blocks=blocks)
c = blosc2.linspace(-10, 10, N, cparams=cparams)  # broadcasting is supported
print("Time to create data: ", time() - t0)
print("a.chunks, a.blocks, a.schunk.cratio: ", a.chunks, a.blocks, a.schunk.cratio)

# Expression
t0 = time()
expr = ((a ** 3 + blosc2.sin(a * 2)) < c) & (b > 0)
print(f"Time to create expression: {time() - t0:.5f}")

# Evaluate while reducing (yep, reductions are in) along axis 1
t0 = time()
out = blosc2.sum(expr, axis=1) # , cparams=cparams)
t1 = time() - t0
print(f"Time to compute with Blosc2: {t1:.5f}")

# Evaluate using NumPy operands
na, nb, nc = a[:], b[:], c[:]

@blosc2.cengine
def compute_expression(na, nb, nc):
    return np.sum(((na ** 3 + np.sin(na * 2)) < nc) & (nb > 0), axis=1)

t0 = time()
out2 = compute_expression(na, nb, nc)
t1 = time() - t0
print(f"Time to compute with NumPy operands and Blosc2 engine: {t1:.5f}")

def compute_expression_numpy(na, nb, nc):
    return np.sum(((na ** 3 + np.sin(na * 2)) < nc) & (nb > 0), axis=1)

# Evaluate using NumPy compute engine
t0 = time()
nout = compute_expression_numpy(na, nb, nc)
t2 = time() - t0
print(f"Time to compute with NumPy: {t2:.5f}")
print(f"Speedup: {t2 / t1:.2f}x")

assert np.all(out == nout)
assert np.all(out2 == nout)
print("All results are equal!")
