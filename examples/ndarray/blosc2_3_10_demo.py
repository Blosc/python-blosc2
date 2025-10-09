import time

import blosc2

N, M = 5_000, 10_000
dtype = blosc2.float64
working_set = dtype().itemsize * (2 * N * M + N * N) / 2**30
print(f"Working set size of {round(working_set, 2)} GB")
shape1 = (N, M)
shape2 = (M, N)
a = blosc2.ones(shape=shape1, urlpath="a.b2nd", mode="w", dtype=dtype)
b = blosc2.full(fill_value=2.0, shape=shape2, urlpath="b.b2nd", mode="w", dtype=dtype)

# Expression
t0 = time.time()
# Define the operands and expression
expression, operands = "matmul(a, b) + sin(b[2])", {"a": a, "b": b}
# Create a lazy expression
lexpr = blosc2.lazyexpr(expression, operands)
print(f"Result of {expression} will have shape {lexpr.shape} and dtype {lexpr.dtype}")
# Save the lazy expression to the specified path
url_path = "my_expr.b2nd"
lexpr.save(urlpath=url_path, mode="w")
dt = time.time() - t0
print(f"Defined expression, got metadata, and persisted it on disk in {round(dt * 1000, 3)} ms!")

# Reopen persistent expression, compute, and write to disk with blosc2
t0 = time.time()
lexpr = blosc2.open(urlpath=url_path)
dt = time.time() - t0
print(f"In {round(dt * 1000, 3)} ms opened lazy expression: shape = {lexpr.shape}, dtype = {lexpr.dtype}")
t1 = time.time()
result1 = lexpr.compute(urlpath="result.b2nd", mode="w")
t2 = time.time()
print(f"blosc2 fetched operands from disk, computed {expression}, wrote to disk in: {t2 - t1:.3f} s")
