# Imports
import numpy as np
import blosc2
import time


file = "dset-ones.b2nd"
# a = blosc2.open(file)
# expr = blosc2.where(a < 5, a * 2**14, a)
d = 200
shape = (d,) * 4
chunks = (d // 4,) * 4
blocks = (d // 10,) * 4
print(f"Creating a 4D array of shape {shape} with chunks {chunks} and blocks {blocks}...")
t = time.time()
#a = blosc2.linspace(0, d, num=d**4, shape=(d,) * 4, blocks=(d//10,) * 4, chunks=(d//2,) * 4, urlpath=file, mode="w")
#a = blosc2.linspace(0, d, num = d**4, shape=(d,)*4, blocks=(d//10,)*4, chunks=(d//2,)*4)
# a = blosc2.arange(0, d**4, shape=(d,) * 4, blocks=(d//10,) * 4, chunks=(d//2,) * 4, urlpath=file, mode="w")
a = blosc2.ones(shape=shape, chunks=chunks, blocks=blocks) #, urlpath=file, mode="w")
t = time.time() - t
print(f"Time to create array: {t:.6f} seconds")
t = time.time()
#expr = a * 30
expr = a * 2
print(f"Time to create expression: {time.time() - t:.6f} seconds")

# dim0
t = time.time()
res = expr[1]
t0 = time.time() - t
print(f"Time to access dim0: {t0:.6f} seconds")

# dim1
t = time.time()
res = expr[:,1]
t1 = time.time() - t
print(f"Time to access dim1: {t1:.6f} seconds")

# dim2
t = time.time()
res = expr[:,:,1]
t2 = time.time() - t
print(f"Time to access dim2: {t2:.6f} seconds")

# dim3
t = time.time()
res = expr[:,:,:,1]
#res = expr[1]
t3 = time.time() - t

print(f"Time to access dim3: {t3:.6f} seconds")
