# Imports
import numpy as np
import blosc2
import time
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

file = "dset-ones.b2nd"
# a = blosc2.open(file)
# expr = blosc2.where(a < 5, a * 2**14, a)
d = 160
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
def slice_dim0():
    t = time.time()
    res = expr[1]
    t0 = time.time() - t
    print(f"Time to access dim0: {t0:.6f} seconds")

# dim1
def slice_dim1():
    t = time.time()
    res = expr[:,1]
    t1 = time.time() - t
    print(f"Time to access dim1: {t1:.6f} seconds")

# dim2
def slice_dim2():
    t = time.time()
    res = expr[:,:,1]
    t2 = time.time() - t
    print(f"Time to access dim2: {t2:.6f} seconds")

# dim3
def slice_dim3():
    t = time.time()
    res = expr[:,:,:,1]
    #res = expr[1]
    t3 = time.time() - t

    print(f"Time to access dim3: {t3:.6f} seconds")

fig = plt.figure()
interval = 0.001
offset = 0
for f in [slice_dim0, slice_dim1, slice_dim2, slice_dim3]:
    mem = memory_usage((f,), interval=interval)
    times = offset + interval * np.arange(len(mem))
    offset = times[-1]
    plt.plot(times, mem)

plt.xlabel('Time (s)')
plt.ylabel('Memory usage (MiB)')
plt.title('Memory usage over time for slicing operations, slice-expr.py')
plt.legend(['dim0', 'dim1', 'dim2', 'dim3'])
plt.savefig('plots/slice-expr.png', format="png")
