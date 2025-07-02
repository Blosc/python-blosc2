#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for computing a slice of a expression in a 4D array.

import numpy as np
import blosc2
import time
from memory_profiler import memory_usage, profile
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
@profile
def slice_dim0():
    t = time.time()
    res = expr[1]
    t0 = time.time() - t
    print(f"Time to access dim0: {t0:.6f} seconds")
    print(f"dim0 slice size: {np.prod(res.shape) * res.dtype.itemsize / 2**30:.6f} GB")

# dim1
@profile
def slice_dim1():
    t = time.time()
    res = expr[:,1]
    t1 = time.time() - t
    print(f"Time to access dim1: {t1:.6f} seconds")
    print(f"dim1 slice size: {np.prod(res.shape) * res.dtype.itemsize / 2**30:.6f} GB")

# dim2
@profile
def slice_dim2():
    t = time.time()
    res = expr[:,:,1]
    t2 = time.time() - t
    print(f"Time to access dim2: {t2:.6f} seconds")
    print(f"dim2 slice size: {np.prod(res.shape) * res.dtype.itemsize / 2**30:.6f} GB")

# dim3
@profile
def slice_dim3():
    t = time.time()
    res = expr[:,:,:,1]
    t3 = time.time() - t
    print(f"Time to access dim3: {t3:.6f} seconds")
    print(f"dim3 slice size: {np.prod(res.shape) * res.dtype.itemsize / 2**30:.6f} GB")

if __name__ == '__main__':
    interval = 0.001
    offset = 0
    for f in [slice_dim0, slice_dim1, slice_dim2, slice_dim3]:
        mem = memory_usage((f,), interval=interval)
        times = offset + interval * np.arange(len(mem))
        offset = times[-1]
        plt.plot(times, mem)

    plt.xlabel('Time (s)')
    plt.ylabel('Memory usage (MiB)')
    plt.title('Memory usage lazyexpr slice (fast path), Linux Blosc2 3.5.1')
    plt.legend(['expr[1]', 'expr[:,1]', 'expr[:,:,1]', 'expr[:,:,:,1]'])
    plt.savefig('Linux_Blosc3_5_1_fast.png', format="png")
