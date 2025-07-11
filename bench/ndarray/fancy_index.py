#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Benchmark for computing a fancy index of a blosc2 array

import numpy as np
import ndindex
import blosc2
import time
from memory_profiler import memory_usage, profile
import matplotlib.pyplot as plt

MEM_PROFILE = False

# @profile
def fancyBlosc2(arr, fancyIdx):
    t = time.time()
    res = arr[fancyIdx]
    dt = time.time() - t
    return res, dt

# @profile
def fancyNumpy(arr, fancyIdx):
    t = time.time()
    res = arr[:]
    res = res[fancyIdx]
    dt = time.time() - t
    return res, dt

def genarray(d, verbose=False):
    shape = (d,) * 4
    chunks = (d // 4,) * 4
    blocks = (max(d // 10, 1),) * 4
    t = time.time()
    arr = blosc2.ones(shape=shape, chunks=chunks, blocks=blocks)  # , urlpath=file, mode="w")
    t = time.time() - t
    if verbose:
        print(f"Array shape: {arr.shape}")
        print(f"Array size: {np.prod(arr.shape) * arr.dtype.itemsize / 2 ** 30:.6f} GB")
        print(f"Time to create array: {t:.6f} seconds")
    return arr

if __name__ == '__main__':
    blosc_times = []
    np_times = []
    sizes = []
    # dims = np.int64(np.array([2**6.76, 2**6.8, 2**6.85, 2**6.9, 2**6.95]))
    dims = np.int64(np.array([2**4.3, 2**5.25, 2**6.25, 2**6.75, 2**7, 2**7.25, 2**7.4, 2**7.5]))

    for d in dims:
        arr = genarray(d)
        arr_size = np.prod(arr.shape) * arr.dtype.itemsize / 2 ** 30
        print(arr_size)
        sizes.append(arr_size)
        idx = np.random.randint(low=0, high=d, size=(d,))
        if MEM_PROFILE:
            fancyIdx = np.s_[idx, :, :d // 2, d // 2:]

            interval = 0.001
            offset = 0
            for f in [fancyBlosc2, fancyNumpy]:
                mem = memory_usage((f, (arr, fancyIdx)), interval=interval)
                times = offset + interval * np.arange(len(mem))
                offset = times[-1]
                plt.plot(times, mem)

            plt.xlabel('Time (s)')
            plt.ylabel('Memory usage (MiB)')
            plt.title('Memory usage fancy indexing')
            plt.legend(['blosc2', 'numpy'])
            plt.savefig(f'plots/MemoryUsagefancyIdx_d{d}.png', format="png")

        row = idx
        col = np.random.permutation(idx)
        mask = np.random.randint(low=0, high=2, size=(d,)) == 1

        ## Test fancy indexing for different use cases
        loc_blosc_times = []
        loc_np_times = []
        m, M = np.min(idx), np.max(idx)
        for i, fancyIdx in enumerate([
            [m, M//2, M],  # i)
            [[[m//2, M//2], [m//4, M//4]]],  # ii)
            [row, col],  # iii)
            # [row[:, None], col],  # iv)
            [m, col],  # v)
            [slice(1, M//2, 5), col],  # vi)
            # [row[:, None], mask],  # vii)
        ]):
            # print(f'\n(case {i + 1})')
            try:
                r, c = fancyIdx
                idx = (r, c)
            except:
                r, c = fancyIdx, None
                idx = r
            b, blosctime = fancyBlosc2(arr,idx)
            n, nptime = fancyNumpy(arr,idx)
            slice_size = np.prod(b.shape) * b.dtype.itemsize / 2 ** 30
            np.testing.assert_allclose(b, n)
            loc_blosc_times.append(blosctime)
            loc_np_times.append(nptime)
        blosc_times.append(loc_blosc_times)
        np_times.append(loc_np_times)

    blosc_times = np.array(blosc_times)
    np_times = np.array(np_times)
    x = np.arange(len(sizes))
    width = 0.25

    # Create bars for axis 0 plot
    plt.bar(x - width, np_times.mean(axis=1), width, color='b', alpha=0.5)
    plt.bar(x - width, np_times.max(axis=1), width, color='b', alpha=0.25)
    plt.bar(x - width, np_times.min(axis=1), width, label='NumPy', color='b', alpha=1)
    plt.bar(x, blosc_times.mean(axis=1), width, color='r', alpha=0.5)
    plt.bar(x, blosc_times.max(axis=1), width, color='r', alpha=0.25)
    plt.bar(x, blosc_times.min(axis=1), width, label='Blosc2', color='r', alpha=1)
    plt.bar([],[], label='max', color='k', alpha=0.25)
    plt.bar([],[], label='min', color='k', alpha=1)
    plt.bar([],[], label='mean', color='k', alpha=0.5)

    plt.xlabel('Array size (GB)')
    plt.legend()
    plt.xticks(x, np.round(sizes, 2))
    plt.ylabel("Time (s)")
    plt.title('Fancy indexing, Blosc2 vs NumPy')
    plt.savefig('plots/fancyIdx.png', format="png")
    plt.show()
    ## slowest cases are the ones with broadcasting
    ## in fact it's a problem for blosc2
