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
import zarr
import h5py
plt.rcParams.update({'text.usetex':True,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rc('text', usetex=True)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')

NUMPY_BLOSC = False

def genarray(r, ndims=1, verbose=True):
    d = int((r*2**30/8)**(1/ndims))
    shape = (d,) * ndims
    chunks = (d // 4,) * ndims
    blocks = (max(d // 10, 1),) * ndims
    t = time.time()
    arr = blosc2.ones(shape=shape, chunks=chunks, blocks=blocks, dtype=np.int64)  # , urlpath=file, mode="w")
    t = time.time() - t
    if verbose:
        print(f"Array shape: {arr.shape}")
        print(f"Array size: {np.prod(arr.shape) * arr.dtype.itemsize / 2 ** 30:.6f} GB")
        print(f"Time to create array: {t:.6f} seconds")
    return arr

blosc_times = []
np_times = []
zarr_times = []
sizes = []
dims = np.int64(np.array([1, 2, 4, 6, 8]))
rng = np.random.default_rng()
blosctimes = []
nptimes = []
zarrtimes = []
h5pytimes = []

for d in dims:
    arr = genarray(d, ndims=2)
    sizes.append(d)
    idx = rng.integers(low=0, high=arr.shape[0], size=(arr.shape[0],))
    row = np.arange(arr.shape[0])
    col = row
    mask = rng.integers(low=0, high=2, size=(d,)) == 1


    ## Test fancy indexing for different use cases
    m, M = np.min(idx), np.max(idx)
    def timer(arr, skip_flag=True, row=row, col=col):
        time_list = []
        if not skip_flag:
            t = time.time()
            b = arr[row, col]
            time_list += [time.time() - t]
            t = time.time()
            b = arr[slice(1, M // 2, 5), col]
            time_list += [time.time() - t]
            t = time.time()
            b = arr[[[m // 2, M // 2], [m // 4, M // 4]]]
            time_list += [time.time() - t]
        t = time.time()
        b = arr[[m, M//2, M]]
        time_list += [time.time() - t]
        t = time.time()
        b = arr[m, col]
        time_list += [time.time() - t]
        return np.array(time_list)

    if NUMPY_BLOSC:
        blosctimes += [timer(arr, skip_flag=False, row=idx, col=idx)]
        arr=arr[:]
        nptimes += [timer(arr, skip_flag=False, row=idx, col=idx)]
    else:
        blosctimes += [timer(arr)]
        arr = arr[:]
        nptimes += [timer(arr)]
        z_test = zarr.zeros(shape=arr.shape, dtype=arr.dtype)
        z_test[:] = arr
        # zarr is more limited, as must provide same number of coord arrays as dims of array
        # also cannot mix with slices
        zarrtimes += [timer(z_test)]
        with h5py.File('my_hdf5_file.h5', 'w') as f:
            dset = f.create_dataset("init", data=arr)
            h5pytimes += [timer(dset)]

x = np.arange(len(sizes))
width = 0.2
blosctimes = np.array(blosctimes)
nptimes = np.array(nptimes)
if NUMPY_BLOSC:
    # Create bars for axis 0 plot
    for i, r in enumerate((["Numpy", nptimes, -width], ["Blosc2", blosctimes, 0])):
        label, times, w = r
        c = ['b', 'r'][i]
        plt.bar(x + w, times.mean(axis=1), width, color=c, alpha=0.5)
        plt.bar(x + w, times.max(axis=1), width, color=c, alpha=0.25)
        plt.bar(x + w, times.min(axis=1), width, label=label, color=c, alpha=1)

    plt.xlabel('Array size (GB)')
    plt.legend()
    plt.xticks(x, np.round(sizes, 2))
    plt.ylabel("Time (s)")
    plt.title('Fancy indexing NumPy vs Blosc2')
    plt.savefig('plots/fancyIdxNumpyVsBlosc.png', format="png")
    plt.show()
else:
    zarrtimes = np.array(zarrtimes)
    h5pytimes = np.array(h5pytimes)

    # Create bars for axis 0 plot
    for i, r in enumerate((["Numpy",nptimes,-2*width],["Blosc2",blosctimes, -width],["Zarr",zarrtimes, 0],["HDF5",h5pytimes, width])):
        label,times,w = r
        c = ['b', 'r', 'g', 'm'][i]
        plt.bar(x + w, times.mean(axis=1), width, color=c, alpha=0.5)
        plt.bar(x + w, times.max(axis=1), width, color=c, alpha=0.25)
        plt.bar(x + w, times.min(axis=1), width, label=label, color=c, alpha=1)

    plt.xlabel('Array size (GB)')
    plt.legend()
    plt.xticks(x, np.round(sizes, 2))
    plt.ylabel("Time (s)")
    plt.title('Fancy indexing performance comparison')
    plt.savefig('plots/fancyIdx.png', format="png")
    plt.show()

print("Finished everything!")
