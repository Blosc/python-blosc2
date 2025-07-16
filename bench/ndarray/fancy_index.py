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
import matplotlib.pyplot as plt
import zarr
import h5py
import pickle
import os
plt.rcParams.update({'text.usetex':False,'font.serif': ['cm'],'font.size':16})
plt.rcParams['figure.dpi'] = 1000
plt.rcParams['savefig.dpi'] = 1000
plt.rc('text', usetex=False)
plt.rc('font',**{'serif':['cm']})
plt.style.use('seaborn-v0_8-paper')

NUMPY_BLOSC = True # activate NUMPY and BLOSC tests
NUMPY_BLOSC_ZARR = False # activate NUMPY, BLOSC and Zarr tests
# default if both are false is to run tests for Numpy, Blosc, Zarr and HDF5

def genarray(r, ndims=1, verbose=True):
    d = int((r*2**30/8)**(1/ndims))
    shape = (d,) * ndims
    chunks = (d // 4,) * ndims
    blocks = (max(d // 10, 1),) * ndims
    t = time.time()
    if os.path.exists(f'linspace{r}.b2nd'):
        arr = blosc2.open(urlpath=f'linspace{r}.b2nd')
    else:
        arr = blosc2.linspace(0, 1000, num=np.prod(shape), shape=shape, dtype=np.float64, urlpath=f'linspace{r}.b2nd', mode='w')
    t = time.time() - t
    if verbose:
        print(f"Array shape: {arr.shape}")
        print(f"Array size: {np.prod(arr.shape) * arr.dtype.itemsize / 2 ** 30:.6f} GB")
        print(f"Time to create array: {t:.6f} seconds")
    return arr


sizes = np.int64(np.array([1, 2, 4, 8, 16, 24]))
rng = np.random.default_rng()
blosctimes = []
nptimes = []
zarrtimes = []
h5pytimes = []
x = np.arange(len(sizes))
width = 0.2
labs = 'NumpyBlosc2' if NUMPY_BLOSC else 'NumpyBlosc2ZarrHDF5'
labs = 'NumpyBlosc2Zarr' if NUMPY_BLOSC_ZARR else labs
try:
    with open(f"results{labs}.pkl", 'rb') as f:
        result_tuple = pickle.load(f)
    labs = ''
    for i, r in enumerate(result_tuple):
        if r[1].shape != (0,):
            label,times,w = r
            c = ['b', 'r', 'g', 'm'][i]
            mean = times.mean(axis=1)
            err = [mean - times.min(axis=1), times.max(axis=1)-mean]
            plt.bar(x + w, mean , width, color=c, label=label, yerr=err, capsize=5, ecolor='k',
            error_kw=dict(lw=2, capthick=2, ecolor='k'))
            labs+=label
except:
    for d in sizes:
        arr = genarray(d, ndims=2)
        idx = rng.integers(low=0, high=arr.shape[0], size=(arr.shape[0],))
        row = np.sort(np.unique(idx))
        col = np.sort(np.unique(rng.integers(low=0, high=arr.shape[0], size=(arr.shape[0],))))
        mask = rng.integers(low=0, high=2, size=(arr.shape[0],)) == 1


        ## Test fancy indexing for different use cases
        m, M = np.min(idx), np.max(idx)
        def timer(arr, row=row, col=col):
            time_list = []
            if NUMPY_BLOSC or NUMPY_BLOSC_ZARR:
                t = time.time()
                b = arr[row, col]
                time_list += [time.time() - t]
            if NUMPY_BLOSC:
                t = time.time()
                b = arr[slice(1, M // 2, 5), col]
                time_list += [time.time() - t]
                t = time.time()
                b = arr[[[row], [col]]]
                time_list += [time.time() - t]
                t = time.time()
                b = arr[row[:10, None], col[:10]]
                time_list += [time.time() - t]
                t = time.time()
                b = arr[row[:10, None], mask]
                time_list += [time.time() - t]
            t = time.time()
            b = arr[row]
            time_list += [time.time() - t]
            t = time.time()
            b = arr[m, col]
            time_list += [time.time() - t]
            return np.array(time_list)

        if NUMPY_BLOSC or NUMPY_BLOSC_ZARR:
            blosctimes += [timer(arr, row=idx, col=idx)]
            arr=arr[:]
            nptimes += [timer(arr, row=idx, col=idx)]
            if NUMPY_BLOSC_ZARR:
                z_test = zarr.zeros(shape=arr.shape, dtype=arr.dtype)
                z_test[:] = arr
                zarrtimes += [timer(z_test, row=idx, col=idx)]
        else:
            blosctimes += [timer(arr)]
            arr=arr[:]
            nptimes += [timer(arr)]
            z_test = zarr.zeros(shape=arr.shape, dtype=arr.dtype)
            z_test[:] = arr
            zarrtimes += [timer(z_test)]
            with h5py.File('my_hdf5_file.h5', 'w') as f:
                    dset = f.create_dataset("init", data=arr)
                    h5pytimes += [timer(dset)]

    blosctimes = np.array(blosctimes)
    nptimes = np.array(nptimes)
    zarrtimes = np.array(zarrtimes)
    h5pytimes = np.array(h5pytimes)
    labs=''
    result_tuple = (["Numpy",nptimes,-2*width],["Blosc2",blosctimes, -width],["Zarr",zarrtimes, 0],["HDF5",h5pytimes, width])

    # Create barplot for Numpy vs Blosc vs Zarr vs H5py
    for i, r in enumerate(result_tuple):
        if r[1].shape != (0,):
            label, times, w = r
            c = ['b', 'r', 'g', 'm'][i]
            mean = times.mean(axis=1)
            err = (mean - times.min(axis=1), times.max(axis=1)-mean)
            plt.bar(x + w, mean , width, color=c, label=label, yerr=err, capsize=5, ecolor='k',
            error_kw=dict(lw=2, capthick=2, ecolor='k'))
            labs+=label

    with open(f"results{labs}.pkl", 'wb') as f:
        pickle.dump(result_tuple, f)

plt.xlabel('Array size (GB)')
plt.legend()
plt.xticks(x-width, np.round(sizes, 2))
plt.ylabel("Time (s)")
plt.title('Fancy indexing performance comparison')
# plt.ylim([0,10])
plt.gca().set_yscale('log')
plt.savefig(f'plots/fancyIdx{labs}.png', format="png")
plt.show()

print("Finished everything!")
