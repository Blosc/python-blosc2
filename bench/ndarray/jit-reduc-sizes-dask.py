#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Compute reductions for different array sizes, using the jit decorator
# and different operands (NumPy and NDArray).  Different compression
# levels and codecs can be selected.

from time import time
import blosc2
import numpy as np
import sys
import dask
import dask.array as da
import zarr
from numcodecs import Blosc

niter = 5
#dtype = np.dtype("float32")
dtype = np.dtype("float64")
clevel = 1
numpy = False
numpy_jit = False
dask_da = False
cparams = cparams_out = None
check_result = False

# For 64 GB RAM
# sizes_numpy = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55)
# sizes_numpy_jit = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70)
# sizes_clevel0 = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70)
# size_list = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110)  # limit clevel>=1 float64

# For 24 GB RAM
sizes_numpy = (1, 5, 10, 20, 30)  # limit numpy float64
sizes_numpy_jit = (1, 5, 10, 20, 30)  # limit numpy float64
sizes_clevel0 = (1, 5, 10, 20, 30)  # limit clevel==0 float64
size_list = (1, 5, 10, 20, 30)

codec = "LZ4"  # default codec
if len(sys.argv) > 2:
    codec = sys.argv[2]
if len(sys.argv) > 1:
    try:
        clevel = int(sys.argv[1])
    except ValueError:
        clevel = 0
        if sys.argv[1] == "numpy":
            numpy = True
        elif sys.argv[1] == "numpy_jit":
            numpy = True
            numpy_jit = True
        else:
            raise ValueError("Invalid argument")

if check_result:
    print("*** Enabling check_result: beware that this will slow down the benchmarking!")

if len(sys.argv) > 3:
    if sys.argv[3] == "dask":
        dask_da = True


# The reductions to compute
def compute_reduction_numpy(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

@blosc2.jit
def compute_reduction(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

def compute_reduction_dask(a, b, c):
    return (((a ** 3 + da.sin(a * 2)) < c) & (b > 0)).sum(axis=1)


# Compute for both disk or memory
#for disk in (True, False):
for disk in (False,):
    if disk and (numpy or numpy_jit or dask_da):
        continue
    print(f"\n*** Using disk={disk} ***\n")
    apath = bpath = None
    if numpy:
        print("Using NumPy arrays as operands")
    else:
        print("Using NDArray arrays as operands")
        cparams = cparams_out = blosc2.CParams(clevel=clevel, codec=blosc2.Codec[codec])
        # zcodecs = zcodecs_out = zarr.codecs.BloscCodec(
        #     cname=codec.lower(), clevel=clevel, shuffle=zarr.codecs.BloscShuffle.shuffle)
        zcompressor = zcompressor_out = Blosc(cname=codec.lower(), clevel=clevel, shuffle=Blosc.SHUFFLE)
        # cparams_out = blosc2.CParams(clevel=clevel, codec=blosc2.Codec.LZ4)
        print("Using cparams: ", cparams)
        if disk:
            apath = "a.b2nd"
            bpath = "b.b2nd"

    create_times = []
    compute_times = []
    # Iterate over different sizes
    for n in size_list:
        if clevel == 0 and n not in sizes_clevel0:
            continue
        if numpy_jit and n not in sizes_numpy_jit:
            continue
        if numpy and not numpy_jit and n not in sizes_numpy:
            continue
        N = n * 1000
        print(f"\nN = {n}000, {dtype=}, size={N ** 2 * 2 * dtype.itemsize / 2**30:.3f} GB")
        chunks = (100, N)
        blocks = (1, N)
        #chunks, blocks = None, None  # automatic chunk and block sizes
        # Lossy compression
        #filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
        #filters_meta = [8, 0]  # keep 8 bits of precision in mantissa
        #cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=filters, filters_meta=filters_meta)

        # Create some data operands
        if check_result or dask_da:
            na = np.linspace(0, 1, N * N, dtype=dtype).reshape(N, N)
            nb = na + 1
            nc = np.linspace(-10, 10, N, dtype=dtype)
            nout = compute_reduction_numpy(na, nb, nc)
        t0 = time()
        if numpy or numpy_jit and not dask_da:
            na = np.linspace(0, 1, N * N, dtype=dtype).reshape(N, N)
            nb = na + 1
            nc = np.linspace(-10, 10, N, dtype=dtype)
        elif dask_da:
            # Use zarr for operands
            za = zarr.array(na, chunks=chunks, compressor=zcompressor, zarr_format=2)
            zb = zarr.array(nb, chunks=chunks, compressor=zcompressor, zarr_format=2)
            zc = zarr.array(nc, chunks=chunks[1], compressor=zcompressor, zarr_format=2)
        else:
            a = blosc2.linspace(0, 1, N * N, dtype=dtype, shape=(N, N), cparams=cparams, urlpath=apath, mode="w")
            #print("a.chunks, a.blocks, a.schunk.cratio: ", a.chunks, a.blocks, a.schunk.cratio)
            print(f"{a.chunks=}, {a.blocks=}, {a.schunk.cratio=:.2f}x")

            b = blosc2.linspace(1, 2, N * N, dtype=dtype, shape=(N, N), cparams=cparams, urlpath=bpath, mode="w")
            #b = (a + 1).compute(cparams=cparams, chunks=chunks, blocks=blocks)
            #print(b.chunks, b.blocks, b.schunk.cratio, b.cparams)
            c = blosc2.linspace(-10, 10, N, dtype=dtype, cparams=cparams)  # broadcasting is supported
            #c = blosc2.linspace(-10, 10, N * N, dtype=dtype, shape=(N, N), cparams=cparams)
        t1 = time() - t0
        print(f"Time to create data: {t1:.4f}")
        create_times.append(t1)

        if numpy and not dask_da:
            if numpy_jit and not numpy:
                out = compute_reduction(na, nb, nc)
                t0 = time()
                for i in range(niter):
                    out = compute_reduction(na, nb, nc)
                t1 = (time() - t0) / niter
                print(f"Time to compute with numpy_jit and NumPy operands: {t1:.4f}")
            else:
                t0 = time()
                nout = compute_reduction_numpy(na, nb, nc)
                t1 = time() - t0
                print(f"Time to compute with NumPy engine: {t1:.4f}")
        elif dask_da:
            niter = 1
            if numpy:
                a = na
                b = nb
                c = nc
            else:
                a = da.from_zarr(za)
                b = da.from_zarr(zb)
                c = da.from_zarr(zc)

            scheduler = "single-threaded" if blosc2.nthreads == 1 else "threads"
            t0 = time()
            for i in range(niter):
                if numpy:
                    dexpr = da.map_blocks(compute_reduction_dask, a, b, c)
                    out = dexpr.compute(scheduler=scheduler)
                else:
                    dexpr = (((a ** 3 + da.sin(a * 2)) < c) & (b > 0)).sum(axis=1)
                    zout = zarr.open(shape=(N,), chunks=chunks[1], dtype=dtype, compressor=zcompressor_out, zarr_format=2)
                    with dask.config.set(scheduler=scheduler, num_workers=blosc2.nthreads):
                        da.to_zarr(dexpr, zout)
                    if check_result and i == 0:
                        out = zout[:]
            t1 = (time() - t0) / niter
            print(f"Time to compute with dask and {clevel=}: {t1:.4f}")
            if check_result:
                np.testing.assert_allclose(out, nout)
        else:
            # out = compute_reduction(a, b, c)
            t0 = time()
            for i in range(niter):
                out = compute_reduction(a, b, c)
            t1 = (time() - t0) / niter
            print(f"Time to compute with blosc2_jit and {clevel=}: {t1:.4f}")
        compute_times.append(t1)
        #del a, b, c

    print("\nCreate times: [", ", ".join([f"{t:.4f}" for t in create_times]), "]")
    print("Compute times: [", ", ".join([f"{t:.4f}" for t in compute_times]), "]")
    print("End of run!\n\n")
