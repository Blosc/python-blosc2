#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compute reductions for different array sizes, using the jit decorator
# and different operands (NumPy and NDArray).  Different compression
# levels and codecs can be selected.

from time import time
import blosc2
import numpy as np
import sys

niter = 5
#dtype = np.dtype("float32")
dtype = np.dtype("float64")
clevel = 1
numpy = False
numpy_jit = False
cparams = cparams_out = None

# For 64 GB RAM
# sizes_numpy = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55)
# sizes_numpy_jit = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70)
# sizes_clevel0 = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70)
# size_list = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110)  # limit clevel>=1 float64

# For 24 GB RAM
sizes_numpy = (1, 5, 10, 20, 30, 35, 40)  # limit numpy float64
sizes_numpy_jit = (1, 5, 10, 20, 30, 35, 40, 45)  # limit numpy float64
sizes_clevel0 = (1, 5, 10, 20, 30, 35, 40, 45)  # limit clevel==0 float64
#sizes_clevel0 = (50, 55, 60, 65, 70)  # extra sizes for clevel==0 float64
size_list = (1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90)  # limit clevel>=1 float64

codec = "LZ4"  # default codec
if len(sys.argv) > 2:
    codec = sys.argv[2]
if len(sys.argv) > 1:
    try:
        clevel = int(sys.argv[1])
    except ValueError:
        if sys.argv[1] == "numpy":
            numpy = True
        elif sys.argv[1] == "numpy_jit":
            numpy = True
            numpy_jit = True
        else:
            raise ValueError("Invalid argument")


# The reductions to compute
def compute_reduction_numpy(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)

@blosc2.jit
def compute_reduction(a, b, c):
    return np.sum(((a ** 3 + np.sin(a * 2)) < c) & (b > 0), axis=1)


# Compute for both disk or memory
for disk in (True, False):
    print(f"\n*** Using disk={disk} ***\n")
    apath = bpath = None
    if numpy:
        print("Using NumPy arrays as operands")
    else:
        print("Using NDArray arrays as operands")
        cparams = cparams_out = blosc2.CParams(clevel=clevel, codec=blosc2.Codec[codec])
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
        chunks, blocks = None, None  # automatic chunk and block sizes
        # Lossy compression
        #filters = [blosc2.Filter.TRUNC_PREC, blosc2.Filter.SHUFFLE]
        #filters_meta = [8, 0]  # keep 8 bits of precision in mantissa
        #cparams = blosc2.CParams(clevel=1, codec=blosc2.Codec.LZ4, filters=filters, filters_meta=filters_meta)

        # Create some data operands
        t0 = time()
        if numpy:
            a = np.linspace(0, 1, N * N, dtype=dtype).reshape(N, N)
            b = np.linspace(1, 2, N * N, dtype=dtype).reshape(N, N)
            #b = a + 1
            c = np.linspace(-10, 10, N, dtype=dtype)
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

        if numpy:
            if numpy_jit:
                out = compute_reduction(a, b, c)
                t0 = time()
                for i in range(niter):
                    out = compute_reduction(a, b, c)
                t1 = (time() - t0) / niter
                print(f"Time to compute with numpy_jit and NumPy operands: {t1:.4f}")
            else:
                t0 = time()
                nout = compute_reduction_numpy(a, b, c)
                t1 = time() - t0
                print(f"Time to compute with NumPy engine: {t1:.4f}")
        else:
            out = compute_reduction(a, b, c)
            t0 = time()
            for i in range(niter):
                out = compute_reduction(a, b, c)
            t1 = (time() - t0) / niter
            print(f"Time to compute with numpy_jit and {clevel=}: {t1:.4f}")
        compute_times.append(t1)
        del a, b, c

    print("\nCreate times: [", ", ".join([f"{t:.4f}" for t in create_times]), "]")
    print("Compute times: [", ", ".join([f"{t:.4f}" for t in compute_times]), "]")
    print("End of run!\n\n")
