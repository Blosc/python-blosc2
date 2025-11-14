#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import math
import os
import pprint

import numpy as np
from time import time
import blosc2

dtype = np.float32
shape = (20_000, 20_000)
nelem = math.prod(shape)
large_shape = (50,) + shape
large_nelem = math.prod(large_shape)

# cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=9)
cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=9)
# cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=1)

# a = np.ones(shape, dtype=np.float64)
# a = np.arange(nelem, dtype=np.float64).reshape(shape)
a = np.linspace(0, 1, nelem, dtype=dtype).reshape(shape)


def compute_example(a, large_a=None, intensity=None, cparams=None):
    t0 = time()
    nelem_compute = nelem if type(a) is np.ndarray else large_nelem
    if intensity == "low":
        if type(a) is not np.ndarray:
            a = large_a
        b = a + 1
        intensity = 1 / 8  # 1 FLOPs / (2 * 4 bytes)
    elif intensity == "medium":
        if type(a) is not np.ndarray:
            a = large_a
        b = np.sqrt(a + 2 * a + (a / 2)) ** 1.2
        intensity = 22 / 8  # 22 FLOPs / (2 * 4 bytes)
    elif intensity == "high":
        if type(a) is not np.ndarray:
            a = large_a
        b = np.exp(np.sqrt(np.sin(a) ** 2 + (np.cos(a) + np.arctan(a)) ** 3))
        intensity = 133 / (2 * 4)  # 133 FLOPs / (2 * 4 bytes)
    elif intensity.startswith("matmul"):
        if type(a) is not np.ndarray:
            if intensity == "matmul2":
                c = a
            elif intensity == "matmul1":
                n = shape[0] // 2
                c = a.slice((slice(n, n + n), slice(n, n + n)))
            elif intensity == "matmul0":
                n = shape[0] // 5
                c = a.slice((slice(n, n + n), slice(n, n + n)))
            else:
                raise ValueError("Invalid intensity")
            b = blosc2.matmul(c, c, cparams=cparams, urlpath="result_array.b2nd", mode="w")
            # print(f"b.chunks: {b.chunks}, b.blocks: {b.blocks}")
        else:
            if intensity == "matmul2":
                c = a
            elif intensity == "matmul1":
                n = shape[0] // 2
                c = a[n:n+n, n:n+n]
            elif intensity == "matmul0":
                n = shape[0] // 5
                c = a[n:n+n, n:n+n]
            else:
                raise ValueError("Invalid intensity")
            b = np.matmul(c, c)
        nelem_compute = b.size
        # print(type(c), type(b), nelem)
        intensity = int((2 * c.shape[0]) / 3)
    else:
        raise ValueError("Invalid intensity")
    print(f"Intensity = {intensity}", end=", ")
    if hasattr(b, "compute"):
        b = b.compute(cparams=cparams, urlpath="result_array.b2nd", mode="w")
        # print(f"cratio result = {b.cratio:.2f}")
    t = (time() - t0)
    print(f"Time = {t:.2f}s", end=", ")
    gflops = intensity * nelem_compute / t / 10**9
    print(f"GFLOPS = {gflops:.2f}", end=", ")
    print(f"Mem BW = {nelem_compute * b.dtype.itemsize * 2 / (t * 10**9):.2f} GB/s")
    return {"GFLOPS": gflops, "Intensity": intensity, "Time": t}

results = {"numpy": {}, "blosc2": {}, "blosc2-nocomp": {}}

print("Shape:", shape)
print("*** Numpy array")
intensities = ["low", "medium", "high", "matmul0", "matmul1", "matmul2"]
for intensity in intensities:
    results["numpy"][intensity] = compute_example(a, None, intensity)

print("*** Blosc2 array (compressed, on-disk)")
b2a = blosc2.asarray(a, cparams=cparams, urlpath="a_array.b2nd", mode="w")
large_a = blosc2.linspace(0, 1, large_nelem, dtype=dtype, shape=large_shape,
                          cparams=cparams, urlpath="large_array.b2nd", mode="w")
#print(f"large_a.cratio = {large_a.cratio:.2f}, b2a.cratio = {b2a.cratio:.2f}")
for intensity in intensities:
    results["blosc2"][intensity] = compute_example(b2a, large_a, intensity, cparams)

print("*** Blosc2 array (no compressed)")
no_compr_cparams = blosc2.CParams(clevel=0)
b2a = blosc2.asarray(a, cparams=no_compr_cparams, urlpath="a_array.b2nd", mode="w")
large_a = blosc2.copy(large_a, cparams=no_compr_cparams, urlpath="large_array2.b2nd", mode="w")
#print(f"large_a.cratio = {large_a.cratio:.2f}, b2a.cratio = {b2a.cratio:.2f}")
for intensity in intensities:
    results["blosc2-nocomp"][intensity] = compute_example(b2a, large_a, intensity, no_compr_cparams)

pprint.pprint(results)

os.unlink("a_array.b2nd")
os.unlink("large_array.b2nd")
os.unlink("large_array2.b2nd")
os.unlink("result_array.b2nd")
