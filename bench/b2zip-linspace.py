#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This compares performance of creating and reading a NumPy array in different ways:
# 1) memory
# 2) disk
# 3) disk with b2zip format

import blosc2

from time import time

# Number of elements in array
N = 2**27

def b2_native(urlpath=None):
    t0 = time()
    a = blosc2.linspace(0., 1., N, urlpath=urlpath, mode="w")
    # a = blosc2.linspace(0., 1., 2**27, cparams=blosc2.CParams(codec=blosc2.Codec.LZ4))
    # a = blosc2.linspace(0., 1., 2**27, dparams=blosc2.DParams(nthreads=1))
    t1 = time()
    print(f"Time to create a linspace array: {t1 - t0:.2f}s, bandwidth: {a.nbytes / (t1 - t0) / 1e9:.2f} GB/s")
    #print(a.info)

    t0 = time()
    b = a[:]
    t1 = time()
    print(f"Time to read the array: {t1 - t0:.2f}s, bandwidth: {b.nbytes / (t1 - t0) / 1e9:.2f} GB/s")

def b2_b2zip(urlpath):
    t0 = time()
    with blosc2.TreeStore(localpath=urlpath, mode="w") as tstore:
        a = blosc2.linspace(0., 1., N)
        # a = blosc2.linspace(0., 1., 2**27, cparams=blosc2.CParams(codec=blosc2.Codec.LZ4))
        tstore["/b"] = a
    t1 = time()
    print(f"Time to store a linspace array: {t1 - t0:.2f}s, bandwidth: {a.nbytes / (t1 - t0) / 1e9:.2f} GB/s")

    t0 = time()
    with blosc2.TreeStore(localpath=urlpath, mode="r") as tstore_read:
        b = tstore_read["/b"][:]
    t1 = time()
    print(f"Time to read the array: {t1 - t0:.2f}s, bandwidth: {b.nbytes / (t1 - t0) / 1e9:.2f} GB/s")


if __name__ == "__main__":
    print("Blosc2 in-memory")
    b2_native()
    print("Blosc2 on disk")
    b2_native("linspace.b2nd")
    print("Blosc2 on disk with b2zip format")
    b2_b2zip("my_tstore.b2z")
