#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Shows how you can make a proxy of a local array on disk.

import os

import blosc2

cparams = blosc2.CParams(
    clevel=5, codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.BITSHUFFLE], filters_meta=[0]
)

cwd = os.getcwd()
a = blosc2.full((128, 128), 1, dtype="float64", urlpath=f"{cwd}/a.b2nd", mode="w", cparams=cparams)
b = blosc2.Proxy(a, urlpath=f"{cwd}/proxy.b2nd", mode="w")

# Check metadata
print("*** Metadata ***")
print(f"Codec in 'a': {a.cparams.codec}")
print(f"Codec in 'b': {b.cparams.codec}")
print(f"Clevel in 'a': {a.cparams.clevel}")
print(f"Clevel in 'b': {b.cparams.clevel}")
print(f"Filters in 'a': {a.cparams.filters}")
print(f"Filters in 'b': {b.cparams.filters}")

# Check array properties
print("*** Array properties ***")
print(f"Shape in 'a': {a.shape}")
print(f"Shape in 'b': {b.shape}")
print(f"Type in 'a': {a.dtype}")
print(f"Type in 'b': {b.dtype}")

# Check data
print("*** Fetching data ***")
print(f"Data in 'a': {a[0, 0:10]}")
print(f"Data in 'b': {b[0, 0:10]}")

# Check sizes. Note that the proxy will only have the 'touched' chunks (only 1 in this case)
print("*** Sizes ***")
print(f"Size in 'a': {a.schunk.cbytes}")
print(f"Size in 'b': {b.schunk.cbytes}")
# Check sizes on disk
print("*** Disk sizes ***")
print(f"Size 'a' (disk): {os.stat(a.urlpath).st_size}")
print(f"Size 'b' (disk): {os.stat(b.urlpath).st_size}")

# Check vlmeta
print("*** VLmeta ***")
print(f"VLmeta in 'a': {list(a.vlmeta)}")
print(f"VLmeta in 'b': {list(b.vlmeta)}")
