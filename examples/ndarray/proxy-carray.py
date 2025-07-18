#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Shows how you can make a proxy of a remote array (served with Caterva2) on disk
# Note that, for running this example, you will need the blosc2-grok package.

import os
from time import time

import blosc2

urlbase = "https://cat2.cloud/demo"
path = "@public/examples/lung-jpeg2000_10x.b2nd"
a = blosc2.C2Array(path, urlbase=urlbase)
b = blosc2.Proxy(a, urlpath="proxy.b2nd", mode="w")

# Check metadata (note that all should be the same)
print("*** Metadata ***")
print(f"Codec in 'a': {a.cparams.codec}")
print(f"Codec in 'b': {b.cparams.codec}")
print(f"Filters in 'a': {a.cparams.filters}")
print(f"Filters in 'b': {b.cparams.filters}")

# Check array properties
print("*** Array properties ***")
print(f"Shape in 'a': {a.shape}")
print(f"Shape in 'b': {b.shape}")
print(f"Type in 'a': {a.dtype}")
print(f"Type in 'b': {b.dtype}")

print("*** Fetching data ***")
t0 = time()
print(f"Data in 'a': {a[0, 0, 0:10]}")
print(f"Time to fetch data in 'a': {time() - t0:.3f}s")
t0 = time()
print(f"Data in 'b': {b[0, 0, 0:10]}")
print(f"Time to fetch data in 'b': {time() - t0:.3f}s")
t0 = time()
print(f"Data in 'b': {b[0, 0, 0:10]}")
print(f"Time to fetch data in 'b' (cached): {time() - t0:.3f}s")

# Check sizes. Note that the proxy will only have the 'touched' chunks (only 1 in this case)
print("*** Sizes ***")
print(f"Size in 'a': {a.meta['schunk']['cbytes']}")
print(f"Size in 'b': {b.schunk.cbytes}")
# Check sizes on disk
print("*** Disk sizes ***")
print(f"Size 'b' (disk): {os.stat(b.urlpath).st_size}")
