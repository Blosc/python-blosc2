#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Store metadata in persistent arrays

import numpy as np

import blosc2

shape = (128, 128)
urlpath = "ex_meta.b2nd"
dtype = np.complex128

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

meta = {
    "m1": b"1111",
    "m2": b"2222",
}
# Create a NDArray from a numpy array (on disk)
a = blosc2.frombuffer(bytes(nparray), nparray.shape, urlpath=urlpath, mode="w", dtype=dtype, meta=meta)
print(a.info)

# Read a b2nd array from disk
b = blosc2.open(urlpath)

# Deal with meta
m1 = b.schunk.meta.get("m5", b"0000")
m2 = b.schunk.meta["m2"]
print("m1 meta:", m1)
print("m2 meta:", m2)
