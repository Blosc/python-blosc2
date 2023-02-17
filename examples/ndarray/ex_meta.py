#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os

import blosc2
import numpy as np

shape = (128, 128)

urlpath = "ex_meta.b2nd"
blosc2.remove_urlpath(urlpath)

dtype = np.dtype(np.complex128)

# Create a numpy array
nparray = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)

meta = {
    "m1": b"1111",
    "m2": b"2222",
}
# Create a NDArray from a numpy array (on disk)
a = blosc2.frombuffer(bytes(nparray), nparray.shape, urlpath=urlpath, dtype=dtype, meta=meta)
print(a.info)

# Read a b2nd array from disk
b = blosc2.open(urlpath)

# Deal with meta
m1 = b.schunk.meta.get("m5", b"0000")
m2 = b.schunk.meta["m2"]

# Remove file on disk
os.remove(urlpath)
