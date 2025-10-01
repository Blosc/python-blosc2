#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Example on how to use xarray containers as operands in Blosc2 expressions
# Note that there is no special support for xarray in Blosc2; the techniques
# below works for any object that implements the Array protocol (i.e. having
# a shape and dtype attributes, and a __getitem__ method and a __len__ method.

import numpy as np
import xarray

import blosc2


class NewObj(blosc2.Array):
    def __init__(self, a):
        self.a = a

    @property
    def shape(self):
        return self.a.shape

    @property
    def dtype(self):
        return self.a.dtype

    def __getitem__(self, key):
        return self.a[key]

    def __len__(self):
        return len(self.a)


a = np.arange(100, dtype=np.int64).reshape(10, 10)
res = a + np.sin(a) + np.hypot(a, a) + 1

a = xarray.DataArray(a)  # supported natively by blosc2; no copies
b = NewObj(a)  #  minimal Array protocol implementation; no copies
assert isinstance(b, blosc2.Array)  # any Array compliant object works
c = blosc2.asarray(a)  # convert into a blosc2.NDArray; data is copied
d = blosc2.SimpleProxy(a)  # SimpleProxy conversion; no copies
# Define a lazy expression (defer computation until needed)
lb = blosc2.lazyexpr("a + sin(b) + hypot(c, d) + 1")

# Check!
np.testing.assert_array_equal(lb[:], res)
# One can also evaluate the expression directly (eager computation)
resb2 = blosc2.evaluate("a + sin(b) + hypot(c, d) + 1")
np.testing.assert_array_equal(resb2, res)
