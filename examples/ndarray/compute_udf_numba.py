#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# This shows how to evaluate expressions with NDArray instances as operands.

import numba as nb
import numpy as np

import blosc2


# The UDF to be evaluated
@nb.jit(nopython=True, parallel=True)
def func_numba(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    output[:] = x + 1


# Create a NDArray from a NumPy array
shape = (13, 13)
npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
npc = npa + 1
a = blosc2.asarray(npa)

lazyarray = blosc2.lazyudf(func_numba, (npa,), npa.dtype)
print(lazyarray.info)
res = lazyarray.compute()
print(res.info)
np.testing.assert_allclose(res[...], npc)
print("Numba + LazyArray evaluated correctly!")
