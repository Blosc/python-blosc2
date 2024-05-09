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

shape = (13, 13)
dtype = np.float64

# Create a NDArray from a NumPy array
npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
npc = npa + 1
a = blosc2.asarray(npa)


@nb.jit(nopython=True, parallel=True)
def func_numba(inputs_tuple, output, offset):
    x = inputs_tuple[0]
    output[:] = x + 1


chunks = [10, 10]
expr = blosc2.lazyudf(func_numba, (npa,), npa.dtype, chunks=chunks, blocks=chunks)
res = expr.eval()
print(res.info)
np.testing.assert_allclose(res[...], npc)

print(func_numba)
print(func_numba.__name__)
print(func_numba.__annotations__)
print(func_numba.__code__)
print(func_numba.__code__.co_code)
print(func_numba.__code__.co_linetable)
print(func_numba.__code__.co_lines())


import inspect
lines = inspect.getsource(func_numba)
print(lines)
print(type(lines))
res.schunk.vlmeta["sb"] = lines
print(res.schunk.vlmeta["sb"])


