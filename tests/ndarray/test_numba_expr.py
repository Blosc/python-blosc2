#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest
import numba as nb

import blosc2

@pytest.mark.parametrize(
    "shape, chunks, blocks",
    [
        ((10, 10), (10, 10), (10, 10),),
        ((20, 20), (10, 10), (10, 10),),
        ((20, 20), (10, 10), (5, 5),),
        ((13, 13), (10, 10), (10, 10),),
        ((13, 13), (10, 10), (5, 5),),
        ((10, 10), (10, 10), (4, 4),),
        ((13, 13), (10, 10), (4, 4),),
    ],
)
def test_numba_expr(shape, chunks, blocks):  #, dtype, urlpath, contiguous
    @nb.jit(nopython=True, parallel=True)
    def func_numba(inputs_tuple, output, offset):
        x = inputs_tuple[0]
        output[:] = x + 1

    # Create a NDArray from a NumPy array
    npa = np.linspace(0, 1, np.prod(shape)).reshape(shape)
    npc = npa + 1

    # a = blosc2.asarray(npa)
    expr = blosc2.expr_from_udf(func_numba, ((npa, npa.dtype), ), npa.dtype, chunks=chunks, blocks=blocks)
    res = expr.eval()
    tol = 1e-5 if res.dtype is np.float32 else 1e-14
    if res.dtype in (np.float32, np.float64):
        np.testing.assert_allclose(res[...], npc, rtol=tol, atol=tol)

