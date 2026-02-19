#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2


@blosc2.dsl_kernel
def _wasm_kernel(x, y):
    return (x + y) * 1.5 - 0.25


@pytest.mark.skipif(not blosc2.IS_WASM, reason="WASM-only integration test")
def test_wasm_dsl_tcc_jit_smoke():
    assert getattr(blosc2, "_WASM_MINIEXPR_ENABLED", False)

    a_np = np.linspace(-1.0, 1.0, 64, dtype=np.float64).reshape(8, 8)
    b_np = np.linspace(0.0, 2.0, 64, dtype=np.float64).reshape(8, 8)
    a = blosc2.asarray(a_np, chunks=(4, 4), blocks=(2, 2))
    b = blosc2.asarray(b_np, chunks=(4, 4), blocks=(2, 2))

    expr = blosc2.lazyudf(_wasm_kernel, (a, b), dtype=np.float64)
    out = expr.compute(jit=True, jit_backend="tcc", strict_miniexpr=True)
    expected = (a_np + b_np) * 1.5 - 0.25
    np.testing.assert_allclose(out[...], expected, rtol=1e-6, atol=1e-8)
