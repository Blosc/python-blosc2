#######################################################################
# Copyright (C) 2019-present, Blosc Development team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import pytest

import blosc2
import numpy as np


@pytest.mark.parametrize("shape, chunks, blocks, fill_value, dtype, cparams, dparams",
                         [
                             ((100, 1230), (200, 100), (55, 3), b"0123", None,
                              {"clevel": 4, "use_dict": 0, "nthreads": 1}, {"nthreads": 1}),
                             ((23, 34), (20, 20), (10, 10), b"sun", None,
                              {"codec": blosc2.Codec.LZ4HC, "clevel": 8, "use_dict": False, "nthreads": 2},
                              {"nthreads": 2}),
                             ((80, 51, 60), (20, 10, 33), (6, 6, 26), 3.14, np.float64,
                              {"codec": blosc2.Codec.ZLIB, "clevel": 5, "use_dict": True, "nthreads": 2},
                              {"nthreads": 1}),
                             ((13, 13), (12, 12), (11, 11), 123456789, None,
                              {"codec": blosc2.Codec.LZ4HC, "clevel": 8, "use_dict": False, "nthreads": 2},
                              {"nthreads": 2})
                         ])
def test_full(shape, chunks, blocks, fill_value, cparams, dparams, dtype):
    a = blosc2.full(shape, chunks, blocks, fill_value, dtype=dtype, cparams=cparams, dparams=dparams)
    assert a.schunk.dparams == dparams
    if isinstance(fill_value, bytes):
        dtype = np.dtype(f"S{len(fill_value)}")
    assert a.dtype == np.dtype(dtype) if dtype is not None else np.dtype(np.uint8)

    b = np.full(shape=shape, fill_value=fill_value, dtype=a.dtype)
    tol = 1e-5 if dtype is np.float32 else 1e-14
    if dtype in [np.float32, np.float64]:
        np.testing.assert_allclose(a[...], b, rtol=tol, atol=tol)
    else:
        np.array_equal(a[...], b)
