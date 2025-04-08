#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((100, 1230), np.float64),
        ((23, 34), np.float32),
        ((80, 51, 60), "f4"),
        ((13, 13), None),
    ],
)
def test_nans_simple(shape, dtype):
    a = blosc2.nans(shape, dtype=dtype)
    assert a.dtype == np.dtype(dtype) if dtype is not None else np.dtype(np.float64)

    b = np.full(shape=shape, fill_value=np.nan, dtype=a.dtype)
    np.testing.assert_allclose(a[...], b)


@pytest.mark.parametrize("asarray", [True, False])
@pytest.mark.parametrize("typesize", [1, 3, 255, 256, 257, 256 * 256])
@pytest.mark.parametrize("shape", [(1,), (3,), (10,), (2 * 10,)])
def test_large_typesize(shape, typesize, asarray):
    dtype = np.dtype([("f_001", "f8", (typesize,)), ("f_002", "f4", (typesize,))])
    a = np.full(shape, np.nan, dtype=dtype)
    if asarray:
        b = blosc2.asarray(a)
    else:
        # b = blosc2.nans(shape, dtype=dtype)  # TODO: this is not working; perhaps deprecate blosc2.nans()?
        b = blosc2.full(shape, np.nan, dtype=dtype)
    for field in dtype.fields:
        np.testing.assert_allclose(b[field][:], a[field], equal_nan=True)
