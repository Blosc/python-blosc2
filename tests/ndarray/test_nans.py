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
    "shape, dtype",
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
