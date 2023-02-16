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


@pytest.mark.parametrize("shape, dtype",
                         [
                             ((100, 1230), "f4,f8"),
                             ((234, 125), "f4,(2,)f8", ),
                             ((80, 51, 60), [('f0', '<f4'), ('f1', '<f8')]),
                             ((400, 399, 401), [('field 1', '<f4'), ('mamà', '<f8')]),
                         ])
def test_scalar(shape, dtype):
    a = blosc2.zeros(shape, dtype=dtype)
    b = np.zeros(shape=shape, dtype=dtype)
    assert np.array_equal(a[:], b)

    dtype = np.dtype(dtype)
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.schunk.typesize == dtype.itemsize
