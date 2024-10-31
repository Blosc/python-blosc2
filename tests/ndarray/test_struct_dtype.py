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
    "shape, dtype, urlpath",
    [
        ((100, 123), "f4,f8", None),
        ((234, 125), "f4,(2,)f8", "test1.b2nd"),
        (80, [("f0", "<f4"), ("f1", "<f8")], "test2.b2nd"),
        ((40,), [("field 1", "<f4"), ("mamà", "<f8")], None),
        ((40,), [("field 1", "<f4"), ("mamà", "<f8")], "test3.b2nd"),
    ],
)
def test_scalar(shape, dtype, urlpath):
    blosc2.remove_urlpath(urlpath)

    a = blosc2.zeros(shape, dtype=dtype, urlpath=urlpath)
    b = np.zeros(shape=shape, dtype=dtype)
    assert np.array_equal(a[:], b)

    dtype = np.dtype(dtype)
    assert shape in (a.shape, a.shape[0])
    assert a.dtype == dtype
    assert a.schunk.typesize == dtype.itemsize
    assert a.shape == b.shape
    assert a.dtype == b.dtype

    if urlpath is not None:
        c = blosc2.open(urlpath)
        assert np.array_equal(c[:], b)
        assert c.shape == a.shape
        assert c.dtype == a.dtype
        assert c.schunk.typesize == dtype.itemsize
        assert c.shape == a.shape
        assert c.dtype == a.dtype

    blosc2.remove_urlpath(urlpath)
