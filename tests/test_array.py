########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("size, dtype", [(1e6, "int64")])
def test_pack_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array(parray)
    assert np.array_equal(nparray, a2)

@pytest.mark.parametrize(
    "size, dtype", [
    (1e6, "int64"),
    pytest.param(4e8, "int64", marks=pytest.mark.heavy),
    pytest.param(4e8 + 10, "int64", marks=pytest.mark.heavy),
    pytest.param(4e8 - 10, "int64", marks=pytest.mark.heavy),
    ])
def test_pack_array2(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array2(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array2(parray)
    assert np.array_equal(nparray, a2)
