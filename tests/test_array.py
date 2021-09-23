########################################################################
#
#       Created: April 30, 2021
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("size, dtype", [(1e6, "int64")])
def test_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array(parray)
    assert np.array_equal(nparray, a2)
