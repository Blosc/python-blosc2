import blosc2
import pytest
import numpy as np

@pytest.mark.parametrize("size, dtype",
                         [
                             (1e6, None)
                         ])

def test_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array(parray)
    assert np.array_equal(nparray, a2)
