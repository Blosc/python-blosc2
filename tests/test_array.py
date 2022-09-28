########################################################################
#
#       Author:  The Blosc development team - blosc@blosc.org
#
########################################################################


import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("size, dtype", [
    (1e6, "int64"),
    (1e6, "f8"),
    (1e6, "i1"),
])
def test_pack_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array(parray)
    assert np.array_equal(nparray, a2)

@pytest.mark.parametrize(
    "size, dtype", [
    (1e6, "int64"),
    (1e6, "float64"),
    (1e6, np.float64),
    (1e6, np.int8),
    pytest.param(3e8, "int64", marks=pytest.mark.heavy),  # > 2 GB
    ])
def test_pack_array2(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array2(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array2(parray)
    assert np.array_equal(nparray, a2)

@pytest.mark.parametrize(
    "size, dtype", [
        (100_000, "i4,i4"),
        (10_000, "i4,f8"),
        (3000, "i4,f4,S8")
    ])
def test_pack_array2_struct(size, dtype):
    nparray = np.fromiter(iter(range(size)), dtype="i4,f4,S8")
    parray = blosc2.pack_array2(nparray)
    assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array2(parray)
    assert np.array_equal(nparray, a2)

@pytest.mark.parametrize(
    "size, dtype, urlpath", [
    (1e6, "int64", "test.bl2"),
    ])
def test_save_array(size, dtype, urlpath):
    nparray = np.arange(size, dtype=dtype)
    serial_size = blosc2.save_array(nparray, urlpath, mode="w")
    assert serial_size < nparray.size * nparray.itemsize

    a2 = blosc2.load_array(urlpath)
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, a2)
