#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################


import os

import numpy as np
import pytest

import blosc2

##### pack / unpack  #####


@pytest.mark.parametrize(
    "size, dtype",
    [
        (1e6, "int64"),
        (1e6, "f8"),
        (1e6, "i1"),
    ],
)
def test_pack_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array(nparray)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array(parray)
    assert np.array_equal(nparray, a2)


@pytest.mark.parametrize(
    "size, dtype",
    [
        (1e6, "int64"),
        (1e6, "float64"),
        (1e6, np.float64),
        (1e6, np.int8),
        pytest.param(3e8, "int64", marks=pytest.mark.heavy),  # > 2 GB
    ],
)
def test_pack_array2(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_array2(nparray)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array2(parray)
    assert np.array_equal(nparray, a2)


@pytest.mark.parametrize("size, dtype", [(100_000, "i4,i4"), (10_000, "i4,f8"), (3000, "i4,f4,S8")])
def test_pack_array2_struct(size, dtype):
    nparray = np.fromiter(iter(range(size)), dtype="i4,f4,S8")
    parray = blosc2.pack_array2(nparray)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_array2(parray)
    assert np.array_equal(nparray, a2)


@pytest.mark.skipif(os.name == "nt", reason="Torch has issues with oldest-supported-numpy on Windows")
@pytest.mark.parametrize(
    "size, dtype",
    [
        (1e6, "float32"),
        (1e6, "float64"),
        (1e6, "int8"),
    ],
)
def test_pack_tensor_torch(size, dtype):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, dtype)
    tensor = torch.arange(size, dtype=dtype)
    cframe = blosc2.pack_tensor(tensor)
    atensor = np.asarray(tensor)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(cframe) < atensor.size * atensor.dtype.itemsize

    tensor2 = blosc2.unpack_tensor(cframe)
    assert np.array_equal(atensor, np.asarray(tensor2))


@pytest.mark.parametrize(
    "size, dtype",
    [
        (1e6, np.float32),
        (1e6, np.float64),
        (1e6, np.int8),
    ],
)
def test_pack_tensor_tensorflow(size, dtype):
    tensorflow = pytest.importorskip("tensorflow")
    array = np.arange(size, dtype=dtype)
    tensor = tensorflow.constant(array)
    cframe = blosc2.pack_tensor(tensor)
    atensor = np.asarray(tensor)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(cframe) < atensor.size * atensor.dtype.itemsize

    tensor2 = blosc2.unpack_tensor(cframe)
    assert np.array_equal(atensor, np.asarray(tensor2))


@pytest.mark.parametrize(
    "size, dtype",
    [
        (1e6, "int64"),
        (1e6, "float64"),
        (1e6, np.float64),
        (1e6, np.int8),
        pytest.param(3e8, "int64", marks=pytest.mark.heavy),  # > 2 GB
    ],
)
def test_pack_tensor_array(size, dtype):
    nparray = np.arange(size, dtype=dtype)
    parray = blosc2.pack_tensor(nparray)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert len(parray) < nparray.size * nparray.itemsize

    a2 = blosc2.unpack_tensor(parray)
    assert np.array_equal(nparray, a2)


##### save / load  #####


@pytest.mark.parametrize(
    "size, dtype, urlpath",
    [
        (1e6, "int64", "test.bl2"),
        (1e6, "float32", "test.bl2"),
    ],
)
def test_save_array(size, dtype, urlpath):
    nparray = np.arange(size, dtype=dtype)
    serial_size = blosc2.save_array(nparray, urlpath, mode="w")
    if not os.getenv("BTUNE_TRADEOFF"):
        assert serial_size < nparray.size * nparray.itemsize

    a2 = blosc2.load_array(urlpath)
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, a2)


@pytest.mark.parametrize(
    "size, dtype, urlpath",
    [
        (1e6, "int64", "test.bl2"),
        (1e6, "float32", "test.bl2"),
    ],
)
def test_save_tensor_array(size, dtype, urlpath):
    nparray = np.arange(size, dtype=dtype)
    serial_size = blosc2.save_tensor(nparray, urlpath, mode="w")
    if not os.getenv("BTUNE_TRADEOFF"):
        assert serial_size < nparray.size * nparray.itemsize

    a2 = blosc2.load_tensor(urlpath)
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, a2)


@pytest.mark.parametrize(
    "size, dtype, urlpath",
    [
        (1e6, "int64", "test.bl2"),
        (1e6, "float32", "test.bl2"),
    ],
)
def test_save_tensor_tensorflow(size, dtype, urlpath):
    tensorflow = pytest.importorskip("tensorflow")
    nparray = np.arange(size, dtype=dtype)
    tensor = tensorflow.constant(nparray)
    serial_size = blosc2.save_tensor(tensor, urlpath, mode="w")
    if not os.getenv("BTUNE_TRADEOFF"):
        assert serial_size < nparray.size * nparray.itemsize

    tensor2 = blosc2.load_tensor(urlpath)
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, np.asarray(tensor2))


@pytest.mark.skipif(os.name == "nt", reason="Torch has issues with oldest-supported-numpy on Windows")
@pytest.mark.parametrize(
    "size, dtype, urlpath",
    [
        (1e6, "int64", "test.bl2"),
        (1e6, "float32", "test.bl2"),
    ],
)
def test_save_tensor_torch(size, dtype, urlpath):
    torch = pytest.importorskip("torch")
    nparray = np.arange(size, dtype=dtype)
    tensor = torch.tensor(nparray)
    serial_size = blosc2.save_tensor(tensor, urlpath, mode="w")
    if not os.getenv("BTUNE_TRADEOFF"):
        assert serial_size < nparray.size * nparray.itemsize

    tensor2 = blosc2.load_tensor(urlpath)
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, np.asarray(tensor2))


@pytest.mark.parametrize(
    "size, sparse, urlpath",
    [
        (1e6, True, "test.bl2"),
        (1e6, False, "test.bl2"),
    ],
)
def test_save_tensor_sparse(size, sparse, urlpath):
    nparray = np.arange(size, dtype=np.int32)
    serial_size = blosc2.save_tensor(nparray, urlpath, mode="w", contiguous=not sparse)
    if not os.getenv("BTUNE_TRADEOFF"):
        assert serial_size < nparray.size * nparray.itemsize

    a2 = blosc2.load_tensor(urlpath)
    assert os.path.isdir(urlpath) == sparse
    blosc2.remove_urlpath(urlpath)
    assert np.array_equal(nparray, a2)
