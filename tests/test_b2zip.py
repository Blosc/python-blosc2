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
from blosc2.b2zip import ZipStore


@pytest.fixture
def cleanup_zipstore():
    files = [
        "test_zipstore.b2z",
        "ext_node3.b2nd",
    ]
    yield
    for f in files:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture
def with_zipstore(cleanup_zipstore):
    with ZipStore("test_zipstore.b2z", mode="w") as zipstore:
        zipstore["/node1"] = np.array([1, 2, 3])
        zipstore["/node2"] = blosc2.ones(2)
        arr_external = blosc2.arange(3, urlpath="ext_node3.b2nd", mode="w")
        zipstore["/dir1/node3"] = arr_external
        yield zipstore


def test_basic_zipstore(with_zipstore):
    zipstore = with_zipstore
    assert set(zipstore.keys()) == {"/node1", "/node2", "/dir1/node3"}
    assert np.all(zipstore["/node1"][:] == np.array([1, 2, 3]))
    assert np.all(zipstore["/node2"][:] == np.ones(2))
    assert np.all(zipstore["/dir1/node3"][:] == np.arange(3))
    # TODO
    # assert zipstore["/dir1/node3"].urlpath == "ext_node3.b2nd"

    del zipstore["/node1"]
    assert "/node1" not in zipstore

    # Persist and reopen
    zipstore.close()
    with ZipStore("test_zipstore.b2z", mode="r") as zipstore_read:
        keys = set(zipstore_read.keys())
        assert "/node2" in keys
        assert "/dir1/node3" in keys
        # for key, value in zipstore_read.items():
        for _, value in zipstore_read.items():
            assert hasattr(value, "shape")
            assert hasattr(value, "dtype")
            # TODO
            # if key == "/dir1/node3":
            #     assert value.urlpath == "ext_node3.b2nd"


def test_external_value_set(with_zipstore):
    zipstore = with_zipstore
    node3 = zipstore["/dir1/node3"]
    node3[:] = np.ones(3)
    assert np.all(node3[:] == np.ones(3))


def test_to_b2z_and_reopen(cleanup_zipstore):
    with ZipStore("test_zipstore.b2z", mode="w") as zipstore:
        zipstore["/nodeA"] = np.arange(5)
        zipstore["/nodeB"] = np.arange(6)

    with ZipStore("test_zipstore.b2z", mode="r") as zipstore_read:
        assert set(zipstore_read.keys()) == {"/nodeA", "/nodeB"}
        assert np.all(zipstore_read["/nodeA"][:] == np.arange(5))
        assert np.all(zipstore_read["/nodeB"][:] == np.arange(6))


def test_map_tree_precedence(cleanup_zipstore):
    # Create external file and add to zipstore
    arr_external = blosc2.arange(4, urlpath="ext_nodeX.b2nd", mode="w")
    with ZipStore("test_zipstore.b2z", mode="w") as zipstore:
        zipstore["/nodeX"] = np.arange(4)
        zipstore["/externalX"] = arr_external

    # Reopen and check map_tree precedence
    with ZipStore("test_zipstore.b2z", mode="r") as zipstore_read:
        # Should prefer external file if key is in map_tree
        assert "/externalX" in zipstore_read.map_tree
        arr = zipstore_read["/externalX"]
        assert np.all(arr[:] == np.arange(4))


# TODO
def _test_len_and_iter(cleanup_zipstore):
    with ZipStore("test_zipstore.b2z", mode="w") as zipstore:
        for i in range(10):
            zipstore[f"/node_{i}"] = np.full((5,), i)

    with ZipStore("test_zipstore.b2z", mode="r") as zipstore_read:
        assert len(zipstore_read) == 10
        keys = set(zipstore_read)
        assert keys == {f"/node_{i}" for i in range(10)}
        for key in keys:
            arr = zipstore_read[key]
            i = int(key.split("_")[-1])
            print("i:", i)
            assert np.all(arr[:] == np.full((5,), i))
