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
from blosc2.dict_store import DictStore


@pytest.fixture
def cleanup_dstore():
    files = [
        "test_dstore.b2z",
        "ext_node3.b2nd",
    ]
    yield
    for f in files:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture
def with_dstore(cleanup_dstore):
    with DictStore("test_dstore.b2z", mode="w") as dstore:
        dstore["/node1"] = np.array([1, 2, 3])
        dstore["/node2"] = blosc2.ones(2)
        arr_external = blosc2.arange(3, urlpath="ext_node3.b2nd", mode="w")
        arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
        dstore["/dir1/node3"] = arr_external
        yield dstore


def test_basic_dstore(with_dstore):
    dstore = with_dstore
    assert set(dstore.keys()) == {"/node1", "/node2", "/dir1/node3"}
    assert np.all(dstore["/node1"][:] == np.array([1, 2, 3]))
    assert np.all(dstore["/node2"][:] == np.ones(2))
    assert np.all(dstore["/dir1/node3"][:] == np.arange(3))
    # The next is insecure, as vlmeta can be reclaimed by garbage collection
    # assert dstore["/dir1/node3"].vlmeta["description"] == "This is vlmeta for /dir1/node3"
    # This is safe, as we keep a reference to the node
    node3 = dstore["/dir1/node3"]
    assert node3.vlmeta["description"] == "This is vlmeta for /dir1/node3"

    del dstore["/node1"]
    assert "/node1" not in dstore

    # Persist and reopen
    dstore.close()
    with DictStore("test_dstore.b2z", mode="r") as dstore_read:
        keys = set(dstore_read.keys())
        assert "/node2" in keys
        assert "/dir1/node3" in keys
        # for key, value in dstore_read.items():
        for _, value in dstore_read.items():
            assert hasattr(value, "shape")
            assert hasattr(value, "dtype")
            # TODO
            # if key == "/dir1/node3":
            #     node3 = dstore["/dir1/node3"]
            #     assert node3.vlmeta["description"] == "This is vlmeta for /dir1/node3"


def test_external_value_set(with_dstore):
    dstore = with_dstore
    node3 = dstore["/dir1/node3"]
    node3[:] = np.ones(3)
    assert np.all(node3[:] == np.ones(3))


def test_to_b2z_and_reopen(cleanup_dstore):
    with DictStore("test_dstore.b2z", mode="w") as dstore:
        dstore["/nodeA"] = np.arange(5)
        dstore["/nodeB"] = np.arange(6)

    with DictStore("test_dstore.b2z", mode="r") as dstore_read:
        assert set(dstore_read.keys()) == {"/nodeA", "/nodeB"}
        assert np.all(dstore_read["/nodeA"][:] == np.arange(5))
        assert np.all(dstore_read["/nodeB"][:] == np.arange(6))


def test_map_tree_precedence(cleanup_dstore):
    # Create external file and add to dstore
    arr_external = blosc2.arange(4, urlpath="ext_nodeX.b2nd", mode="w")
    with DictStore("test_dstore.b2z", mode="w") as dstore:
        dstore["/nodeX"] = np.arange(4)
        dstore["/externalX"] = arr_external

    # Reopen and check map_tree precedence
    with DictStore("test_dstore.b2z", mode="r") as dstore_read:
        # Should prefer external file if key is in map_tree
        assert "/externalX" in dstore_read.map_tree
        arr = dstore_read["/externalX"]
        assert np.all(arr[:] == np.arange(4))


def test_len_and_iter(cleanup_dstore):
    with DictStore("test_dstore.b2z", mode="w") as dstore:
        for i in range(10):
            dstore[f"/node_{i}"] = np.full((5,), i)

    with DictStore("test_dstore.b2z", mode="r") as dstore_read:
        assert len(dstore_read) == 10
        keys = set(dstore_read)
        assert keys == {f"/node_{i}" for i in range(10)}
        for key in keys:
            arr = dstore_read[key]
            i = int(key.split("_")[-1])
            assert np.all(arr[:] == np.full((5,), i))
