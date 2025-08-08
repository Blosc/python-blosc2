#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
import shutil
import zipfile

import numpy as np
import pytest

import blosc2
from blosc2.dict_store import DictStore


@pytest.fixture(params=["b2d", "b2z"])
def populated_dict_store(request):
    """A fixture that creates and populates a DictStore.

    It is parametrized to use both zip (.b2z) and directory (.b2d)
    storage formats. It also handles cleanup of created files and
    directories.
    """
    storage_type = request.param
    path = f"test_dstore.{storage_type}"
    ext_path = "ext_node3.b2nd"

    # Setup: create and populate the store
    with DictStore(path, mode="w") as dstore:
        dstore["/node1"] = np.array([1, 2, 3])
        dstore["/node2"] = blosc2.ones(2)
        arr_external = blosc2.arange(3, urlpath=ext_path, mode="w")
        arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
        dstore["/dir1/node3"] = arr_external
        yield dstore, path

    # Teardown: clean up created files and directories
    if os.path.exists(ext_path):
        os.remove(ext_path)
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)


def test_basic_dstore(populated_dict_store):
    dstore, path = populated_dict_store
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
    with DictStore(path, mode="r") as dstore_read:
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


def test_external_value_set(populated_dict_store):
    dstore, _ = populated_dict_store
    node3 = dstore["/dir1/node3"]
    node3[:] = np.ones(3)
    assert np.all(node3[:] == np.ones(3))


def test_to_b2z_and_reopen(populated_dict_store):
    dstore, path = populated_dict_store
    dstore["/nodeA"] = np.arange(5)
    dstore["/nodeB"] = np.arange(6)
    dstore.close()

    with DictStore(path, mode="r") as dstore_read:
        assert "/nodeA" in dstore_read
        assert "/nodeB" in dstore_read
        assert np.all(dstore_read["/nodeA"][:] == np.arange(5))
        assert np.all(dstore_read["/nodeB"][:] == np.arange(6))


def test_map_tree_precedence(populated_dict_store):
    dstore, path = populated_dict_store
    # Create external file and add to dstore
    ext_path = "ext_nodeX.b2nd"
    arr_external = blosc2.arange(4, urlpath=ext_path, mode="w")
    dstore["/nodeX"] = np.arange(4)  # in embed store
    dstore["/externalX"] = arr_external  # in map_tree
    dstore.close()

    # Reopen and check map_tree precedence
    with DictStore(path, mode="r") as dstore_read:
        # Should prefer external file if key is in map_tree
        assert "/externalX" in dstore_read.map_tree
        arr = dstore_read["/externalX"]
        assert np.all(arr[:] == np.arange(4))
    if os.path.exists(ext_path):
        os.remove(ext_path)


def test_len_and_iter(populated_dict_store):
    dstore, path = populated_dict_store
    # The fixture already adds 3 nodes
    for i in range(3, 10):
        dstore[f"/node_{i}"] = np.full((5,), i)
    print("->", dstore.keys())
    dstore.close()

    with DictStore(path, mode="r") as dstore_read:
        keys = set(dstore_read)
        print(keys)
        assert len(dstore_read) == 10
        expected_keys = {"/node1", "/node2", "/dir1/node3"} | {f"/node_{i}" for i in range(3, 10)}
        assert keys == expected_keys


def test_without_embed(populated_dict_store):
    dstore, path = populated_dict_store
    # For this test, we want to start with a clean state
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

    # Create a DictStore without embed files
    with DictStore(path, mode="w") as dstore_new:
        ext_path = "ext_node3.b2nd"
        arr_external = blosc2.arange(3, urlpath=ext_path, mode="w")
        arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
        dstore_new["/dir1/node3"] = arr_external
        assert "/dir1/node3" in dstore_new.map_tree

    if path.endswith(".b2z"):
        with zipfile.ZipFile(path, "r") as zf:
            # Check that the external file is present
            assert "dir1/node3.b2nd" in zf.namelist()

    # Reopen and check vlmeta
    with DictStore(path, mode="r") as dstore_read:
        assert list(dstore_read.keys()) == ["/dir1/node3"]
        node3 = dstore_read["/dir1/node3"]
        assert node3.vlmeta["description"] == "This is vlmeta for /dir1/node3"
        # Check that the value is read-only
        with pytest.raises(ValueError):
            node3[:] = np.arange(5)
