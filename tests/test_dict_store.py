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
    with DictStore(path, mode="w", threshold=None) as dstore:
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
    with DictStore(path, mode="w", threshold=None) as dstore_new:
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


def test_store_and_retrieve_schunk_in_dict():
    # Create a small SChunk and store it in a DictStore (embedded)
    data = b"This is a tiny schunk"
    schunk = blosc2.SChunk(chunksize=None, data=data)
    vlmeta = "DictStore tiny schunk"
    schunk.vlmeta["description"] = vlmeta

    path = "test_dstore_schunk_embed.b2z"
    with DictStore(path, mode="w") as dstore:
        dstore["/schunk"] = schunk
        value = dstore["/schunk"]
        assert isinstance(value, blosc2.SChunk)
        assert value.nbytes == len(data)
        assert value[:] == data
        assert value.vlmeta["description"] == vlmeta
    if os.path.exists(path):
        os.remove(path)


essch_extern = "ext_schunk.b2f"


def test_external_schunk_file_and_reopen():
    # Ensure clean external file
    if os.path.exists(essch_extern):
        os.remove(essch_extern)

    # Create an external SChunk on disk with '.b2f'
    data = b"External schunk data"
    storage = blosc2.Storage(urlpath=essch_extern, mode="w")
    schunk_ext = blosc2.SChunk(chunksize=None, data=data, storage=storage)
    schunk_ext.vlmeta["description"] = "External SChunk"

    path = "test_dstore_schunk_external.b2z"
    with DictStore(path, mode="w", threshold=None) as dstore:
        # With threshold=None and external value, it should be stored as external file in map_tree
        dstore["/dir1/schunk_ext"] = schunk_ext
        assert "/dir1/schunk_ext" in dstore.map_tree
        # It should point to a .b2f file
        assert dstore.map_tree["/dir1/schunk_ext"].endswith(".b2f")

    # Zip should contain the .b2f
    with zipfile.ZipFile(path, "r") as zf:
        assert "dir1/schunk_ext.b2f" in zf.namelist()

    # Reopen and verify contents and type
    with DictStore(path, mode="r") as dstore_read:
        value = dstore_read["/dir1/schunk_ext"]
        assert isinstance(value, blosc2.SChunk)
        assert value[:] == data
        assert value.vlmeta["description"] == "External SChunk"

    # Cleanup
    if os.path.exists(essch_extern):
        os.remove(essch_extern)
    if os.path.exists(path):
        os.remove(path)


def _digest_value(value):
    """Return a bytes digest representative of a stored value content."""
    if isinstance(value, blosc2.SChunk):
        return bytes(value[:])
    # NDArray and potentially C2Array expose slicing to get numpy array
    arr = value[:]
    try:
        # numpy-like
        return np.ascontiguousarray(arr).tobytes()
    except Exception:
        # Fallback to bytes if possible
        return bytes(arr)


def test_values_union_and_precedence(tmp_path):
    # Build a store where a key exists both in embed and as external; external should take precedence in values()
    path = tmp_path / "test_values.dstore.b2z"
    ext_path = tmp_path / "dup_external.b2nd"
    with DictStore(str(path), mode="w", threshold=None) as dstore:
        # First, put an embedded value for /dup
        embed_arr = np.arange(3)
        dstore["/dup"] = embed_arr
        embed_digest = np.ascontiguousarray(embed_arr).tobytes()
        # Now, create an external array for the same key; map_tree should take precedence
        arr_external = blosc2.arange(5, urlpath=str(ext_path), mode="w")
        dstore["/dup"] = arr_external
        assert "/dup" in dstore.map_tree
    # Reopen read-only and verify
    with DictStore(str(path), mode="r") as dstore_read:
        # Collect digests from values()
        values_digests = {_digest_value(v) for v in dstore_read.values()}
        # The external content digest should be present, and the embedded one absent
        external_digest = (
            np.arange(5).astype(np.int64).tobytes()
            if np.arange(5).dtype != np.int64
            else np.arange(5).tobytes()
        )
        assert external_digest in values_digests
        assert embed_digest not in values_digests


def test_values_match_items_values(populated_dict_store):
    dstore, path = populated_dict_store
    # Add a couple of extra nodes
    dstore["/A"] = np.arange(4)
    dstore["/B"] = blosc2.ones(3)
    # Overwrite one with external to ensure mix
    ext_path = "A_ext.b2nd"
    arr_external = blosc2.arange(4, urlpath=ext_path, mode="w")
    dstore["/A"] = arr_external
    dstore.close()

    with DictStore(path, mode="r") as dstore_read:
        items_values = {_digest_value(v) for _, v in dstore_read.items()}
        values_values = {_digest_value(v) for v in dstore_read.values()}
        assert items_values == values_values

    if os.path.exists(ext_path):
        os.remove(ext_path)
