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


@pytest.fixture
def cleanup_files():
    files = [
        "test_estore.b2e",
        "external_node3.b2nd",
    ]
    yield
    for f in files:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture
def populate_nodes(cleanup_files):
    estore = blosc2.EmbedStore(urlpath="test_estore.b2e", mode="w")
    estore["/node1"] = np.array([1, 2, 3])
    arr_embedded = blosc2.arange(3, dtype=np.int32)
    arr_embedded.vlmeta["description"] = "This is vlmeta for /node2"
    estore["/node2"] = arr_embedded
    arr_embedded = blosc2.arange(4, dtype=np.int32, urlpath="external_node3.b2nd", mode="w")
    arr_embedded.vlmeta["description"] = "This is vlmeta for /node3"
    estore["/node3"] = arr_embedded

    return estore


def test_basic(populate_nodes):
    estore = populate_nodes

    assert set(estore.keys()) == {"/node1", "/node2", "/node3"}
    assert np.all(estore["/node1"][:] == np.array([1, 2, 3]))
    assert np.all(estore["/node2"][:] == np.arange(3))
    assert np.all(estore["/node3"][:] == np.arange(4))

    del estore["/node1"]
    assert "/node1" not in estore

    estore_read = blosc2.EmbedStore(urlpath="test_estore.b2e", mode="r")
    assert set(estore_read.keys()) == {"/node2", "/node3"}
    for value in estore_read.values():
        assert hasattr(value, "shape")
        assert hasattr(value, "dtype")


def test_with_remote(populate_nodes):
    estore = populate_nodes

    # Re-open the estore to add a remote node
    estore = blosc2.EmbedStore(urlpath="test_estore.b2e")
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo/")
    arr_remote = blosc2.open(urlpath, mode="r")
    estore["/node4"] = arr_remote

    estore_read = blosc2.EmbedStore(urlpath="test_estore.b2e", mode="r")
    assert set(estore_read.keys()) == {"/node1", "/node2", "/node3", "/node4"}
    for key, value in estore_read.items():
        assert hasattr(value, "shape")
        assert hasattr(value, "dtype")
        if key == "/node4":
            assert hasattr(value, "urlbase")
            assert value.urlbase == urlpath.urlbase
            assert value.path == urlpath.path


def test_with_compression():
    # Create a estore with compressed data
    estore = blosc2.EmbedStore(cparams=blosc2.CParams(codec=blosc2.Codec.BLOSCLZ))
    arr = np.arange(1000, dtype=np.int32)
    estore["/compressed_node"] = arr

    # Read the estore and check the compressed node
    estore_read = blosc2.from_cframe(estore.to_cframe())
    assert set(estore_read.keys()) == {"/compressed_node"}
    assert np.all(estore_read["/compressed_node"][:] == arr)
    value = estore_read["/compressed_node"]
    assert value.cparams.codec == blosc2.Codec.BLOSCLZ


def test_with_many_nodes():
    # Create a estore with many nodes
    N = 200
    estore = blosc2.EmbedStore(urlpath="test_estore.b2e", mode="w")
    for i in range(N):
        estore[f"/node_{i}"] = blosc2.full(
            shape=(10,),
            fill_value=i,
            dtype=np.int32,
        )

    # Read the estore and check the nodes
    estore_read = blosc2.EmbedStore(urlpath="test_estore.b2e", mode="r")
    assert len(estore_read) == N
    for i in range(N):
        assert np.all(estore_read[f"/node_{i}"][:] == np.full((10,), i, dtype=np.int32))


def test_vlmeta_get(populate_nodes):
    estore = populate_nodes
    # Check that vlmeta is present for the nodes
    node2 = estore["/node2"]
    assert "description" in node2.vlmeta
    assert node2.vlmeta["description"] == "This is vlmeta for /node2"
    node3 = estore["/node3"]
    assert "description" in node3.vlmeta
    assert node3.vlmeta["description"] == "This is vlmeta for /node3"
    print(f"node3 type: {type(node3)}")
    print(f"estore['/node3'] type: {type(estore['/node3'])}")
    print(f"Same object? {node3 is estore['/node3']}")
    assert node3.vlmeta["description"] == "This is vlmeta for /node3"
    # TODO: this assertion style is failing, investigate why
    # assert estore["/node3"].vlmeta["description"] == "This is vlmeta for /node3"


# TODO
def _test_embedded_value_set_raise(populate_nodes):
    estore = populate_nodes

    # This should raise an error because value is read-only for embedded nodes
    node2 = estore["/node2"]
    node2[:] = np.arange(5)


# TODO: this should raise an error because vlmeta is read-only for embedded nodes
def _test_vlmeta_set(populate_nodes):
    estore = populate_nodes

    node2 = estore["/node2"]
    node2.vlmeta["description"] = "This is node 2 modified"
    assert node2.vlmeta["description"] == "This is node 2 modified"


# TODO
def _test_vlmeta_set_raise(with_external_nodes):
    estore = with_external_nodes

    # This should raise an error because vlmeta is read-only for embedded nodes
    node2 = estore["/node2"]
    with pytest.raises(AttributeError):
        node2.vlmeta["description"] = "This is node 2 modified"


def test_to_cframe(populate_nodes):
    estore = populate_nodes

    # Convert estore to a cframe
    cframe_data = estore.to_cframe()

    # Check the type and content of the cframe data
    assert isinstance(cframe_data, bytes)
    assert len(cframe_data) > 0

    # Deserialize back
    deserialized_estore = blosc2.from_cframe(cframe_data)
    assert np.all(deserialized_estore["/node2"][:] == np.arange(3))


def test_to_cframe_append(populate_nodes):
    estore = populate_nodes

    # Convert estore to a cframe
    cframe_data = estore.to_cframe()

    # Deserialize back
    new_estore = blosc2.from_cframe(cframe_data)

    # Add a new node to the deserialized estore
    new_estore["/node4"] = np.arange(3)
    assert np.all(new_estore["/node4"][:] == np.arange(3))
    new_estore["/node5"] = np.arange(4, 7)
    assert np.all(new_estore["/node5"][:] == np.arange(4, 7))


def test_store_and_retrieve_schunk():
    # Create a small SChunk and store it in an in-memory EmbedStore
    data = b"This is a small schunk"
    schunk = blosc2.SChunk(chunksize=None, data=data)
    vlmeta = "This is a small schunk for testing"
    schunk.vlmeta["description"] = vlmeta

    estore = blosc2.EmbedStore()
    estore["/schunk"] = schunk

    # Retrieve it back and check type and contents
    value = estore["/schunk"]
    assert isinstance(value, blosc2.SChunk)
    assert value.nbytes == len(data)
    assert value[:] == data
    assert value.vlmeta["description"] == vlmeta
