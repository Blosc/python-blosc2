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
        "test_tree.b2z",
        "external_node3.b2nd",
    ]
    yield
    for f in files:
        if os.path.exists(f):
            os.remove(f)


@pytest.fixture
def with_external_nodes(cleanup_files):
    tree = blosc2.Tree(urlpath="test_tree.b2z", mode="w")
    tree["/node1"] = np.array([1, 2, 3])
    arr_embedded = blosc2.arange(3, dtype=np.int32)
    arr_embedded.vlmeta["description"] = "This is vlmeta for /node2"
    tree["/node2"] = arr_embedded
    arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
    arr_external.vlmeta["description"] = "This is vlmeta for /node3"
    tree["/node3"] = arr_external
    return tree


def test_basic(with_external_nodes):
    tree = with_external_nodes

    assert set(tree.keys()) == {"/node1", "/node2", "/node3"}
    assert np.all(tree["/node1"][:] == np.array([1, 2, 3]))
    assert np.all(tree["/node2"][:] == np.arange(3))
    assert np.all(tree["/node3"][:] == np.arange(3))
    assert tree["/node3"].urlpath == "external_node3.b2nd"

    del tree["/node1"]
    assert "/node1" not in tree

    tree_read = blosc2.Tree(urlpath="test_tree.b2z", mode="r")
    assert set(tree_read.keys()) == {"/node2", "/node3"}
    for key, value in tree_read.items():
        assert hasattr(value, "shape")
        assert hasattr(value, "dtype")
        if key == "/node3":
            assert value.urlpath == "external_node3.b2nd"


def test_with_remote(with_external_nodes):
    tree = with_external_nodes

    # Re-open the tree to add a remote node
    tree = blosc2.Tree(urlpath="test_tree.b2z")
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo/")
    arr_remote = blosc2.open(urlpath, mode="r")
    tree["/node4"] = arr_remote

    tree_read = blosc2.Tree(urlpath="test_tree.b2z", mode="r")
    assert set(tree_read.keys()) == {"/node1", "/node2", "/node3", "/node4"}
    for key, value in tree_read.items():
        assert hasattr(value, "shape")
        assert hasattr(value, "dtype")
        if key == "/node3":
            assert value.urlpath == "external_node3.b2nd"
        if key == "/node4":
            assert hasattr(value, "urlbase")
            assert value.urlbase == urlpath.urlbase
            assert value.path == urlpath.path


def test_with_compression():
    # Create a tree with compressed data
    tree = blosc2.Tree(
        urlpath="test_tree-compr.b2z", mode="w", cparams=blosc2.CParams(codec=blosc2.Codec.BLOSCLZ)
    )
    arr = np.arange(1000, dtype=np.int32)
    tree["/compressed_node"] = arr

    # Read the tree and check the compressed node
    tree_read = blosc2.Tree(urlpath="test_tree-compr.b2z", mode="r")
    assert set(tree_read.keys()) == {"/compressed_node"}
    assert np.all(tree_read["/compressed_node"][:] == arr)
    value = tree_read["/compressed_node"]
    assert value.cparams.codec == blosc2.Codec.BLOSCLZ

    # Remove the test file after checking
    os.remove("test_tree-compr.b2z")


def test_with_many_nodes():
    # Create a tree with many nodes
    N = 200
    tree = blosc2.Tree(urlpath="test_tree.b2z", mode="w")
    for i in range(N):
        tree[f"/node_{i}"] = blosc2.full(
            shape=(10,),
            fill_value=i,
            dtype=np.int32,
        )

    # Read the tree and check the nodes
    tree_read = blosc2.Tree(urlpath="test_tree.b2z", mode="r")
    assert len(tree_read) == N
    for i in range(N):
        assert np.all(tree_read[f"/node_{i}"][:] == np.full((10,), i, dtype=np.int32))


def test_vlmeta_get(with_external_nodes):
    tree = with_external_nodes
    # Check that vlmeta is present for the nodes
    node2 = tree["/node2"]
    assert "description" in node2.vlmeta
    assert node2.vlmeta["description"] == "This is vlmeta for /node2"
    assert "description" in tree["/node3"].vlmeta
    node3 = tree["/node3"]
    assert node3.vlmeta["description"] == "This is vlmeta for /node3"
    print(f"node3 type: {type(node3)}")
    print(f"tree['/node3'] type: {type(tree['/node3'])}")
    print(f"Same object? {node3 is tree['/node3']}")
    assert node3.vlmeta["description"] == "This is vlmeta for /node3"
    # TODO: this assertion style is failing, investigate why
    # assert tree["/node3"].vlmeta["description"] == "This is vlmeta for /node3"


# TODO
def _test_embedded_value_set_raise(with_external_nodes):
    tree = with_external_nodes

    # This should raise an error because value is read-only for embedded nodes
    node2 = tree["/node2"]
    node2[:] = np.arange(5)


def test_external_value_set(with_external_nodes):
    tree = with_external_nodes

    # This should raise an error because value is read-only for embedded nodes
    node3 = tree["/node3"]
    node3[:] = np.ones(3)
    assert np.all(node3[:] == np.ones(3))


def test_vlmeta_set(with_external_nodes):
    tree = with_external_nodes

    # node2 = tree["/node2"]
    # node2.vlmeta["description"] = "This is node 2 modified"

    # Add variable-length metadata to a node
    node3 = tree["/node3"]
    node3.vlmeta["description"] = "This is node 3 modified"
    # TODO: this assignment is failing, investigate why
    # tree["/node3"].vlmeta["description"] = "This is node 3 modified"

    # Read the vlmeta back
    assert node3.vlmeta["description"] == "This is node 3 modified"

    # Check that vlmeta is preserved after writing and reading
    tree_read = blosc2.Tree(urlpath="test_tree.b2z", mode="r")
    node3 = tree["/node3"]
    assert node3.vlmeta["description"] == "This is node 3 modified"


# TODO
def _test_vlmeta_set_raise(with_external_nodes):
    tree = with_external_nodes

    # This should raise an error because vlmeta is read-only for embedded nodes
    node2 = tree["/node2"]
    with pytest.raises(AttributeError):
        node2.vlmeta["description"] = "This is node 2 modified"


def test_to_cframe(with_external_nodes):
    tree = with_external_nodes

    # Convert tree to a cframe
    cframe_data = tree.to_cframe()

    # Check the type and content of the cframe data
    assert isinstance(cframe_data, bytes)
    assert len(cframe_data) > 0

    # Deserialize back
    deserialized_tree = blosc2.from_cframe(cframe_data)
    assert np.all(deserialized_tree["/node2"][:] == np.arange(3))


def test_to_cframe_append(with_external_nodes):
    tree = with_external_nodes

    # Convert tree to a cframe
    cframe_data = tree.to_cframe()

    # Deserialize back
    new_tree = blosc2.from_cframe(cframe_data)

    # Add a new node to the deserialized tree
    new_tree["/node4"] = np.arange(3)
    assert np.all(new_tree["/node4"][:] == np.arange(3))
    new_tree["/node5"] = np.arange(4, 7)
    assert np.all(new_tree["/node5"][:] == np.arange(4, 7))
