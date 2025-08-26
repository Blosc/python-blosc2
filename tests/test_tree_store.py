#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
import shutil

import numpy as np
import pytest

import blosc2
from blosc2.tree_store import TreeStore


@pytest.fixture(params=["b2d", "b2z"])
def populated_tree_store(request):
    """A fixture that creates and populates a TreeStore."""
    storage_type = request.param
    path = f"test_tstore.{storage_type}"
    ext_path = "ext_node3.b2nd"

    with TreeStore(path, mode="w", threshold=None) as tstore:
        tstore["/child0/data"] = np.array([1, 2, 3])
        tstore["/child0/child1/data"] = np.array([4, 5, 6])
        tstore["/child0/child2"] = np.array([7, 8, 9])
        tstore["/child0/child1/grandchild"] = np.array([10, 11, 12])
        tstore["/other"] = np.array([13, 14, 15])

        # Add external file
        arr_external = blosc2.arange(3, urlpath=ext_path, mode="w")
        arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
        tstore["/dir1/node3"] = arr_external

        yield tstore, path

    # Cleanup
    for file_path in [ext_path, path]:
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            else:
                shutil.rmtree(file_path)


def test_basic_tree_store(populated_tree_store):
    """Test basic TreeStore functionality."""
    tstore, _ = populated_tree_store

    # Test key existence - should include both leaf and structural nodes
    expected_keys = {
        "/child0/data",
        "/child0/child1/data",
        "/child0/child2",
        "/child0/child1/grandchild",
        "/other",
        "/dir1/node3",
        "/child0",
        "/child0/child1",
        "/dir1",
    }
    assert set(tstore.keys()) == expected_keys

    # Test data retrieval
    assert np.all(tstore["/child0/data"][:] == np.array([1, 2, 3]))
    assert np.all(tstore["/other"][:] == np.array([13, 14, 15]))

    # Test structural nodes return subtrees
    assert isinstance(tstore["/child0"], TreeStore)
    assert isinstance(tstore["/dir1"], TreeStore)

    # Test vlmeta
    node3 = tstore["/dir1/node3"]
    assert node3.vlmeta["description"] == "This is vlmeta for /dir1/node3"


def test_hierarchical_key_validation():
    """Test key validation for hierarchical structure."""
    with TreeStore("test_validation.b2z", mode="w") as tstore:
        # Valid keys
        tstore["/a"] = np.array([1])
        tstore["/b/c"] = np.array([2])
        tstore["/b/d/e"] = np.array([3])

        assert "/a" in tstore
        assert isinstance(tstore["/b"], TreeStore)

        # Invalid keys
        with pytest.raises(ValueError, match="Key cannot end with '/'"):
            tstore["/invalid/"] = np.array([1])
        with pytest.raises(ValueError, match="empty path segments"):
            tstore["/invalid//path"] = np.array([1])

    os.remove("test_validation.b2z")


def test_structural_path_assignment_prevention():
    """Test that assignment to structural paths is prevented."""
    with TreeStore("test_structural.b2z", mode="w") as tstore:
        tstore["/parent/data"] = np.array([1, 2, 3])
        tstore["/parent/child"] = np.array([4, 5, 6])

        # Cannot assign to structural path
        with pytest.raises(ValueError, match="Cannot assign array to structural path"):
            tstore["/parent"] = np.array([7, 8, 9])

        # Can create new paths
        tstore["/new_leaf"] = np.array([13, 14, 15])
        assert np.all(tstore["/new_leaf"][:] == np.array([13, 14, 15]))

    os.remove("test_structural.b2z")


def test_leaf_to_structural_prevention():
    """Test that adding children to existing leaf nodes is prevented."""
    with TreeStore("test_leaf_protection.b2z", mode="w") as tstore:
        tstore["/parent"] = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Cannot add child"):
            tstore["/parent/child"] = np.array([4, 5, 6])

        assert np.all(tstore["/parent"][:] == np.array([1, 2, 3]))

    os.remove("test_leaf_protection.b2z")


def test_tree_navigation(populated_tree_store):
    """Test tree navigation methods."""
    tstore, _ = populated_tree_store

    # Test get_children
    root_children = sorted(tstore.get_children("/"))
    expected = ["/child0", "/dir1", "/other"]
    assert root_children == expected

    # Test get_descendants
    root_descendants = sorted(tstore.get_descendants("/child0"))
    expected = [
        "/child0/child1",
        "/child0/child1/data",
        "/child0/child1/grandchild",
        "/child0/child2",
        "/child0/data",
    ]
    assert root_descendants == expected

    # Test walk
    walked_paths = [path for path, _, _ in tstore.walk("/")]
    assert "/" in walked_paths
    assert "/child0" in walked_paths


def test_subtree_functionality(populated_tree_store):
    """Test subtree view functionality."""
    tstore, _ = populated_tree_store

    # Get subtree
    root_subtree = tstore.get_subtree("/child0")
    expected_keys = {"/child1", "/child2", "/data", "/child1/data", "/child1/grandchild"}
    assert set(root_subtree.keys()) == expected_keys

    # Test data access through subtree
    assert np.all(root_subtree["/data"][:] == np.array([1, 2, 3]))

    # Test nested subtree
    child1_subtree = root_subtree.get_subtree("/child1")
    expected_nested = {"/data", "/grandchild"}
    assert set(child1_subtree.keys()) == expected_nested


def test_complex_operations():
    """Test complex operations with TreeStore."""
    with TreeStore("test_complex.b2z", mode="w") as tstore:
        # Create complex hierarchy
        paths = [
            "/level1/data",
            "/level1/level2a/data",
            "/level1/level2a/level3a",
            "/level1/level2b/data",
            "/separate_branch/data",
            "/separate_branch/sub1",
        ]

        for i, path in enumerate(paths):
            tstore[path] = np.array([i, i + 1, i + 2])

        # Test walk returns correct number of structural nodes
        walked_paths = [path for path, _, _ in tstore.walk("/")]
        assert len(walked_paths) >= 4  # At least /, /level1, /level1/level2a, /separate_branch

        # Test subtree access
        level2a_subtree = tstore.get_subtree("/level1/level2a")
        assert "/data" in level2a_subtree
        assert "/level3a" in level2a_subtree

        # Test deletion
        del tstore["/level1"]
        remaining_keys = {k for k in tstore if not k.startswith("/level1")}
        assert "/separate_branch/data" in remaining_keys

    os.remove("test_complex.b2z")


def test_getitem_returns_subtree_or_data():
    """Test that __getitem__ returns subtree for intermediate paths and data for leaves."""
    with TreeStore("test_getitem.b2z", mode="w") as tstore:
        # Create structure carefully to avoid structural path assignment
        tstore["/parent/data"] = np.array([1, 2, 3])  # Don't assign to /parent directly
        tstore["/parent/child"] = np.array([4, 5, 6])
        tstore["/leaf"] = np.array([7, 8, 9])

        # /parent has children, so should return a subtree
        parent_result = tstore["/parent"]
        assert isinstance(parent_result, TreeStore)
        assert set(parent_result.keys()) == {"/data", "/child"}

        # /leaf has no children, so should return data
        leaf_result = tstore["/leaf"]
        assert isinstance(leaf_result, blosc2.NDArray)
        assert np.all(leaf_result[:] == np.array([7, 8, 9]))

        # Access data through subtree
        parent_data = parent_result["/data"]
        assert isinstance(parent_data, blosc2.NDArray)
        assert np.all(parent_data[:] == np.array([1, 2, 3]))

    os.remove("test_getitem.b2z")


def test_delete_subtree():
    """Test deleting entire subtrees."""
    with TreeStore("test_delete.b2z", mode="w") as tstore:
        # Create structure without assigning to structural paths
        tstore["/parent/data"] = np.array([1, 2, 3])
        tstore["/parent/child1"] = np.array([4, 5, 6])
        tstore["/parent/child2"] = np.array([7, 8, 9])
        tstore["/other"] = np.array([13, 14, 15])

        # Delete the entire /parent subtree
        del tstore["/parent"]

        # Only /other should remain
        remaining_keys = set(tstore.keys())
        assert remaining_keys == {"/other"}

        # Verify /other data is still intact
        assert np.all(tstore["/other"][:] == np.array([13, 14, 15]))

    os.remove("test_delete.b2z")


def test_subtree_walk():  # noqa: C901
    """Test walking within a subtree."""
    with TreeStore("test_subtree_walk.b2z", mode="w") as tstore:
        # Create structure without assigning to structural paths
        tstore["/child0/data"] = np.array([1, 2, 3])
        tstore["/child0/branch1/data"] = np.array([4, 5, 6])
        tstore["/child0/branch1/leaf1"] = np.array([7, 8, 9])
        tstore["/child0/branch1/leaf2"] = np.array([10, 11, 12])
        tstore["/child0/leaf3"] = np.array([13, 14, 15])
        tstore["/child0/branch2/leaf4"] = np.array([113, 114, 115])
        tstore["/other"] = np.array([16, 17, 18])

        # Get subtree and walk it
        root_subtree = tstore.get_subtree("/child0")
        walked_results = list(root_subtree.walk("/"))

        # Should not include /other (outside the subtree)
        all_walked_nodes = []
        for _, _, nodes in walked_results:
            all_walked_nodes.extend(nodes)

        # Verify only nodes within /child0 subtree are visited
        # These should be names only, not full paths
        for node in all_walked_nodes:
            assert "/" not in node  # Should be names only, not paths
            assert node in ["data", "leaf1", "leaf2", "leaf3", "leaf4"]

        # Check values of the walked nodes
        for path, children, nodes in walked_results:
            if path == "/":
                assert sorted(children) == ["branch1", "branch2"]
                assert sorted(nodes) == ["data", "leaf3"]
            elif path == "/branch1":
                assert sorted(children) == []
                assert sorted(nodes) == ["data", "leaf1", "leaf2"]
            elif path == "/branch2":
                assert sorted(children) == []
                assert sorted(nodes) == ["leaf4"]
            # Build the path of nodes to check their values
            for node in nodes:
                full_path = f"{path}/{node}"
                if full_path == "/child0/data":
                    assert np.all(root_subtree[full_path][:] == np.array([1, 2, 3]))
                elif full_path == "/child0/branch1/data":
                    assert np.all(root_subtree[full_path][:] == np.array([4, 5, 6]))
                elif full_path == "/child0/branch1/leaf1":
                    assert np.all(root_subtree[full_path][:] == np.array([7, 8, 9]))
                elif full_path == "/child0/branch1/leaf2":
                    assert np.all(root_subtree[full_path][:] == np.array([10, 11, 12]))
                elif full_path == "/child0/leaf3":
                    assert np.all(root_subtree[full_path][:] == np.array([13, 14, 15]))
                elif full_path == "/child0/branch2/leaf4":
                    assert np.all(root_subtree[full_path][:] == np.array([113, 114, 115]))

    os.remove("test_subtree_walk.b2z")


def test_complex_hierarchy():
    """Test with a more complex hierarchical structure."""
    with TreeStore("test_complex.b2z", mode="w") as tstore:
        # Create a deep hierarchy (avoid assigning to structural paths)
        paths = [
            "/level1/data",
            "/level1/level2a/data",
            "/level1/level2a/level3a",
            "/level1/level2a/level3b",
            "/level1/level2b/data",
            "/level1/level2b/level3c/data",
            "/level1/level2b/level3c/level4",
            "/separate_branch/data",
            "/separate_branch/sub1",
            "/separate_branch/sub2",
        ]

        for i, path in enumerate(paths):
            tstore[path] = np.array([i, i + 1, i + 2])

        # Test deep walking - should visit all structural nodes
        walked_paths = []
        walked_results = []
        for path, children, nodes in tstore.walk("/"):
            walked_paths.append(path)
            walked_results.append((path, children, nodes))

        # Expected structural paths that should be visited:
        # "/", "/level1", "/level1/level2a", "/level1/level2b", "/level1/level2b/level3c", "/separate_branch"
        # That's 6 structural paths total
        assert len(walked_paths) == 6, f"Expected 6 paths, got {len(walked_paths)}: {walked_paths}"

        # Test that children and nodes are names, not full paths
        for path, children, nodes in walked_results:
            # All children should be simple names without "/"
            for child in children:
                assert "/" not in child, f"Child '{child}' in path '{path}' should be a name, not a path"
            # All nodes should be simple names without "/"
            for node in nodes:
                assert "/" not in node, f"Node '{node}' in path '{path}' should be a name, not a path"

        # Test deep subtree
        level2a_subtree = tstore.get_subtree("/level1/level2a")
        subtree_keys = set(level2a_subtree.keys())
        expected_keys = {"/data", "/level3a", "/level3b"}
        assert subtree_keys == expected_keys

        # Test very deep access
        level4_data = tstore["/level1/level2b/level3c/level4"]
        assert isinstance(level4_data, blosc2.NDArray)
        assert np.all(level4_data[:] == np.array([6, 7, 8]))

    os.remove("test_complex.b2z")


def test_mixed_leaf_and_structural_assignment():
    """Test creating both leaf nodes and structural nodes in correct order."""
    with TreeStore("test_mixed.b2z", mode="w") as tstore:
        # Create leaf nodes first
        tstore["/section2"] = np.array([4, 5, 6])

        # Create a hierarchical structure without conflicting with existing data
        # Instead of making /section1 both a leaf and structural, create separate paths
        tstore["/section1/data"] = np.array([1, 2, 3])  # Data goes to /section1/data
        tstore["/section1/child1"] = np.array([7, 8, 9])
        tstore["/section1/child2"] = np.array([10, 11, 12])

        # /section1 should return a subtree since it has children
        section1_subtree = tstore["/section1"]
        assert isinstance(section1_subtree, TreeStore)
        expected_section1_keys = {"/child1", "/child2", "/data"}
        assert set(section1_subtree.keys()) == expected_section1_keys

        # section2 should still return data (it's a leaf)
        section2_data = tstore["/section2"]
        assert isinstance(section2_data, blosc2.NDArray)
        assert np.all(section2_data[:] == np.array([4, 5, 6]))

        # Access section1's data through the subtree
        section1_data = section1_subtree["/data"]
        assert isinstance(section1_data, blosc2.NDArray)
        assert np.all(section1_data[:] == np.array([1, 2, 3]))

    os.remove("test_mixed.b2z")


def test_proper_leaf_vs_structural_creation():
    """Test the proper way to create mixed hierarchies without conflicts."""
    with TreeStore("test_proper_creation.b2z", mode="w") as tstore:
        # Method 1: Create all leaf nodes first, avoiding structural conflicts
        tstore["/data1"] = np.array([1, 2, 3])
        tstore["/data2"] = np.array([4, 5, 6])

        # Method 2: Create hierarchical structure where parent paths are purely structural
        tstore["/hierarchy/level1/data"] = np.array([7, 8, 9])
        tstore["/hierarchy/level1/subdata"] = np.array([10, 11, 12])
        tstore["/hierarchy/level2/data"] = np.array([13, 14, 15])

        # Verify structure
        assert isinstance(tstore["/data1"], blosc2.NDArray)  # Leaf
        assert isinstance(tstore["/data2"], blosc2.NDArray)  # Leaf
        assert isinstance(tstore["/hierarchy"], TreeStore)  # Structural
        assert isinstance(tstore["/hierarchy/level1"], TreeStore)  # Structural
        assert isinstance(tstore["/hierarchy/level1/data"], blosc2.NDArray)  # Leaf

    os.remove("test_proper_creation.b2z")


@pytest.mark.parametrize("storage_type", ["b2d", "b2z"])
def test_treestore_vlmeta_basic_and_bulk(storage_type):
    path = f"vlmeta_basic.{storage_type}"
    with TreeStore(path, mode="w") as tstore:
        # Basic set/get
        tstore.vlmeta["author"] = "blosc2"
        tstore.vlmeta["version"] = 1
        tstore.vlmeta["shape"] = (3, 2)
        assert tstore.vlmeta["author"] == "blosc2"
        assert tstore.vlmeta["version"] == 1
        assert tstore.vlmeta["shape"] == (3, 2)

        # Bulk set via [:] - should merge/update, not replace
        tstore.vlmeta[:] = {"desc": "test", "scale": 2.5}
        # Bulk get via [:]
        all_meta = tstore.vlmeta[:]
        assert all_meta["author"] == "blosc2"
        assert all_meta["version"] == 1
        assert all_meta["shape"] == (3, 2)
        assert all_meta["desc"] == "test"
        assert all_meta["scale"] == 2.5

        # Iteration and len should see all names
        names = sorted(iter(tstore.vlmeta))
        assert set(names) == set(all_meta.keys())
        assert len(tstore.vlmeta) == len(all_meta)

        # Deletion
        del tstore.vlmeta["desc"]
        assert "desc" not in set(iter(tstore.vlmeta))
        assert len(tstore.vlmeta) == len(all_meta) - 1

    # Reopen in read-only to check persistence and read-only protection
    with TreeStore(path, mode="r") as tstore:
        assert tstore.vlmeta["author"] == "blosc2"
        assert tstore.vlmeta["version"] == 1
        assert tstore.vlmeta["shape"] == (3, 2)
        assert "desc" not in set(iter(tstore.vlmeta))
        with pytest.raises(ValueError, match="read-only"):
            tstore.vlmeta["new"] = 123
        with pytest.raises(ValueError, match="read-only"):
            del tstore.vlmeta["author"]

    # Cleanup
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


@pytest.mark.parametrize("storage_type", ["b2d", "b2z"])
def test_treestore_vlmeta_does_not_interfere_with_data(storage_type):
    """Ensure vlmeta keys live in a separate namespace and do not collide with data keys."""
    path = f"vlmeta_isolation.{storage_type}"
    with TreeStore(path, mode="w") as tstore:
        # Put some data keys
        tstore["/group/data"] = np.array([1, 2, 3])
        tstore["/other"] = np.array([4, 5, 6])
        # Add metadata
        tstore.vlmeta["k1"] = {"a": 1}
        tstore.vlmeta["k2"] = [1, 2, 3]

        # Ensure data keys are unaffected
        assert "/group/data" in tstore
        assert "/other" in tstore
        assert np.all(tstore["/group/data"][:] == np.array([1, 2, 3]))
        assert np.all(tstore["/other"][:] == np.array([4, 5, 6]))

        # Ensure vlmeta iteration returns only metadata names (no slashes)
        for name in tstore.vlmeta:
            assert "/" not in name

    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


@pytest.mark.parametrize("storage_type", ["b2d", "b2z"])
def test_subtree_can_use_vlmeta(storage_type):
    """A subtree view should be able to read/write vlmeta independently."""
    path = f"vlmeta_subtree.{storage_type}"
    with TreeStore(path, mode="w") as tstore:
        # Create some structure and a subtree view
        tstore["/group/a"] = np.array([1])
        tstore["/group/b"] = np.array([2])
        subtree = tstore.get_subtree("/group")

        # Set metadata via subtree - should be independent from root
        subtree.vlmeta["note"] = "from_subtree"
        subtree.vlmeta["level"] = 5

        # Set metadata via root - should be independent from subtree
        tstore.vlmeta["rootmeta"] = 42

        # Verify independence - subtree vlmeta is separate from root vlmeta
        assert subtree.vlmeta["note"] == "from_subtree"
        assert subtree.vlmeta["level"] == 5
        assert "rootmeta" not in subtree.vlmeta
        assert "note" not in tstore.vlmeta
        assert "level" not in tstore.vlmeta
        assert tstore.vlmeta["rootmeta"] == 42

        # Bulk ops through subtree - should only affect subtree vlmeta
        subtree.vlmeta[:] = {"owner": "team", "scale": 1.5}
        all_meta_sub = subtree.vlmeta[:]
        expected_subtree_meta = {"note": "from_subtree", "level": 5, "owner": "team", "scale": 1.5}
        assert all_meta_sub == expected_subtree_meta

        # Root vlmeta should be unchanged
        assert tstore.vlmeta["rootmeta"] == 42
        assert "owner" not in tstore.vlmeta

        # Iteration from subtree should only show subtree metadata
        names = set(iter(subtree.vlmeta))
        expected_names = {"note", "level", "owner", "scale"}
        assert names == expected_names
        assert all("/" not in k for k in names)

        # Root vlmeta iteration should only show root metadata
        root_names = set(iter(tstore.vlmeta))
        assert root_names == {"rootmeta"}

        # Ensure data remains unaffected
        assert "/group/a" in tstore
        assert "/group/b" in tstore
        assert np.all(tstore["/group/a"][:] == np.array([1]))
        assert np.all(tstore["/group/b"][:] == np.array([2]))

    # Reopen in read-only and use subtree again
    with TreeStore(path, mode="r") as tstore_ro:
        subtree_ro = tstore_ro.get_subtree("/group")
        assert subtree_ro.vlmeta["note"] == "from_subtree"
        assert subtree_ro.vlmeta["owner"] == "team"
        assert tstore_ro.vlmeta["rootmeta"] == 42
        # Verify independence is maintained after reopening
        assert "rootmeta" not in subtree_ro.vlmeta
        assert "note" not in tstore_ro.vlmeta

        # Cannot modify via subtree in read-only
        with pytest.raises(ValueError, match="read-only"):
            subtree_ro.vlmeta["new"] = 1
        with pytest.raises(ValueError, match="read-only"):
            del subtree_ro.vlmeta["note"]

    # Cleanup
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path)


def test_schunk_support():
    """Test that TreeStore supports SChunk objects."""
    with TreeStore("test_schunk.b2z", mode="w") as tstore:
        # Create an SChunk
        data = b"This is a test SChunk with some data to compress and store."
        schunk = blosc2.SChunk(chunksize=200 * 1000, data=data)
        schunk.vlmeta["description"] = "Test SChunk for TreeStore"
        # Store SChunk in TreeStore
        tstore["/data/schunk1"] = schunk

        # Retrieve and verify
        retrieved_schunk = tstore["/data/schunk1"]
        assert isinstance(retrieved_schunk, blosc2.SChunk)
        assert len(retrieved_schunk) == len(schunk)
        assert retrieved_schunk.nchunks == schunk.nchunks
        assert retrieved_schunk.vlmeta["description"] == schunk.vlmeta["description"]
        assert retrieved_schunk[:] == data

        # Test structural behavior with SChunks
        tstore["/data/schunk2"] = blosc2.SChunk(chunksize=100 * 1000)

        # /data should return a subtree since it has children
        data_subtree = tstore["/data"]
        assert isinstance(data_subtree, TreeStore)
        expected_keys = {"/schunk1", "/schunk2"}
        assert set(data_subtree.keys()) == expected_keys

    os.remove("test_schunk.b2z")


def test_walk_topdown_argument_ordering():
    """Ensure walk supports topdown argument mimicking os.walk order semantics."""
    with TreeStore("test_walk_topdown.b2z", mode="w") as tstore:
        # Build a small hierarchy
        tstore["/a/x"] = np.array([1])
        tstore["/a/b/y"] = np.array([2])
        tstore["/c"] = np.array([3])

        top_paths = [p for p, _, _ in tstore.walk("/", topdown=True)]
        bot_paths = [p for p, _, _ in tstore.walk("/", topdown=False)]

        # Same paths visited, but different order
        assert set(top_paths) == set(bot_paths)
        assert top_paths[0] == "/"
        assert bot_paths[-1] == "/"  # root last in bottom-up

        # In topdown, parent before child; in bottom-up, child before parent
        assert top_paths.index("/a") < top_paths.index("/a/b")
        assert bot_paths.index("/a") > bot_paths.index("/a/b")

    os.remove("test_walk_topdown.b2z")


def test_walk_topdown_false_on_subtree():
    """Bottom-up walk should yield subtree root last."""
    with TreeStore("test_walk_subtree.b2z", mode="w") as tstore:
        tstore["/child0/child1/data"] = np.array([1])
        tstore["/child0/child2/data"] = np.array([2])
        tstore["/child0/data"] = np.array([3])
        sub = tstore.get_subtree("/child0")

        paths_bottom = [p for p, _, _ in sub.walk("/", topdown=False)]
        assert paths_bottom[-1] == "/"  # subtree root yielded last

        # Verify children and nodes contents are still names and consistent
        for _, children, nodes in sub.walk("/", topdown=False):
            for name in children + nodes:
                assert "/" not in name

    os.remove("test_walk_subtree.b2z")


def test_vlmeta_subtree_specific(populated_tree_store):
    """Test that each subtree has its own independent vlmeta."""
    tstore, tmpdir = populated_tree_store

    # Set vlmeta on root tree
    tstore.vlmeta["root_meta"] = "root_value"

    # Get subtree and set vlmeta on it
    subtree = tstore.get_subtree("/child0")
    subtree.vlmeta["subtree_meta"] = "subtree_value"

    # Get another subtree and set vlmeta on it
    subtree2 = tstore.get_subtree("/child0/child1")
    subtree2.vlmeta["nested_subtree_meta"] = "nested_value"

    # Verify that vlmeta are independent
    assert tstore.vlmeta["root_meta"] == "root_value"
    assert "subtree_meta" not in tstore.vlmeta
    assert "nested_subtree_meta" not in tstore.vlmeta

    assert subtree.vlmeta["subtree_meta"] == "subtree_value"
    assert "root_meta" not in subtree.vlmeta
    assert "nested_subtree_meta" not in subtree.vlmeta

    assert subtree2.vlmeta["nested_subtree_meta"] == "nested_value"
    assert "root_meta" not in subtree2.vlmeta
    assert "subtree_meta" not in subtree2.vlmeta


def test_vlmeta_persistence_subtrees(tmp_path):
    """Test that subtree vlmeta persists across store reopening."""
    store_path = tmp_path / "test_vlmeta_subtrees.b2z"

    # Create store and add data with vlmeta
    with TreeStore(str(store_path), mode="w") as tstore:
        tstore["/child0/data"] = np.array([1, 2, 3])
        tstore["/child1/data"] = np.array([4, 5, 6])

        # Set root vlmeta
        tstore.vlmeta["root_info"] = "root_data"

        # Set subtree vlmeta
        subtree0 = tstore.get_subtree("/child0")
        subtree0.vlmeta["child0_info"] = "child0_data"

        subtree1 = tstore.get_subtree("/child1")
        subtree1.vlmeta["child1_info"] = "child1_data"

    # Reopen and verify vlmeta persisted
    with TreeStore(str(store_path), mode="r") as tstore:
        assert tstore.vlmeta["root_info"] == "root_data"

        subtree0 = tstore.get_subtree("/child0")
        assert subtree0.vlmeta["child0_info"] == "child0_data"

        subtree1 = tstore.get_subtree("/child1")
        assert subtree1.vlmeta["child1_info"] == "child1_data"

        # Verify independence
        assert "child0_info" not in tstore.vlmeta
        assert "child1_info" not in tstore.vlmeta
        assert "root_info" not in subtree0.vlmeta
        assert "root_info" not in subtree1.vlmeta


def test_vlmeta_bulk_operations_subtrees(populated_tree_store):
    """Test bulk vlmeta operations on subtrees."""
    tstore, tmpdir = populated_tree_store

    # Set up vlmeta on root and subtree
    tstore.vlmeta["key1"] = "value1"
    tstore.vlmeta["key2"] = "value2"

    subtree = tstore.get_subtree("/child0")
    subtree.vlmeta["sub_key1"] = "sub_value1"
    subtree.vlmeta["sub_key2"] = "sub_value2"

    # Test bulk get
    root_bulk = tstore.vlmeta[:]
    subtree_bulk = subtree.vlmeta[:]

    assert root_bulk == {"key1": "value1", "key2": "value2"}
    assert subtree_bulk == {"sub_key1": "sub_value1", "sub_key2": "sub_value2"}

    # Test bulk set - should merge/update, not replace
    new_root_meta = {"new_key1": "new_value1", "new_key2": "new_value2"}
    new_subtree_meta = {"new_sub_key1": "new_sub_value1"}

    tstore.vlmeta[:] = new_root_meta
    subtree.vlmeta[:] = new_subtree_meta

    # Verify bulk set merged with existing data
    expected_root = {"key1": "value1", "key2": "value2", "new_key1": "new_value1", "new_key2": "new_value2"}
    expected_subtree = {"sub_key1": "sub_value1", "sub_key2": "sub_value2", "new_sub_key1": "new_sub_value1"}

    assert tstore.vlmeta[:] == expected_root
    assert subtree.vlmeta[:] == expected_subtree

    # Verify old keys are still there (merged behavior)
    assert "key1" in tstore.vlmeta
    assert "sub_key1" in subtree.vlmeta


def test_vlmeta_read_only_subtrees(tmp_path):
    """Test vlmeta read-only behavior in subtrees."""
    store_path = tmp_path / "test_vlmeta_readonly_subtrees.b2z"

    # Create store with vlmeta
    with TreeStore(str(store_path), mode="w") as tstore:
        tstore["/child0/data"] = np.array([1, 2, 3])
        tstore.vlmeta["root_key"] = "root_value"

        subtree = tstore.get_subtree("/child0")
        subtree.vlmeta["subtree_key"] = "subtree_value"

    # Open read-only and test
    with TreeStore(str(store_path), mode="r") as tstore:
        # Should be able to read
        assert tstore.vlmeta["root_key"] == "root_value"

        subtree = tstore.get_subtree("/child0")
        assert subtree.vlmeta["subtree_key"] == "subtree_value"

        # Should not be able to write
        with pytest.raises(ValueError, match="read-only mode"):
            tstore.vlmeta["new_key"] = "new_value"

        with pytest.raises(ValueError, match="read-only mode"):
            subtree.vlmeta["new_sub_key"] = "new_sub_value"

        with pytest.raises(ValueError, match="read-only mode"):
            del tstore.vlmeta["root_key"]

        with pytest.raises(ValueError, match="read-only mode"):
            del subtree.vlmeta["subtree_key"]


def test_vlmeta_subtree_read_write():
    """Test that vlmeta added to a subtree can be read correctly."""
    with TreeStore("test_vlmeta_subtree_rw.b2z", mode="w") as tstore:
        # Create a hierarchical structure
        tstore["/department/team1/project_a"] = np.array([1, 2, 3])
        tstore["/department/team1/project_b"] = np.array([4, 5, 6])
        tstore["/department/team2/project_c"] = np.array([7, 8, 9])

        # Add vlmeta to the root
        tstore.vlmeta["organization"] = "Blosc Development Team"
        tstore.vlmeta["year"] = 2025

        # Get subtree and add vlmeta to it
        dept_subtree = tstore.get_subtree("/department")
        dept_subtree.vlmeta["manager"] = "John Doe"
        dept_subtree.vlmeta["budget"] = 100000
        dept_subtree.vlmeta["projects"] = ["project_a", "project_b", "project_c"]

        # Get nested subtree and add vlmeta
        team1_subtree = tstore.get_subtree("/department/team1")
        team1_subtree.vlmeta["lead"] = "Alice Smith"
        team1_subtree.vlmeta["members"] = 5
        team1_subtree.vlmeta["active_projects"] = 2

        # Test reading vlmeta from different levels
        # Root level
        assert tstore.vlmeta["organization"] == "Blosc Development Team"
        assert tstore.vlmeta["year"] == 2025
        assert len(tstore.vlmeta) == 2

        # Department level
        assert dept_subtree.vlmeta["manager"] == "John Doe"
        assert dept_subtree.vlmeta["budget"] == 100000
        assert dept_subtree.vlmeta["projects"] == ["project_a", "project_b", "project_c"]
        assert len(dept_subtree.vlmeta) == 3

        # Team1 level
        assert team1_subtree.vlmeta["lead"] == "Alice Smith"
        assert team1_subtree.vlmeta["members"] == 5
        assert team1_subtree.vlmeta["active_projects"] == 2
        assert len(team1_subtree.vlmeta) == 3

        # Verify independence - each level should only see its own vlmeta
        assert "manager" not in tstore.vlmeta
        assert "lead" not in tstore.vlmeta
        assert "organization" not in dept_subtree.vlmeta
        assert "lead" not in dept_subtree.vlmeta
        assert "organization" not in team1_subtree.vlmeta
        assert "manager" not in team1_subtree.vlmeta

        # Test bulk read operations
        root_all = tstore.vlmeta[:]
        dept_all = dept_subtree.vlmeta[:]
        team1_all = team1_subtree.vlmeta[:]

        assert root_all == {"organization": "Blosc Development Team", "year": 2025}
        assert dept_all == {
            "manager": "John Doe",
            "budget": 100000,
            "projects": ["project_a", "project_b", "project_c"],
        }
        assert team1_all == {"lead": "Alice Smith", "members": 5, "active_projects": 2}

        # Test iteration
        root_keys = set(tstore.vlmeta.keys())
        dept_keys = set(dept_subtree.vlmeta.keys())
        team1_keys = set(team1_subtree.vlmeta.keys())

        assert root_keys == {"organization", "year"}
        assert dept_keys == {"manager", "budget", "projects"}
        assert team1_keys == {"lead", "members", "active_projects"}

        # Verify data integrity is maintained
        assert np.array_equal(tstore["/department/team1/project_a"][:], np.array([1, 2, 3]))
        assert np.array_equal(team1_subtree["/project_a"][:], np.array([1, 2, 3]))

    # Test persistence by reopening
    with TreeStore("test_vlmeta_subtree_rw.b2z", mode="r") as tstore:
        # Re-verify all vlmeta after reopening
        assert tstore.vlmeta["organization"] == "Blosc Development Team"
        assert tstore.vlmeta["year"] == 2025

        dept_subtree = tstore.get_subtree("/department")
        assert dept_subtree.vlmeta["manager"] == "John Doe"
        assert dept_subtree.vlmeta["budget"] == 100000

        team1_subtree = tstore.get_subtree("/department/team1")
        assert team1_subtree.vlmeta["lead"] == "Alice Smith"
        assert team1_subtree.vlmeta["members"] == 5

        # Verify independence is maintained after reopening
        assert "manager" not in tstore.vlmeta
        assert "organization" not in dept_subtree.vlmeta
        assert "organization" not in team1_subtree.vlmeta

    # Cleanup
    os.remove("test_vlmeta_subtree_rw.b2z")


def test_key_normalization():
    """Test that keys without leading '/' are automatically normalized."""
    with TreeStore("test_key_normalization.b2z", mode="w") as tstore:
        # Test assignment without leading '/'
        tstore["data1"] = np.array([1, 2, 3])
        tstore["group/data2"] = np.array([4, 5, 6])
        tstore["group/subgroup/data3"] = np.array([7, 8, 9])

        # Keys should be normalized internally
        assert "/data1" in tstore
        assert "/group/data2" in tstore
        assert "/group/subgroup/data3" in tstore

        # Access with and without leading '/' should work
        assert np.array_equal(tstore["data1"][:], np.array([1, 2, 3]))
        assert np.array_equal(tstore["/data1"][:], np.array([1, 2, 3]))
        assert np.array_equal(tstore["group/data2"][:], np.array([4, 5, 6]))
        assert np.array_equal(tstore["/group/data2"][:], np.array([4, 5, 6]))

        # Structural access should also work
        group_subtree = tstore["group"]
        assert isinstance(group_subtree, TreeStore)
        assert "/data2" in group_subtree
        assert "/subgroup/data3" in group_subtree

        # Test other methods work with non-leading '/' keys
        children = tstore.get_children("group")
        assert "/group/subgroup" in children

        descendants = tstore.get_descendants("group")
        assert "/group/data2" in descendants
        assert "/group/subgroup/data3" in descendants

        # Test contains with both formats
        assert "data1" in tstore
        assert "/data1" in tstore
        assert "group/data2" in tstore
        assert "/group/data2" in tstore

    os.remove("test_key_normalization.b2z")
