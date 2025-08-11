#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Example usage of TreeStore with hierarchical navigation and vlmeta

import numpy as np

import blosc2

# Create a hierarchical store backed by a zip file
with blosc2.TreeStore("example_tree.b2z", mode="w") as tstore:
    # Create a small hierarchy. Data must be stored at leaves.
    tstore["/root/data"] = np.array([1, 2, 3])  # embedded (numpy)
    tstore["/root/child1/data"] = blosc2.ones(3)  # embedded (NDArray)
    tstore["/root/child2"] = blosc2.arange(3)  # embedded (NDArray)

    # External array stored as separate .b2nd file
    ext = blosc2.linspace(0, 1, 5, urlpath="external_leaf.b2nd", mode="w")
    ext.vlmeta["desc"] = "external /dir1/node3 metadata"  # NDArray-level metadata
    tstore["/dir1/node3"] = ext

    # # Remote array (read-only), referenced via URLPath
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    arr_remote = blosc2.open(urlpath, mode="r")
    tstore["/dir2/remote"] = arr_remote

    # TreeStore-level metadata (persists with the store)
    tstore.vlmeta["author"] = "blosc2"
    tstore.vlmeta["version"] = 1
    tstore.vlmeta[:] = {"purpose": "TreeStore example", "scale": 2.5}

    print("TreeStore keys:", sorted(tstore.keys()))
    print("/root/data:", tstore["/root/data"][:])
    print("/dir1/node3 (external) first 3:", tstore["/dir1/node3"][:3])
    print("/dir2/remote first 3:", tstore["/dir2/remote"][:3])
    print("Stored vlmeta:", tstore.vlmeta[:])
    node3 = tstore["/dir1/node3"]
    print("Node '/dir1/node3' vlmeta.desc:", node3.vlmeta["desc"])  # NDArray metadata

    # Access a subtree view rooted at /root
    root = tstore["/root"]  # or tstore["/root"]
    print("Subtree '/root' keys:", sorted(root.keys()))

    # Walk the subtree structure top-down
    print("Walk '/root' subtree:")
    for path, children, nodes in root.walk("/"):
        print(f"  Path: {path}, children: {sorted(children)}, nodes: {sorted(nodes)}")

    # Query children and descendants from the full tree
    print("Children of '/':", tstore.get_children("/"))
    print("Descendants of '/root':", tstore.get_descendants("/root"))

    # Deleting a structural subtree removes all its leaves
    del tstore["/root/child1"]
    print("After deleting '/root/child1', keys:", sorted(tstore.keys()))

# Reopen and add another leaf under an existing subtree
with blosc2.TreeStore("example_tree.b2z", mode="a") as tstore2:
    tstore2["/root/new_leaf"] = np.array([9, 9, 9])
    print("Reopened keys:", sorted(tstore2.keys()))
    # Read via subtree view
    rsub = tstore2.get_subtree("/root")
    print("/root/new_leaf via subtree:", rsub["/new_leaf"][:])
    print(f"TreeStore file at: {tstore2.localpath}")
