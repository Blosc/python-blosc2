#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np

import blosc2

# Example usage
persistent = False
if persistent:
    tree = blosc2.Tree(urlpath="example_tree.b2t", mode="w")
else:
    tree = blosc2.Tree()
tree["/node1"] = np.array([1, 2, 3])
tree["/node2"] = blosc2.ones(2)
# Make /node3 an external file
arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
tree["/node3"] = arr_external
urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
arr_remote = blosc2.open(urlpath, mode="r")
tree["/node4"] = arr_remote

print("Tree keys:", list(tree.keys()))
print("Node1 data (embedded, numpy):", tree["/node1"][:])
print("Node2 data (embedded, blosc2):", tree["/node2"][:])
print("Node3 data (external):", tree["/node3"][:])
print("Node4 data (remote):", tree["/node3"][:])

del tree["/node1"]
print("After deletion, keys:", list(tree.keys()))

# Reading back the tree
if persistent:
    tree_read = blosc2.Tree(urlpath="example_tree.b2t", mode="a")
else:
    tree_read = blosc2.from_cframe(tree.to_cframe())

# Add another node to the tree
tree_read["/node5"] = np.array([4, 5, 6])
print("Node5 data:", tree_read["/node5"][:])

print("Read keys:", list(tree_read.keys()))
for key, value in tree_read.items():
    print(
        f"shape of {key}: {value.shape}, dtype: {value.dtype}, map: {tree_read._tree_map[key]}, "
        f"values: {value[:10] if len(value) > 3 else value[:]}"
    )
