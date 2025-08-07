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
    tree = blosc2.EmbedStore(urlpath="example_tree.b2t", mode="w")
else:
    tree = blosc2.EmbedStore()
tree["/node1"] = np.array([1, 2, 3])
tree["/node2"] = blosc2.ones(2)
urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
arr_remote = blosc2.open(urlpath, mode="r")
tree["/dir1/node3"] = arr_remote

print("EmbedStore keys:", list(tree.keys()))
print("Node1 data (embedded, numpy):", tree["/node1"][:])
print("Node2 data (embedded, blosc2):", tree["/node2"][:])
print("Node3 3 first row data (remote):", tree["/dir1/node3"][:3])

del tree["/node1"]
print("After deletion, keys:", list(tree.keys()))

# Reading back the tree
if persistent:
    tree_read = blosc2.EmbedStore(urlpath="example_tree.b2t", mode="a")
else:
    tree_read = blosc2.from_cframe(tree.to_cframe())

# Add another node to the tree
tree_read["/node4"] = np.array([4, 5, 6])
print("Node4 data:", tree_read["/node4"][:])

print("Read keys:", list(tree_read.keys()))
for key, value in tree_read.items():
    print(
        f"shape of {key}: {value.shape}, dtype: {value.dtype}, map: {tree_read._embed_map[key]}, "
        f"values: {value[:10] if len(value) > 3 else value[:]}"
    )
