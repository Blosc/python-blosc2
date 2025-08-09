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
with blosc2.DictStore("example_dstore.b2z", mode="w") as tree:
    tree["/node1"] = np.array([1, 2, 3])
    tree["/node2"] = blosc2.ones(2)
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    arr_remote = blosc2.open(urlpath, mode="r")
    tree["/dir1/node3"] = arr_remote
    arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
    arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
    tree["/dir2/node4"] = arr_external

    print("DictStore keys:", list(tree.keys()))
    print("Node1 data (embedded, numpy):", tree["/node1"][:])
    print("Node2 data (embedded, blosc2):", tree["/node2"][:])
    print("Node3 3 first row data (remote):", tree["/dir1/node3"][:3])
    print("Node4 3 first row data (external):", tree["/dir2/node4"][:3])

    del tree["/node1"]
    print("After deletion, keys:", list(tree.keys()))

# Reading back the tree
with blosc2.DictStore("example_dstore.b2z", mode="a") as tree2:
    # Add another node to the tree
    tree2["/dir2/node5"] = np.array([4, 5, 6])
    print("Node5 data:", tree2["/dir2/node5"][:])

    print("Read keys:", list(tree2.keys()))
    for key, value in tree2.items():
        print(
            f"shape of {key}: {value.shape}, dtype: {value.dtype} "
            f"values: {value[:10] if len(value) > 3 else value[:]}"
        )

print(f"DictStore file at: {tree2.localpath}")
