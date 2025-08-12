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
with blosc2.DictStore("example_dstore.b2z", mode="w") as dstore:
    dstore["/node1"] = np.array([1, 2, 3])
    dstore["/node2"] = blosc2.ones(2)
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    arr_remote = blosc2.open(urlpath, mode="r")
    dstore["/dir1/node3"] = arr_remote
    arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
    arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node3"
    dstore["/dir2/node4"] = arr_external

    print("DictStore keys:", list(dstore.keys()))
    print("Node1 data (embedded, numpy):", dstore["/node1"][:])
    print("Node2 data (embedded, blosc2):", dstore["/node2"][:])
    print("Node3 3 first row data (remote):", dstore["/dir1/node3"][:3])
    print("Node4 3 first row data (external):", dstore["/dir2/node4"][:3])

    del dstore["/node1"]
    print("After deletion, keys:", list(dstore.keys()))

# Reading back the dstore
with blosc2.DictStore("example_dstore.b2z", mode="a") as dstore2:
    # Add another node to the dstore
    dstore2["/dir2/node5"] = np.array([4, 5, 6])
    print("Node5 data:", dstore2["/dir2/node5"][:])

    print("Read keys:", list(dstore2.keys()))
    for key, value in dstore2.items():
        print(
            f"shape of {key}: {value.shape}, dtype: {value.dtype} "
            f"values: {value[:10] if len(value) > 3 else value[:]}"
        )

print(f"DictStore file at: {dstore2.localpath}")
