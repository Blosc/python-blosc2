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
persistent = True
if persistent:
    estore = blosc2.EmbedStore(urlpath="example_estore.b2e", mode="w")
else:
    estore = blosc2.EmbedStore()
estore["/node1"] = np.array([1, 2, 3])
estore["/node2"] = blosc2.ones(2)
urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
arr_remote = blosc2.open(urlpath, mode="r")
estore["/dir1/node3"] = arr_remote
arr_external = blosc2.arange(3, urlpath="external_node3.b2nd", mode="w")
arr_external.vlmeta["description"] = "This is vlmeta for /dir1/node4"
estore["/dir2/node4"] = arr_external

print("EmbedStore keys:", list(estore.keys()))
print("Node1 data (embedded, numpy):", estore["/node1"][:])
print("Node2 data (embedded, blosc2):", estore["/node2"][:])
print("Node3 3 first row data (remote):", estore["/dir1/node3"][:3])

del estore["/node1"]
print("After deletion, keys:", list(estore.keys()))

# Reading back the tree
if persistent:
    estore_read = blosc2.EmbedStore(urlpath="example_estore.b2e", mode="a")
else:
    estore_read = blosc2.from_cframe(estore.to_cframe())

# Add another node to the tree
estore_read["/node5"] = np.array([4, 5, 6])
print("Node5 data:", estore_read["/node5"][:])

print("Read keys:", list(estore_read.keys()))
for key, value in estore_read.items():
    print(
        f"shape of {key}: {value.shape}, dtype: {value.dtype}, map: {estore_read._embed_map[key]}, "
        f"values: {value[:10] if len(value) > 3 else value[:]}"
    )

print(f"EmbedStore file at: {estore_read.urlpath}")
