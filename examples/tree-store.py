#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Example usage of TreeStore with hierarchical navigation, vlmeta, and CTables

from dataclasses import dataclass

import numpy as np

import blosc2

# Create a hierarchical store backed by a zip file
with blosc2.TreeStore("example_tree.b2z", mode="w") as tstore:
    # Create a small hierarchy
    tstore["/child0/data"] = np.array([1, 2, 3])
    tstore["/child0/child1/data"] = blosc2.ones(3)
    tstore["/child0/child2"] = blosc2.arange(3)

    # External arrays can also be included
    ext = blosc2.linspace(0, 1, 5, urlpath="external_leaf.b2nd", mode="w")
    ext.vlmeta["desc"] = "external /dir1/node3 metadata"  # NDArray-level metadata
    tstore["/dir1/node3"] = ext

    # Remote array (read-only), referenced via URLPath
    urlpath = blosc2.URLPath("@public/examples/ds-1d.b2nd", "https://cat2.cloud/demo")
    arr_remote = blosc2.open(urlpath, mode="r")
    tstore["/dir2/remote"] = arr_remote

    # TreeStore-level metadata (persists with the store)
    tstore.vlmeta["author"] = "blosc2"
    tstore.vlmeta["version"] = 1
    tstore.vlmeta[:] = {"purpose": "TreeStore example", "scale": 2.5}

    print("TreeStore keys:", sorted(tstore.keys()))
    print("/child0/data:", tstore["/child0/data"][:])
    print("/dir1/node3 (external) first 3:", tstore["/dir1/node3"][:3])
    print("/dir2/remote first 3:", tstore["/dir2/remote"][:3])
    print("Stored vlmeta:", tstore.vlmeta[:])
    node3 = tstore["/dir1/node3"]
    print("Node '/dir1/node3' vlmeta.desc:", node3.vlmeta["desc"])  # NDArray metadata

    # Access a subtree view rooted at /child0
    root = tstore["/child0"]  # or tstore["/child0"]
    print("Subtree '/child0' keys:", sorted(root.keys()))

    # Walk the subtree structure top-down
    print("Walk '/child0' subtree:")
    for path, children, nodes in root.walk("/"):
        print(f"  Path: {path}, children: {sorted(children)}, nodes: {sorted(nodes)}")

    # Query children and descendants from the full tree
    print("Children of '/':", tstore.get_children("/"))
    print("Descendants of '/child0':", tstore.get_descendants("/child0"))

    # Deleting a structural subtree removes all its leaves
    del tstore["/child0/child1"]
    print("After deleting '/child0/child1', keys:", sorted(tstore.keys()))

# Reopen and add another leaf under an existing subtree
with blosc2.open("example_tree.b2z", mode="a") as tstore2:
    tstore2["/child0/new_leaf"] = np.array([9, 9, 9])
    print("Reopened keys:", sorted(tstore2.keys()))
    # Read via subtree view
    rsub = tstore2["/child0"]
    print("/child0/new_leaf via subtree:", rsub["/new_leaf"][:])
    print(f"TreeStore file at: {tstore2.localpath}")

# ---------------------------------------------------------------------------
# Mixing NDArrays and CTables in the same TreeStore
# ---------------------------------------------------------------------------


@dataclass
class Reading:
    sensor_id: int = 0
    value: float = 0.0


with blosc2.TreeStore("example_tree.b2z", mode="a") as ts:
    # Create a small CTable in memory and store it inline
    t = blosc2.CTable(Reading)
    for i in range(5):
        t.append(Reading(sensor_id=i, value=float(i) * 1.1))

    # Assignment syntax is identical to NDArray
    ts["/readings"] = t
    print("Keys after adding CTable:", sorted(ts.keys()))

    # Object internals are hidden from normal traversal
    print("/readings/_meta in ts:", "/readings/_meta" in ts)  # False

with blosc2.open("example_tree.b2z", mode="r") as ts:
    # CTable is returned transparently; no special open call needed
    readings = ts["/readings"]
    print(f"CTable type: {type(readings).__name__}, rows: {len(readings)}")
    print("sensor_id column:", list(readings["sensor_id"][:]))
    print("value column    :", list(readings["value"][:]))

# Append a row to an inline CTable via append mode
with blosc2.TreeStore("example_tree.b2z", mode="a") as ts:
    readings = ts["/readings"]
    readings.append(Reading(sensor_id=99, value=-1.0))
    readings.close()  # explicit close before outer store repacks
    print("After append, rows:", len(ts["/readings"]))

# Delete the CTable object root (removes all internal leaves)
with blosc2.TreeStore("example_tree.b2z", mode="a") as ts:
    del ts["/readings"]
    print("After deleting /readings:", sorted(ts.keys()))
