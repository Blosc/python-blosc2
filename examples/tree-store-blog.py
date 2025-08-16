#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import os

import numpy as np

import blosc2

# --- 1. Creating and populating a TreeStore ---
print("--- 1. Creating and populating a TreeStore ---")
# Create a new TreeStore in write mode ('w')
with blosc2.TreeStore("my_experiment.b2z", mode="w") as ts:
    # You can store numpy arrays, which are converted to blosc2.NDArray
    ts["/group1/dataset1"] = np.arange(100)

    # You can also store blosc2 arrays directly
    ts["/group1/dataset2"] = blosc2.full((5, 5), fill_value=3.14)

    # And external arrays with vlmeta attached (these are included internally too)
    ext = blosc2.zeros((10,), urlpath="external_array.b2nd", mode="w")
    ext.vlmeta["desc"] = "included array metadata"
    ts["/group1/included_array"] = ext

    # Create another group with a dataset
    ts["/group2/another_dataset"] = blosc2.zeros((10,))
print("Created 'my_experiment.b2z' with initial data.\n")


# --- 2. Reading from a TreeStore ---
print("--- 2. Reading from a TreeStore ---")
# Open the TreeStore in read mode ('r')
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    # Access a dataset
    dataset1 = ts["/group1/dataset1"]
    print("Dataset 1:", dataset1[:])  # Use [:] to decompress and get a NumPy array

    # Access the external array that has been included internally
    ext_array = ts["/group1/included_array"]
    print("Included array:", ext_array[:])
    print("Included array metadata:", ext_array.vlmeta[:])

    # List all paths in the store
    print("Paths in TreeStore:", list(ts))
print()


# --- 3. Storing Metadata with `vlmeta` ---
print("--- 3. Storing Metadata with `vlmeta` ---")
with blosc2.TreeStore("my_experiment.b2z", mode="a") as ts:  # 'a' for append/modify
    # Add metadata to the root
    ts.vlmeta["author"] = "The Blosc Team"
    ts.vlmeta["date"] = "2025-07-10"

    # Add metadata to a group
    ts["/group1"].vlmeta["description"] = "Data from the first run"

# Reading metadata
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    print("Root metadata:", ts.vlmeta[:])
    print("Group 1 metadata:", ts["/group1"].vlmeta[:])
print()


# --- 4. Working with Subtrees (Groups) ---
print("--- 4. Working with Subtrees (Groups) ---")
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    # Get the group as a subtree
    group1 = ts["/group1"]

    # Now you can access datasets relative to this group
    dataset2 = group1["dataset2"]
    print("Dataset 2 from group object:", dataset2[:])

    # You can also list contents relative to the group
    print("Contents of group1:", list(group1))
print()


# --- 5. Iterating Through a TreeStore ---
print("--- 5. Iterating Through a TreeStore ---")
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    for path, node in ts.items():
        if isinstance(node, blosc2.NDArray):
            print(f"Found dataset at '{path}' with shape {node.shape}")
        else:  # It's a group
            print(f"Found group at '{path}' with metadata: {node.vlmeta[:]}")
print()

# --- Cleanup ---
print("--- Cleanup ---")
os.remove("my_experiment.b2z")
print("Removed 'my_experiment.b2z'.")
