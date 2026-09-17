#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os

import numpy as np

import blosc2

# --- 1. Creating and populating a TreeStore ---
print("--- 1. Creating and populating a TreeStore ---")
# Create a new TreeStore
with blosc2.TreeStore("my_experiment.b2z", mode="w") as ts:
    # You can store numpy arrays, which are converted to blosc2.NDArray
    ts["/dataset0"] = np.arange(100)

    # Create a group with a dataset that can be a blosc2 NDArray
    ts["/group1/dataset1"] = blosc2.zeros((10,))

    # You can also store blosc2 arrays directly (attrs included)
    ext = blosc2.linspace(0, 1, 10_000, dtype=np.float32)
    ext.attrs["desc"] = "dataset2 metadata"
    ts["/group1/dataset2"] = ext
print("Created 'my_experiment.b2z' with initial data.\n")


# --- 2. Reading from a TreeStore ---
print("--- 2. Reading from a TreeStore ---")
# Open the TreeStore in read-only mode ('r')
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    # Access a dataset
    dataset1 = ts["/group1/dataset1"]
    print("Dataset 1:", dataset1[:])  # Use [:] to decompress and get a NumPy array

    # Access the external array that has been stored internally
    dataset2 = ts["/group1/dataset2"]
    print("Dataset 2", dataset2[:])
    print("Dataset 2 metadata:", dataset2.attrs[:])

    # List all paths in the store
    print("Paths in TreeStore:", list(ts))
print()


# --- 3. Storing Metadata with `attrs` ---
print("--- 3. Storing Metadata with `attrs` ---")
with blosc2.TreeStore("my_experiment.b2z", mode="a") as ts:  # 'a' for append/modify
    # Add metadata to the root
    ts.attrs["author"] = "The Blosc Team"
    ts.attrs["date"] = "2025-08-17"

    # Add metadata to a group
    ts["/group1"].attrs["description"] = "Data from the first run"

# Reading metadata
with blosc2.TreeStore("my_experiment.b2z", mode="r") as ts:
    print("Root metadata:", ts.attrs[:])
    print("Group 1 metadata:", ts["/group1"].attrs[:])
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
            print(f"Found group at '{path}' with metadata: {node.attrs[:]}")
print()

# --- Cleanup ---
print("--- Cleanup ---")
os.remove("my_experiment.b2z")
print("Removed 'my_experiment.b2z'.")
