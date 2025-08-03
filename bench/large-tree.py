#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
import os
import time
import numpy as np
import blosc2
from blosc2.b2tree import Tree
from memory_profiler import memory_usage

def make_arrays(n, min_size, max_size, dtype="i4"):
    sizes = np.linspace(min_size, max_size, n).astype(int)
    arrays = [blosc2.arange(size, dtype=dtype) for size in sizes]
    # arrays = [np.random.randint(0, 100, size=size, dtype=dtype) for size in sizes]
    # Calculate uncompressed size
    uncompressed_size = sum(arr.nbytes for arr in arrays)
    print(f"Uncompressed data size: {uncompressed_size / 1e9:.2f} GB")
    return arrays, sizes, uncompressed_size

def get_file_size(filepath):
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / 1e6
    return 0

def run_inner_tree(arrays, sizes, tree_path, uncompressed_size):
    def inner_process():
        tree = Tree(urlpath=tree_path, mode="w")
        for i, arr in enumerate(arrays):
            tree[f"/node{i}"] = arr
        return tree

    t0 = time.time()
    mem_usage = memory_usage((inner_process, ()), interval=0.1)
    t1 = time.time()
    peak_mem = max(mem_usage) - min(mem_usage)
    file_size = get_file_size(tree_path)
    compression_ratio = uncompressed_size / (file_size * 1e6) if file_size > 0 else 0
    print(f"[Inner] Time: {t1-t0:.2f}s, Memory: {peak_mem:.2f} MB, File size: {file_size:.2f} MB, Compression: {compression_ratio:.1f}x")
    return t1-t0, peak_mem, file_size

def run_local_tree(arrays, sizes, tree_path, arr_prefix, uncompressed_size):
    def local_process():
        tree = Tree(urlpath=tree_path, mode="w")
        for i, arr in enumerate(arrays):
            arr_path = f"{arr_prefix}_node{i}.b2nd"
            arr_b2 = blosc2.asarray(arr, urlpath=arr_path, mode="w")
            tree[f"/node{i}"] = arr_b2
        return tree

    t0 = time.time()
    mem_usage = memory_usage((local_process, ()), interval=0.1)
    t1 = time.time()
    peak_mem = max(mem_usage) - min(mem_usage)
    file_size = get_file_size(tree_path)
    total_external_size = sum(get_file_size(f"{arr_prefix}_node{i}.b2nd") for i in range(len(arrays)))
    total_size_mb = (file_size + total_external_size)
    compression_ratio = uncompressed_size / (total_size_mb * 1e6) if total_size_mb > 0 else 0
    print(f"[Local] Time: {t1-t0:.2f}s, Memory: {peak_mem:.2f} MB, Tree file size: {file_size:.2f} MB, External files size: {total_external_size:.2f} MB, Total: {total_size_mb:.2f} MB, Compression: {compression_ratio:.1f}x")
    return t1-t0, peak_mem, file_size, total_external_size

def cleanup_files(tree_path, arr_prefix, n):
    if os.path.exists(tree_path):
        os.remove(tree_path)
    for i in range(n):
        arr_path = f"{arr_prefix}_node{i}.b2nd"
        if os.path.exists(arr_path):
            os.remove(arr_path)

if __name__ == "__main__":
    N = 100
    min_size = int(1e6)   # 1 MB
    max_size = int(1e8)   # 100 MB
    arrays, sizes, uncompressed_size = make_arrays(N, min_size, max_size)

    print("Benchmarking Tree with inner arrays...")
    tree_path_inner = "large_inner_tree.b2z"
    t_inner, mem_inner, file_size_inner = run_inner_tree(arrays, sizes, tree_path_inner, uncompressed_size)

    print("Benchmarking Tree with local arrays...")
    tree_path_local = "large_local_tree.b2z"
    arr_prefix = "large_local"
    t_local, mem_local, file_size_local, external_size = run_local_tree(arrays, sizes, tree_path_local, arr_prefix, uncompressed_size)

    print("\nSummary:")
    print(f"Inner arrays:   Time = {t_inner:.2f}s, Memory = {mem_inner:.2f} MB, File size = {file_size_inner:.2f} MB")
    print(f"Local arrays:   Time = {t_local:.2f}s, Memory = {mem_local:.2f} MB, Tree file size = {file_size_local:.2f} MB, External files size = {external_size:.2f} MB")

    speedup = t_inner / t_local if t_local > 0 else float('inf')
    mem_ratio = mem_inner / mem_local if mem_local > 0 else float('inf')
    file_ratio = file_size_inner / file_size_local if file_size_local > 0 else float('inf')
    storage_ratio = file_size_inner / (file_size_local + external_size)
    print(f"Time ratio (inner/local): {speedup:.2f}x")
    print(f"Memory ratio (inner/local): {mem_ratio:.2f}x")
    print(f"File size ratio (inner/local tree): {file_ratio:.2f}x")
    print(f"Storage efficiency (inner vs total local): {storage_ratio:.2f}x")

    cleanup_files(tree_path_inner, arr_prefix, N)
    cleanup_files(tree_path_local, arr_prefix, N)
