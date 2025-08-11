#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""
Benchmark for TreeStore with large arrays.

This benchmark creates N numpy arrays with sizes following a normal distribution
and measures the time and memory consumption for storing them in a TreeStore.
"""

import os
import shutil
import time

from memory_profiler import profile, memory_usage
import numpy as np

import blosc2

# Configuration
N_ARRAYS = 100  # Number of arrays to store
NGROUPS_MAX = 10
PEAK_SIZE_MB = 1  # Peak size in MB for the normal distribution
STDDEV_MB = 2  # Standard deviation in MB
OUTPUT_DIR = "large-tree-store.b2d"
MIN_SIZE_MB = 0.1  # Minimum array size in MB
MAX_SIZE_MB = 32  # Maximum array size in MB


def generate_array_sizes(n_arrays, peak_mb, stddev_mb, min_mb, max_mb):
    """Generate array sizes following a normal distribution."""
    # Generate sizes in MB using normal distribution
    sizes_mb = np.random.normal(peak_mb, stddev_mb, n_arrays)

    # Clip to reasonable bounds
    sizes_mb = np.clip(sizes_mb, min_mb, max_mb)

    # Convert to number of elements (assuming float64 = 8 bytes per element)
    sizes_elements = (sizes_mb * 1024 * 1024 / 8).astype(int)

    return sizes_mb, sizes_elements


def create_test_arrays(sizes_elements):
    """Create test arrays using numpy.linspace."""
    arrays = []
    print(f"Creating {len(sizes_elements)} test arrays...")

    for i, size in enumerate(sizes_elements):
        # Create linearly spaced array from 0 to i
        arr = np.linspace(0, i, size, dtype=np.float64)
        arrays.append(arr)

        if (i + 1) % 10 == 0:
            print(f"  Created {i + 1}/{len(sizes_elements)} arrays")

    return arrays


@profile
def store_arrays_in_treestore(arrays, output_dir):
    """Store arrays in TreeStore and measure performance."""
    print(f"\nStoring {len(arrays)} arrays in TreeStore at {output_dir}...")

    # Clean up existing directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    start_time = time.time()

    with blosc2.TreeStore(output_dir, mode="w") as tstore:
        for i, arr in enumerate(arrays):
            # Distribute arrays evenly across NGROUPS_MAX subdirectories
            group_id = i % NGROUPS_MAX
            key = f"/group_{group_id:02d}/array_{i:04d}"
            tstore[key] = arr

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Stored {i + 1}/{len(arrays)} arrays ({elapsed:.2f}s)")

        # Add some metadata
        tstore.vlmeta["n_arrays"] = len(arrays)
        tstore.vlmeta["peak_size_mb"] = PEAK_SIZE_MB
        tstore.vlmeta["benchmark_timestamp"] = time.time()
        tstore.vlmeta["n_groups"] = NGROUPS_MAX

    end_time = time.time()
    total_time = end_time - start_time

    return total_time


def measure_memory_usage(func, *args, **kwargs):
    """Measure memory usage of a function."""
    print("\nMeasuring memory usage...")

    def wrapper():
        return func(*args, **kwargs)

    # Measure memory usage during execution
    mem_usage = memory_usage(wrapper, interval=0.1, timeout=None)

    max_memory_mb = max(mem_usage)
    min_memory_mb = min(mem_usage)
    memory_increase_mb = max_memory_mb - min_memory_mb

    return max_memory_mb, min_memory_mb, memory_increase_mb, mem_usage


def print_statistics(sizes_mb, sizes_elements, total_time, memory_stats):
    """Print benchmark statistics."""
    max_mem, min_mem, mem_increase, _ = memory_stats

    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)

    print(f"Configuration:")
    print(f"  Number of arrays: {N_ARRAYS}")
    print(f"  Peak size: {PEAK_SIZE_MB} MB")
    print(f"  Standard deviation: {STDDEV_MB} MB")
    print(f"  Output directory: {OUTPUT_DIR}")

    print(f"\nArray size statistics:")
    print(f"  Mean size: {np.mean(sizes_mb):.2f} MB")
    print(f"  Median size: {np.median(sizes_mb):.2f} MB")
    print(f"  Min size: {np.min(sizes_mb):.2f} MB")
    print(f"  Max size: {np.max(sizes_mb):.2f} MB")
    print(f"  Total data: {np.sum(sizes_mb):.2f} MB")

    print(f"\nPerformance metrics:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Average time per array: {total_time / N_ARRAYS:.3f} seconds")
    print(f"  Throughput: {np.sum(sizes_mb) / total_time:.2f} MB/s")

    print(f"\nMemory usage:")
    print(f"  Baseline memory: {min_mem:.2f} MB")
    print(f"  Peak memory: {max_mem:.2f} MB")
    print(f"  Memory increase: {mem_increase:.2f} MB")

    # Check final directory size
    if os.path.exists(OUTPUT_DIR):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(OUTPUT_DIR):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)

        compressed_size_mb = total_size / (1024 * 1024)
        compression_ratio = np.sum(sizes_mb) / compressed_size_mb

        print(f"\nStorage efficiency:")
        print(f"  Compressed size: {compressed_size_mb:.2f} MB")
        print(f"  Compression ratio: {compression_ratio:.2f}x")


def main():
    """Run the benchmark."""
    print("TreeStore Large Array Benchmark")
    print("="*60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate array sizes
    print(f"Generating {N_ARRAYS} array sizes with peak at {PEAK_SIZE_MB} MB...")
    sizes_mb, sizes_elements = generate_array_sizes(
        N_ARRAYS, PEAK_SIZE_MB, STDDEV_MB, MIN_SIZE_MB, MAX_SIZE_MB
    )

    # Create test arrays
    arrays = create_test_arrays(sizes_elements)

    # Measure memory usage during storage
    memory_stats = measure_memory_usage(store_arrays_in_treestore, arrays, OUTPUT_DIR)
    total_time = memory_stats[0]  # This will be overwritten, we need the actual time

    # Get the actual timing by running the storage function again
    # (memory_profiler doesn't return the function result easily)
    print("\nRunning final timing measurement...")
    actual_time = store_arrays_in_treestore(arrays, OUTPUT_DIR)

    # Print results
    print_statistics(sizes_mb, sizes_elements, actual_time, memory_stats)

    print(f"\nBenchmark completed. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
