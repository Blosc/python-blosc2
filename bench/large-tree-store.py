#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""
Benchmark for TreeStore vs h5py with large arrays.

This benchmark creates N numpy arrays with sizes following a normal distribution
and measures the time and memory consumption for storing them in both TreeStore and h5py.
"""

import os
import shutil
import time

from memory_profiler import profile, memory_usage
import numpy as np

import blosc2

try:
    import h5py
    import hdf5plugin
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Configuration
N_ARRAYS = 100  # Number of arrays to store
NGROUPS_MAX = 10
PEAK_SIZE_MB = 1  # Peak size in MB for the normal distribution
STDDEV_MB = .2  # Standard deviation in MB
OUTPUT_DIR_TSTORE = "large-tree-store.b2z"
OUTPUT_FILE_H5PY = "large-h5py-store.h5"
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


#@profile
def store_arrays_in_treestore(arrays, output_dir):
    """Store arrays in TreeStore and measure performance."""
    print(f"\nStoring {len(arrays)} arrays in TreeStore at {output_dir}...")

    # Clean up existing directory
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    elif os.path.exists(output_dir):
        os.remove(output_dir)

    start_time = time.time()

    # with blosc2.TreeStore(output_dir, mode="w", threshold=2**13) as tstore:
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


#@profile
def store_arrays_in_h5py(arrays, output_file):
    """Store arrays in h5py and measure performance."""
    if not HAS_H5PY:
        return None

    print(f"\nStoring {len(arrays)} arrays in h5py at {output_file}...")

    # Clean up existing file
    if os.path.exists(output_file):
        os.remove(output_file)

    start_time = time.time()

    with h5py.File(output_file, "w") as f:
        for i, arr in enumerate(arrays):
            # Distribute arrays evenly across NGROUPS_MAX subdirectories
            group_id = i % NGROUPS_MAX
            group_name = f"group_{group_id:02d}"
            dataset_name = f"array_{i:04d}"

            # Create group if it doesn't exist
            if group_name not in f:
                grp = f.create_group(group_name)
            else:
                grp = f[group_name]

            # Store array with compression
            grp.create_dataset(dataset_name, data=arr,
                               # compression="gzip", shuffle=True,
                               # To compare apples with apples, use Blosc2 compression with Zstd compression
                               compression=hdf5plugin.Blosc2(cname='zstd', clevel=5,
                                                             filters=hdf5plugin.Blosc2.SHUFFLE)
                               )

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  Stored {i + 1}/{len(arrays)} arrays ({elapsed:.2f}s)")

        # Add some metadata
        f.attrs["n_arrays"] = len(arrays)
        f.attrs["peak_size_mb"] = PEAK_SIZE_MB
        f.attrs["benchmark_timestamp"] = time.time()
        f.attrs["n_groups"] = NGROUPS_MAX

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


def get_storage_size(path):
    """Get storage size in MB for a file or directory."""
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    elif os.path.isdir(path):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size / (1024 * 1024)
    return 0


def print_comparison_table(sizes_mb, tstore_results, h5py_results):
    """Print a comparison table of TreeStore vs h5py results."""
    total_data_mb = np.sum(sizes_mb)

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: TreeStore vs h5py")
    print("="*80)

    # Configuration info
    print(f"Configuration:")
    print(f"  Arrays: {N_ARRAYS:,} | Peak size: {PEAK_SIZE_MB} MB | Total data: {total_data_mb:.1f} MB")
    print()

    # Extract results
    tstore_time, tstore_memory, tstore_storage = tstore_results
    if h5py_results:
        h5py_time, h5py_memory, h5py_storage = h5py_results
        has_h5py = True
    else:
        has_h5py = False

    # Table header
    print(f"{'Metric':<25} {'TreeStore':<15} {'h5py':<15} {'Ratio (T/H)':<12}")
    print("-" * 70)

    # Time metrics
    print(f"{'Total time (s)':<25} {tstore_time:<15.2f} ", end="")
    if has_h5py:
        ratio = tstore_time / h5py_time if h5py_time > 0 else float('inf')
        print(f"{h5py_time:<15.2f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print(f"{'Throughput (MB/s)':<25} {total_data_mb/tstore_time:<15.1f} ", end="")
    if has_h5py:
        h5py_throughput = total_data_mb / h5py_time
        ratio = (total_data_mb/tstore_time) / h5py_throughput if h5py_throughput > 0 else float('inf')
        print(f"{h5py_throughput:<15.1f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print()

    # Memory metrics
    print(f"{'Peak memory (MB)':<25} {tstore_memory[0]:<15.1f} ", end="")
    if has_h5py:
        ratio = tstore_memory[0] / h5py_memory[0] if h5py_memory[0] > 0 else float('inf')
        print(f"{h5py_memory[0]:<15.1f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print(f"{'Memory increase (MB)':<25} {tstore_memory[2]:<15.1f} ", end="")
    if has_h5py:
        ratio = tstore_memory[2] / h5py_memory[2] if h5py_memory[2] > 0 else float('inf')
        print(f"{h5py_memory[2]:<15.1f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print()

    # Storage metrics
    print(f"{'Storage size (MB)':<25} {tstore_storage:<15.1f} ", end="")
    if has_h5py:
        ratio = tstore_storage / h5py_storage if h5py_storage > 0 else float('inf')
        print(f"{h5py_storage:<15.1f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print(f"{'Compression ratio':<25} {total_data_mb/tstore_storage:<15.2f} ", end="")
    if has_h5py:
        h5py_compression = total_data_mb / h5py_storage
        ratio = (total_data_mb/tstore_storage) / h5py_compression if h5py_compression > 0 else float('inf')
        print(f"{h5py_compression:<15.2f} {ratio:<12.2f}")
    else:
        print(f"{'N/A':<15} {'N/A':<12}")

    print()

    # Summary
    print("Summary:")
    if has_h5py:
        if tstore_time < h5py_time:
            print(f"  TreeStore is {h5py_time/tstore_time:.1f}x faster")
        else:
            print(f"  h5py is {tstore_time/h5py_time:.1f}x faster")

        if tstore_storage < h5py_storage:
            print(f"  TreeStore uses {h5py_storage/tstore_storage:.1f}x less storage")
        else:
            print(f"  h5py uses {tstore_storage/h5py_storage:.1f}x less storage")
    else:
        print("  h5py not available for comparison")


def main():
    """Run the benchmark."""
    print("TreeStore vs h5py Large Array Benchmark")
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

    # Benchmark TreeStore
    print("\n" + "="*60)
    print("BENCHMARKING TreeStore")
    print("="*60)
    tstore_memory_stats = measure_memory_usage(store_arrays_in_treestore, arrays, OUTPUT_DIR_TSTORE)
    tstore_time = store_arrays_in_treestore(arrays, OUTPUT_DIR_TSTORE)
    tstore_storage_size = get_storage_size(OUTPUT_DIR_TSTORE)
    tstore_results = (tstore_time, tstore_memory_stats, tstore_storage_size)

    # Benchmark h5py if available
    h5py_results = None
    if HAS_H5PY:
        print("\n" + "="*60)
        print("BENCHMARKING h5py")
        print("="*60)
        h5py_memory_stats = measure_memory_usage(store_arrays_in_h5py, arrays, OUTPUT_FILE_H5PY)
        h5py_time = store_arrays_in_h5py(arrays, OUTPUT_FILE_H5PY)
        h5py_storage_size = get_storage_size(OUTPUT_FILE_H5PY)
        h5py_results = (h5py_time, h5py_memory_stats, h5py_storage_size)
    else:
        print("\n" + "="*60)
        print("h5py not available - skipping h5py benchmark")
        print("="*60)

    # Print comparison table
    print_comparison_table(sizes_mb, tstore_results, h5py_results)

    print(f"\nBenchmark completed.")
    print(f"TreeStore results saved to: {OUTPUT_DIR_TSTORE}")
    if HAS_H5PY:
        print(f"h5py results saved to: {OUTPUT_FILE_H5PY}")


if __name__ == "__main__":
    main()
