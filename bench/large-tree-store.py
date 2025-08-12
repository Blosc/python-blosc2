#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""
Benchmark for TreeStore vs h5py vs zarr with large arrays.

This benchmark creates N numpy arrays with sizes following a normal distribution
and measures the time and memory consumption for storing them in TreeStore, h5py, and zarr.
"""

import os
import shutil
import time

from memory_profiler import profile, memory_usage
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import blosc2

try:
    import h5py
    import hdf5plugin
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

# Configuration
N_ARRAYS = 100  # Number of arrays to store
NGROUPS_MAX = 10
PEAK_SIZE_MB = 10  # Peak size in MB for the normal distribution
STDDEV_MB = 2  # Standard deviation in MB
OUTPUT_DIR_TSTORE = "large-tree-store.b2d"
OUTPUT_FILE_H5PY = "large-h5py-store.h5"
OUTPUT_DIR_ZARR = "large-zarr-store.zarr"
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

        # if (i + 1) % 10 == 0:
        #     print(f"  Created {i + 1}/{len(sizes_elements)} arrays")

    return arrays


#@profile
def store_arrays_in_treestore(arrays, output_dir):
    """Store arrays in TreeStore and measure performance."""
    print(f"Storing {len(arrays)} arrays in TreeStore at {output_dir}...")

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

            # if (i + 1) % 10 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"  Stored {i + 1}/{len(arrays)} arrays ({elapsed:.2f}s)")

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

    print(f"Storing {len(arrays)} arrays in h5py at {output_file}...")

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

            # if (i + 1) % 10 == 0:
            #     elapsed = time.time() - start_time
            #     print(f"  Stored {i + 1}/{len(arrays)} arrays ({elapsed:.2f}s)")

        # Add some metadata
        f.attrs["n_arrays"] = len(arrays)
        f.attrs["peak_size_mb"] = PEAK_SIZE_MB
        f.attrs["benchmark_timestamp"] = time.time()
        f.attrs["n_groups"] = NGROUPS_MAX

    end_time = time.time()
    total_time = end_time - start_time

    return total_time


#@profile
def store_arrays_in_zarr(arrays, output_dir):
    """Store arrays in zarr and measure performance."""
    if not HAS_ZARR:
        return None

    print(f"Storing {len(arrays)} arrays in zarr at {output_dir}...")

    # Clean up existing directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    start_time = time.time()

    # Create zarr store with blosc2 compression
    store = zarr.DirectoryStore(output_dir)
    root = zarr.group(store=store)

    for i, arr in enumerate(arrays):
        # Distribute arrays evenly across NGROUPS_MAX subdirectories
        group_id = i % NGROUPS_MAX
        group_name = f"group_{group_id:02d}"
        dataset_name = f"array_{i:04d}"

        # Create group if it doesn't exist
        if group_name not in root:
            grp = root.create_group(group_name)
        else:
            grp = root[group_name]

        # Store array with blosc2 compression
        grp.create_dataset(
            dataset_name,
            data=arr,
            compressor=zarr.Blosc(cname='zstd', clevel=5, shuffle=zarr.Blosc.SHUFFLE),
            chunks=True
        )

        # if (i + 1) % 10 == 0:
        #     elapsed = time.time() - start_time
        #     print(f"  Stored {i + 1}/{len(arrays)} arrays ({elapsed:.2f}s)")

    # Add some metadata
    root.attrs["n_arrays"] = len(arrays)
    root.attrs["peak_size_mb"] = PEAK_SIZE_MB
    root.attrs["benchmark_timestamp"] = time.time()
    root.attrs["n_groups"] = NGROUPS_MAX

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
    """Get storage size in MB for a file or directory (cross-platform)."""
    if not os.path.exists(path):
        return 0

    total_size = 0
    if os.path.isfile(path):
        if os.name == 'nt':  # Windows
            total_size = os.path.getsize(path)
        else:  # macOS, Linux
            # st_blocks is in 512-byte units
            total_size = os.stat(path).st_blocks * 512
    elif os.path.isdir(path):
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                filepath = os.path.join(dirpath, f)
                if not os.path.islink(filepath):
                    if os.name == 'nt': # Windows
                        total_size += os.path.getsize(filepath)
                    else: # macOS, Linux
                        try:
                            total_size += os.stat(filepath).st_blocks * 512
                        except (FileNotFoundError, PermissionError):
                            pass # Ignore broken symlinks or permission errors
            # Add directory size itself on non-Windows systems
            if os.name != 'nt':
                try:
                    total_size += os.stat(dirpath).st_blocks * 512
                except (FileNotFoundError, PermissionError):
                    pass

    return total_size / (1024 * 1024)


def create_comparison_plot(sizes_mb, tstore_results, h5py_results, zarr_results):
    """Create a bar plot comparing the three backends across different metrics."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available - skipping plot generation")
        return

    # Extract data
    total_data_mb = np.sum(sizes_mb)

    # Prepare data for plotting
    backends = []
    times = []
    memory_increases = []
    storage_sizes = []

    # TreeStore data
    backends.append('TreeStore')
    times.append(tstore_results[0])
    memory_increases.append(tstore_results[1][2])  # memory increase
    storage_sizes.append(tstore_results[2])

    # h5py data
    if h5py_results:
        backends.append('h5py')
        times.append(h5py_results[0])
        memory_increases.append(h5py_results[1][2])
        storage_sizes.append(h5py_results[2])

    # zarr data
    if zarr_results:
        backends.append('zarr')
        times.append(zarr_results[0])
        memory_increases.append(zarr_results[1][2])
        storage_sizes.append(zarr_results[2])

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # Colors for each backend
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    backend_colors = {backend: colors[i] for i, backend in enumerate(['TreeStore', 'h5py', 'zarr'])}
    plot_colors = [backend_colors[backend] for backend in backends]

    # Plot 1: Total Time
    bars1 = ax1.bar(backends, times, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Total Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Memory Increase
    bars2 = ax2.bar(backends, memory_increases, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Memory Increase', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Memory (MB)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, mem_val in zip(bars2, memory_increases):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mem_val:.1f}MB', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Storage Size
    bars3 = ax3.bar(backends, storage_sizes, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title('Storage Size', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Size (MB)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, size_val in zip(bars3, storage_sizes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size_val:.1f}MB', ha='center', va='bottom', fontweight='bold')

    # Adjust layout and add overall title
    plt.tight_layout()
    fig.suptitle(f'Performance Comparison: {N_ARRAYS} arrays, {total_data_mb:.1f} MB total data',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add extra space at the top for the title
    plt.subplots_adjust(top=0.85)

    # Add compression ratio annotations
    for i, (backend, storage_size) in enumerate(zip(backends, storage_sizes)):
        compression_ratio = total_data_mb / storage_size
        ax3.text(i, storage_size/2, f'{compression_ratio:.1f}x',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Save plot
    plot_filename = 'benchmark_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")

    # Show plot
    plt.show()


def print_comparison_table(sizes_mb, tstore_results, h5py_results, zarr_results):
    """Print a comparison table of TreeStore vs h5py vs zarr results."""
    total_data_mb = np.sum(sizes_mb)

    print("\n" + "="*90)
    print("PERFORMANCE COMPARISON: TreeStore vs h5py vs zarr")
    print("="*90)

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

    if zarr_results:
        zarr_time, zarr_memory, zarr_storage = zarr_results
        has_zarr = True
    else:
        has_zarr = False

    # Table header
    print(f"{'Metric':<25} {'TreeStore':<15} {'h5py':<15} {'zarr':<15} {'Best':<12}")
    print("-" * 85)

    # Time metrics
    times = [tstore_time]
    time_labels = ['TreeStore']
    print(f"{'Total time (s)':<25} {tstore_time:<15.2f} ", end="")

    if has_h5py:
        print(f"{h5py_time:<15.2f} ", end="")
        times.append(h5py_time)
        time_labels.append('h5py')
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        print(f"{zarr_time:<15.2f} ", end="")
        times.append(zarr_time)
        time_labels.append('zarr')
    else:
        print(f"{'N/A':<15} ", end="")

    best_time_idx = np.argmin(times)
    print(f"{time_labels[best_time_idx]:<12}")

    # Throughput
    throughputs = [total_data_mb/tstore_time]
    print(f"{'Throughput (MB/s)':<25} {total_data_mb/tstore_time:<15.1f} ", end="")

    if has_h5py:
        h5py_throughput = total_data_mb / h5py_time
        print(f"{h5py_throughput:<15.1f} ", end="")
        throughputs.append(h5py_throughput)
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        zarr_throughput = total_data_mb / zarr_time
        print(f"{zarr_throughput:<15.1f} ", end="")
        throughputs.append(zarr_throughput)
    else:
        print(f"{'N/A':<15} ", end="")

    best_throughput_idx = np.argmax(throughputs)
    print(f"{time_labels[best_throughput_idx]:<12}")

    print()

    # Memory metrics
    memories = [tstore_memory[0]]
    print(f"{'Peak memory (MB)':<25} {tstore_memory[0]:<15.1f} ", end="")

    if has_h5py:
        print(f"{h5py_memory[0]:<15.1f} ", end="")
        memories.append(h5py_memory[0])
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        print(f"{zarr_memory[0]:<15.1f} ", end="")
        memories.append(zarr_memory[0])
    else:
        print(f"{'N/A':<15} ", end="")

    best_memory_idx = np.argmin(memories)
    print(f"{time_labels[best_memory_idx]:<12}")

    # Memory increase
    mem_increases = [tstore_memory[2]]
    print(f"{'Memory increase (MB)':<25} {tstore_memory[2]:<15.1f} ", end="")

    if has_h5py:
        print(f"{h5py_memory[2]:<15.1f} ", end="")
        mem_increases.append(h5py_memory[2])
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        print(f"{zarr_memory[2]:<15.1f} ", end="")
        mem_increases.append(zarr_memory[2])
    else:
        print(f"{'N/A':<15} ", end="")

    best_mem_inc_idx = np.argmin(mem_increases)
    print(f"{time_labels[best_mem_inc_idx]:<12}")

    print()

    # Storage metrics
    storages = [tstore_storage]
    print(f"{'Storage size (MB)':<25} {tstore_storage:<15.1f} ", end="")

    if has_h5py:
        print(f"{h5py_storage:<15.1f} ", end="")
        storages.append(h5py_storage)
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        print(f"{zarr_storage:<15.1f} ", end="")
        storages.append(zarr_storage)
    else:
        print(f"{'N/A':<15} ", end="")

    best_storage_idx = np.argmin(storages)
    print(f"{time_labels[best_storage_idx]:<12}")

    # Compression ratio
    compressions = [total_data_mb/tstore_storage]
    print(f"{'Compression ratio':<25} {total_data_mb/tstore_storage:<15.2f} ", end="")

    if has_h5py:
        h5py_compression = total_data_mb / h5py_storage
        print(f"{h5py_compression:<15.2f} ", end="")
        compressions.append(h5py_compression)
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        zarr_compression = total_data_mb / zarr_storage
        print(f"{zarr_compression:<15.2f} ", end="")
        compressions.append(zarr_compression)
    else:
        print(f"{'N/A':<15} ", end="")

    best_compression_idx = np.argmax(compressions)
    print(f"{time_labels[best_compression_idx]:<12}")

    print()

    # Summary
    print("Summary:")
    best_overall = time_labels[best_time_idx]
    print(f"  Fastest: {best_overall} ({times[best_time_idx]:.2f}s)")

    best_storage = time_labels[best_storage_idx]
    print(f"  Most compact: {best_storage} ({storages[best_storage_idx]:.1f} MB)")

    best_memory = time_labels[best_memory_idx]
    print(f"  Lowest memory: {best_memory} ({memories[best_memory_idx]:.1f} MB)")


def main():
    """Run the benchmark."""
    print("TreeStore vs h5py vs zarr Large Array Benchmark")
    print("="*70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate array sizes
    print(f"Generating {N_ARRAYS} array sizes with peak at {PEAK_SIZE_MB} MB...")
    sizes_mb, sizes_elements = generate_array_sizes(
        N_ARRAYS, PEAK_SIZE_MB, STDDEV_MB, MIN_SIZE_MB, MAX_SIZE_MB
    )

    # Create test arrays
    arrays = create_test_arrays(sizes_elements)

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

    # Benchmark zarr if available
    zarr_results = None
    if HAS_ZARR:
        print("\n" + "="*60)
        print("BENCHMARKING zarr")
        print("="*60)
        zarr_memory_stats = measure_memory_usage(store_arrays_in_zarr, arrays, OUTPUT_DIR_ZARR)
        zarr_time = store_arrays_in_zarr(arrays, OUTPUT_DIR_ZARR)
        zarr_storage_size = get_storage_size(OUTPUT_DIR_ZARR)
        zarr_results = (zarr_time, zarr_memory_stats, zarr_storage_size)
    else:
        print("\n" + "="*60)
        print("zarr not available - skipping zarr benchmark")
        print("="*60)

    # Benchmark TreeStore (run last)
    print("\n" + "="*60)
    print("BENCHMARKING TreeStore")
    print("="*60)
    tstore_memory_stats = measure_memory_usage(store_arrays_in_treestore, arrays, OUTPUT_DIR_TSTORE)
    tstore_time = store_arrays_in_treestore(arrays, OUTPUT_DIR_TSTORE)
    tstore_storage_size = get_storage_size(OUTPUT_DIR_TSTORE)
    tstore_results = (tstore_time, tstore_memory_stats, tstore_storage_size)

    # Print comparison table
    print_comparison_table(sizes_mb, tstore_results, h5py_results, zarr_results)

    # Create comparison plot
    create_comparison_plot(sizes_mb, tstore_results, h5py_results, zarr_results)

    print(f"\nBenchmark completed.")
    print(f"TreeStore results saved to: {OUTPUT_DIR_TSTORE}")
    if HAS_H5PY:
        print(f"h5py results saved to: {OUTPUT_FILE_H5PY}")
    if HAS_ZARR:
        print(f"zarr results saved to: {OUTPUT_DIR_ZARR}")


if __name__ == "__main__":
    main()
