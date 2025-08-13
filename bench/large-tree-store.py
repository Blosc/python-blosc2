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

The arrays in h5py/zarr are compressed with the same defaults as in TreeStore.
Moreover, the chunks for storing arrays in h5py/zarr are set to Blosc2's blocks
(first partition) which should lead to same compression ratio as in TreeStore.

Note: This adapts to zarr v3+ API if available.
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
OUTPUT_DIR_TSTORE = "large-tree-store.b2z"
OUTPUT_FILE_H5PY = "large-h5py-store.h5"
OUTPUT_DIR_ZARR = "large-zarr-store.zarr"
MIN_SIZE_MB = 0  # Minimum array size in MB
MAX_SIZE_MB = PEAK_SIZE_MB * 10  # Maximum array size in MB
CHECK_VALUES = True  # Set to False to disable value checking (it is fast anyway)


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
        # arr = np.linspace(0, i, size, dtype=np.float64)
        arr = blosc2.linspace(0, i, size, dtype=np.float64)
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

    # Setting cparams here is not good because storages use different defaults
    # Better leave without specifying compression, although ZSTD level 5 should be used internally
    # cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, filters=[blosc2.Filter.SHUFFLE])
    # with blosc2.TreeStore(output_dir, mode="w", cparams=cparams) as tstore:
    with blosc2.TreeStore(output_dir, mode="w") as tstore:
        for i, arr in enumerate(arrays):
            # Distribute arrays evenly across NGROUPS_MAX subdirectories
            group_id = i % NGROUPS_MAX
            key = f"/group_{group_id:02d}/array_{i:04d}"
            tstore[key] = arr[:]

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

            # Store array with compression; use arr.blocks (first partition in Blosc2) as chunks
            grp.create_dataset(dataset_name, data=arr[:],
                               # compression="gzip", shuffle=True,
                               # To compare apples with apples, use Blosc2 compression with Zstd compression
                               compression=hdf5plugin.Blosc2(cname='zstd', clevel=5,
                                                             filters=hdf5plugin.Blosc2.SHUFFLE),
                               chunks=arr.blocks,
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

    # Create zarr store
    if zarr.__version__ >= "3":
        # (zarr v3+ API)
        store = zarr.storage.LocalStore(output_dir)
    else:
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

        # Store array with blosc2 compression; use arr.blocks (first partition in Blosc2) as chunks
        if zarr.__version__ >= "3":
            grp.create_array(
                name=dataset_name,
                data=arr[:],
                compressors=zarr.codecs.BloscCodec(
                    cname="zstd", clevel=5, shuffle=zarr.codecs.BloscShuffle.shuffle),
                chunks=arr.blocks,
            )
        else:
            grp.create_dataset(
                name=dataset_name,
                data=arr[:],
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=zarr.Blosc.SHUFFLE),
                chunks=arr.blocks,
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


def measure_access_time(arrays, results_tuple, backend_name):
    """Measure average access time for reading 10 elements from the middle of arrays."""
    if results_tuple is None:
        return None

    print(f"\nMeasuring access time for {backend_name}...")

    # Determine the path/file to open
    if backend_name == "TreeStore":
        store_path = OUTPUT_DIR_TSTORE
    elif backend_name == "h5py":
        store_path = OUTPUT_FILE_H5PY
    elif backend_name == "zarr":
        store_path = OUTPUT_DIR_ZARR
    else:
        return None

    access_times = []

    try:
        if backend_name == "TreeStore":
            with blosc2.TreeStore(store_path, mode="r") as store:
                for i, arr in enumerate(arrays):
                    group_id = i % NGROUPS_MAX
                    key = f"/group_{group_id:02d}/array_{i:04d}"

                    # Get middle slice indices
                    mid_point = len(arr) // 2
                    start_idx = max(0, mid_point - 5)
                    end_idx = min(len(arr), start_idx + 10)

                    start_time = time.perf_counter()
                    retrieved_slice = store[key][start_idx:end_idx]
                    end_time = time.perf_counter()

                    # Check values if enabled
                    if CHECK_VALUES:
                        expected_slice = arr[start_idx:end_idx]
                        if not np.allclose(retrieved_slice, expected_slice):
                            raise ValueError(f"Value mismatch for {backend_name} key {key}")

                    access_times.append(end_time - start_time)

        elif backend_name == "h5py" and HAS_H5PY:
            with h5py.File(store_path, "r") as f:
                for i, arr in enumerate(arrays):
                    group_id = i % NGROUPS_MAX
                    group_name = f"group_{group_id:02d}"
                    dataset_name = f"array_{i:04d}"

                    # Get middle slice indices
                    mid_point = len(arr) // 2
                    start_idx = max(0, mid_point - 5)
                    end_idx = min(len(arr), start_idx + 10)

                    start_time = time.perf_counter()
                    retrieved_slice = f[group_name][dataset_name][start_idx:end_idx]
                    end_time = time.perf_counter()

                    # Check values if enabled
                    if CHECK_VALUES:
                        expected_slice = arr[start_idx:end_idx]
                        if not np.allclose(retrieved_slice, expected_slice):
                            raise ValueError(f"Value mismatch for {backend_name} key {group_name}/{dataset_name}")

                    access_times.append(end_time - start_time)

        elif backend_name == "zarr" and HAS_ZARR:
            if zarr.__version__ >= "3":
                store = zarr.storage.LocalStore(store_path)
            else:
                store = zarr.DirectoryStore(store_path)
            root = zarr.group(store=store)

            for i, arr in enumerate(arrays):
                group_id = i % NGROUPS_MAX
                group_name = f"group_{group_id:02d}"
                dataset_name = f"array_{i:04d}"

                # Get middle slice indices
                mid_point = len(arr) // 2
                start_idx = max(0, mid_point - 5)
                end_idx = min(len(arr), start_idx + 10)

                start_time = time.perf_counter()
                retrieved_slice = root[group_name][dataset_name][start_idx:end_idx]
                end_time = time.perf_counter()

                # Check values if enabled
                if CHECK_VALUES:
                    expected_slice = arr[start_idx:end_idx]
                    if not np.allclose(retrieved_slice, expected_slice):
                        raise ValueError(f"Value mismatch for {backend_name} key {group_name}/{dataset_name}")

                access_times.append(end_time - start_time)

    except Exception as e:
        print(f"Error measuring access time for {backend_name}: {e}")
        return None

    avg_access_time = np.mean(access_times) * 1000  # Convert to milliseconds

    if CHECK_VALUES:
        print(f"  Value checking passed for {backend_name}")

    return avg_access_time


def measure_complete_read_time(arrays, results_tuple, backend_name):
    """Measure time to read all arrays completely into memory as numpy arrays."""
    if results_tuple is None:
        return None

    print(f"\nMeasuring complete read time for {backend_name}...")

    # Determine the path/file to open
    if backend_name == "TreeStore":
        store_path = OUTPUT_DIR_TSTORE
    elif backend_name == "h5py":
        store_path = OUTPUT_FILE_H5PY
    elif backend_name == "zarr":
        store_path = OUTPUT_DIR_ZARR
    else:
        return None

    try:
        start_time = time.perf_counter()

        if backend_name == "TreeStore":
            with blosc2.TreeStore(store_path, mode="r") as store:
                for i, _ in enumerate(arrays):
                    group_id = i % NGROUPS_MAX
                    key = f"/group_{group_id:02d}/array_{i:04d}"
                    # Read complete array into memory
                    _ = np.array(store[key][:])

        elif backend_name == "h5py" and HAS_H5PY:
            with h5py.File(store_path, "r") as f:
                for i, _ in enumerate(arrays):
                    group_id = i % NGROUPS_MAX
                    group_name = f"group_{group_id:02d}"
                    dataset_name = f"array_{i:04d}"
                    # Read complete array into memory
                    _ = np.array(f[group_name][dataset_name][:])

        elif backend_name == "zarr" and HAS_ZARR:
            if zarr.__version__ >= "3":
                store = zarr.storage.LocalStore(store_path)
            else:
                store = zarr.DirectoryStore(store_path)
            root = zarr.group(store=store)

            for i, _ in enumerate(arrays):
                group_id = i % NGROUPS_MAX
                group_name = f"group_{group_id:02d}"
                dataset_name = f"array_{i:04d}"
                # Read complete array into memory
                _ = np.array(root[group_name][dataset_name][:])

        end_time = time.perf_counter()
        total_read_time = end_time - start_time

    except Exception as e:
        print(f"Error measuring complete read time for {backend_name}: {e}")
        return None

    return total_read_time


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
    read_times = []
    storage_sizes = []
    access_times = []

    # TreeStore data
    backends.append('TreeStore')
    times.append(tstore_results[0])
    read_times.append(tstore_results[4] if len(tstore_results) > 4 else 0)
    storage_sizes.append(tstore_results[2])
    access_times.append(tstore_results[3] if len(tstore_results) > 3 else 0)

    # h5py data
    if h5py_results:
        backends.append('h5py')
        times.append(h5py_results[0])
        read_times.append(h5py_results[4] if len(h5py_results) > 4 else 0)
        storage_sizes.append(h5py_results[2])
        access_times.append(h5py_results[3] if len(h5py_results) > 3 else 0)

    # zarr data
    if zarr_results:
        backends.append('zarr')
        times.append(zarr_results[0])
        read_times.append(zarr_results[4] if len(zarr_results) > 4 else 0)
        storage_sizes.append(zarr_results[2])
        access_times.append(zarr_results[3] if len(zarr_results) > 3 else 0)

    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Colors for each backend
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    backend_colors = {backend: colors[i] for i, backend in enumerate(['TreeStore', 'h5py', 'zarr'])}
    plot_colors = [backend_colors[backend] for backend in backends]

    # Plot 1: Total Write Time (top-left)
    bars1 = ax1.bar(backends, times, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_title('Total Write Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, time_val in zip(bars1, times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 2: Total Read Time (top-right)
    bars2 = ax2.bar(backends, read_times, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_title('Total Read Time', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, read_val in zip(bars2, read_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{read_val:.2f}s', ha='center', va='bottom', fontweight='bold')

    # Plot 3: Access Time (bottom-left)
    bars3 = ax3.bar(backends, access_times, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax3.set_title('Average Access Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Time (milliseconds)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, access_val in zip(bars3, access_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{access_val:.3f}ms', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Storage Size (bottom-right)
    bars4 = ax4.bar(backends, storage_sizes, color=plot_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax4.set_title('Storage Size', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Size (MB)', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, size_val in zip(bars4, storage_sizes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{size_val:.1f}MB', ha='center', va='bottom', fontweight='bold')

    # Add compression ratio annotations
    for i, (backend, storage_size) in enumerate(zip(backends, storage_sizes)):
        compression_ratio = total_data_mb / storage_size
        ax4.text(i, storage_size/2, f'{compression_ratio:.1f}x',
                ha='center', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Adjust layout and add overall title
    plt.tight_layout()
    fig.suptitle(f'Performance Comparison: {N_ARRAYS} arrays, {total_data_mb:.1f} MB total data',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add extra space at the top for the title
    plt.subplots_adjust(top=0.90)

    # Save plot
    plot_filename = 'benchmark_comparison.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {plot_filename}")

    # Show plot
    plt.show()


def print_comparison_table(sizes_mb, tstore_results, h5py_results, zarr_results):
    """Print a comparison table of TreeStore vs h5py vs zarr results."""
    total_data_mb = np.sum(sizes_mb)

    print("\n" + "="*115)
    print("PERFORMANCE COMPARISON: TreeStore vs h5py vs zarr")
    print("="*115)

    # Configuration info
    print(f"Configuration:")
    print(f"  Arrays: {N_ARRAYS:,} | Peak size: {PEAK_SIZE_MB} MB | Total data: {total_data_mb:.1f} MB")
    print()

    # Extract results
    tstore_time, tstore_memory, tstore_storage = tstore_results[:3]
    tstore_access = tstore_results[3] if len(tstore_results) > 3 else None
    tstore_read = tstore_results[4] if len(tstore_results) > 4 else None

    if h5py_results:
        h5py_time, h5py_memory, h5py_storage = h5py_results[:3]
        h5py_access = h5py_results[3] if len(h5py_results) > 3 else None
        h5py_read = h5py_results[4] if len(h5py_results) > 4 else None
        has_h5py = True
    else:
        has_h5py = False

    if zarr_results:
        zarr_time, zarr_memory, zarr_storage = zarr_results[:3]
        zarr_access = zarr_results[3] if len(zarr_results) > 3 else None
        zarr_read = zarr_results[4] if len(zarr_results) > 4 else None
        has_zarr = True
    else:
        has_zarr = False

    # Table header
    print(f"{'Metric':<30} {'TreeStore':<15} {'h5py':<15} {'zarr':<15} {'Best':<12}")
    print("-" * 110)

    # Time metrics
    times = [tstore_time]
    time_labels = ['TreeStore']
    print(f"{'Write time (s)':<30} {tstore_time:<15.2f} ", end="")

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

    # Complete read time
    if tstore_read is not None:
        read_times = [tstore_read]
        read_labels = ['TreeStore']
        print(f"{'Total read time (s)':<30} {tstore_read:<15.2f} ", end="")

        if has_h5py and h5py_read is not None:
            print(f"{h5py_read:<15.2f} ", end="")
            read_times.append(h5py_read)
            read_labels.append('h5py')
        else:
            print(f"{'N/A':<15} ", end="")

        if has_zarr and zarr_read is not None:
            print(f"{zarr_read:<15.2f} ", end="")
            read_times.append(zarr_read)
            read_labels.append('zarr')
        else:
            print(f"{'N/A':<15} ", end="")

        best_read_idx = np.argmin(read_times)
        print(f"{read_labels[best_read_idx]:<12}")

    # Throughput
    throughputs = [total_data_mb/tstore_time]
    print(f"{'Write throughput (MB/s)':<30} {total_data_mb/tstore_time:<15.1f} ", end="")

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

    # Read throughput
    if tstore_read is not None:
        read_throughputs = [total_data_mb/tstore_read]
        print(f"{'Read throughput (MB/s)':<30} {total_data_mb/tstore_read:<15.1f} ", end="")

        if has_h5py and h5py_read is not None:
            h5py_read_throughput = total_data_mb / h5py_read
            print(f"{h5py_read_throughput:<15.1f} ", end="")
            read_throughputs.append(h5py_read_throughput)
        else:
            print(f"{'N/A':<15} ", end="")

        if has_zarr and zarr_read is not None:
            zarr_read_throughput = total_data_mb / zarr_read
            print(f"{zarr_read_throughput:<15.1f} ", end="")
            read_throughputs.append(zarr_read_throughput)
        else:
            print(f"{'N/A':<15} ", end="")

        best_read_throughput_idx = np.argmax(read_throughputs)
        print(f"{read_labels[best_read_throughput_idx]:<12}")

    # Access time
    if tstore_access is not None:
        access_times = [tstore_access]
        access_labels = ['TreeStore']
        print(f"{'Access time (ms)':<30} {tstore_access:<15.3f} ", end="")

        if has_h5py and h5py_access is not None:
            print(f"{h5py_access:<15.3f} ", end="")
            access_times.append(h5py_access)
            access_labels.append('h5py')
        else:
            print(f"{'N/A':<15} ", end="")

        if has_zarr and zarr_access is not None:
            print(f"{zarr_access:<15.3f} ", end="")
            access_times.append(zarr_access)
            access_labels.append('zarr')
        else:
            print(f"{'N/A':<15} ", end="")

        best_access_idx = np.argmin(access_times)
        print(f"{access_labels[best_access_idx]:<12}")

    print()

    # Memory metrics (kept in table)
    memories = [tstore_memory[2]]
    print(f"{'Memory increase (MB)':<30} {tstore_memory[2]:<15.1f} ", end="")

    if has_h5py:
        print(f"{h5py_memory[2]:<15.1f} ", end="")
        memories.append(h5py_memory[2])
    else:
        print(f"{'N/A':<15} ", end="")

    if has_zarr:
        print(f"{zarr_memory[2]:<15.1f} ", end="")
        memories.append(zarr_memory[2])
    else:
        print(f"{'N/A':<15} ", end="")

    best_memory_idx = np.argmin(memories)
    print(f"{time_labels[best_memory_idx]:<12}")

    # Storage metrics
    storages = [tstore_storage]
    print(f"{'Storage size (MB)':<30} {tstore_storage:<15.1f} ", end="")

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
    print(f"{'Compression ratio':<30} {total_data_mb/tstore_storage:<15.2f} ", end="")

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
    print(f"  Fastest write: {best_overall} ({times[best_time_idx]:.2f}s)")

    if tstore_read is not None:
        best_read = read_labels[best_read_idx]
        print(f"  Fastest total read: {best_read} ({read_times[best_read_idx]:.2f}s)")

    best_storage = time_labels[best_storage_idx]
    print(f"  Most compact: {best_storage} ({storages[best_storage_idx]:.1f} MB)")

    best_memory = time_labels[best_memory_idx]
    print(f"  Lowest memory increase: {best_memory} ({memories[best_memory_idx]:.1f} MB)")

    if tstore_access is not None:
        best_access = access_labels[best_access_idx]
        print(f"  Fastest access: {best_access} ({access_times[best_access_idx]:.3f} ms)")


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
        h5py_access_time = measure_access_time(arrays, (h5py_time, h5py_memory_stats, h5py_storage_size), "h5py")
        h5py_read_time = measure_complete_read_time(arrays, (h5py_time, h5py_memory_stats, h5py_storage_size), "h5py")
        h5py_results = (h5py_time, h5py_memory_stats, h5py_storage_size, h5py_access_time, h5py_read_time)
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
        zarr_access_time = measure_access_time(arrays, (zarr_time, zarr_memory_stats, zarr_storage_size), "zarr")
        zarr_read_time = measure_complete_read_time(arrays, (zarr_time, zarr_memory_stats, zarr_storage_size), "zarr")
        zarr_results = (zarr_time, zarr_memory_stats, zarr_storage_size, zarr_access_time, zarr_read_time)
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
    tstore_access_time = measure_access_time(arrays, (tstore_time, tstore_memory_stats, tstore_storage_size), "TreeStore")
    tstore_read_time = measure_complete_read_time(arrays, (tstore_time, tstore_memory_stats, tstore_storage_size), "TreeStore")
    tstore_results = (tstore_time, tstore_memory_stats, tstore_storage_size, tstore_access_time, tstore_read_time)

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
