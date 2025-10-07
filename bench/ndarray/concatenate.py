#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import blosc2
import time
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter


def run_benchmark(num_arrays=10, size=500, aligned_chunks=False, axis=0,
                  dtype=np.float64, datadist="linspace", codec=blosc2.Codec.ZSTD):
    """
    Benchmark blosc2.concat performance with different chunk alignments.

    Parameters:
    - num_arrays: Number of arrays to concatenate
    - size: Base size for array dimensions
    - aligned_chunks: Whether to use aligned chunk shapes
    - axis: Axis along which to concatenate (0 or 1)
    - dtype: Data type for the arrays (default is np.float64)
    - datadist: Distribution of data in arrays (default is "linspace")
    - codec: Codec to use for compression (default is blosc2.Codec.ZSTD)

    Returns:
    - duration: Time taken in seconds
    - result_shape: Shape of the resulting array
    - data_size_gb: Size of data processed in GB
    """
    if axis == 0:
        # For concatenating along axis 0, the second dimension must be consistent
        shapes = [(size // num_arrays, size) for _ in range(num_arrays)]
    elif axis == 1:
        # For concatenating along axis 1, the first dimension must be consistent
        shapes = [(size, size // num_arrays) for _ in range(num_arrays)]
    else:
        raise ValueError("Only axis 0 and 1 are supported")

    # Create appropriate chunk shapes
    chunks, blocks = blosc2.compute_chunks_blocks(shapes[0], dtype=dtype, cparams=blosc2.CParams(codec=codec))
    if aligned_chunks:
        # Aligned chunks: divisors of the shape dimensions
        chunk_shapes = [(chunks[0], chunks[1]) for shape in shapes]
    else:
        # Unaligned chunks: not divisors of shape dimensions
        chunk_shapes = [(chunks[0] + 1, chunks[1] - 1) for shape in shapes]

    # Create arrays
    arrays = []
    for i, (shape, chunk_shape) in enumerate(zip(shapes, chunk_shapes)):
        if datadist == "linspace":
            # Create arrays with linearly spaced values
            arr = blosc2.linspace(i, i + 1, num=np.prod(shape),
                                  dtype=dtype, shape=shape, chunks=chunk_shape,
                                  cparams=blosc2.CParams(codec=codec))
        else:
            # Default to arange for simplicity
            arr = blosc2.arange(
                i * np.prod(shape), (i + 1) * np.prod(shape), 1, dtype=dtype, shape=shape, chunks=chunk_shape,
                cparams=blosc2.CParams(codec=codec)
            )
        arrays.append(arr)

    # Calculate total data size in GB (4 bytes per int32)
    total_elements = sum(np.prod(shape) for shape in shapes)
    data_size_gb = total_elements * 4 / (1024**3)  # Convert bytes to GB

    # Time the concatenation
    start_time = time.time()
    result = blosc2.concat(arrays, axis=axis, cparams=blosc2.CParams(codec=codec))
    duration = time.time() - start_time

    return duration, result.shape, data_size_gb


def run_numpy_benchmark(num_arrays=10, size=500, axis=0, dtype=np.float64, datadist="linspace"):
    """
    Benchmark numpy.concat performance for comparison.

    Parameters:
    - num_arrays: Number of arrays to concatenate
    - size: Base size for array dimensions
    - axis: Axis along which to concatenate (0 or 1)
    - dtype: Data type for the arrays (default is np.float64)
    - datadist: Distribution of data in arrays (default is "linspace")

    Returns:
    - duration: Time taken in seconds
    - result_shape: Shape of the resulting array
    - data_size_gb: Size of data processed in GB
    """
    if axis == 0:
        # For concatenating along axis 0, the second dimension must be consistent
        shapes = [(size // num_arrays, size) for _ in range(num_arrays)]
    elif axis == 1:
        # For concatenating along axis 1, the first dimension must be consistent
        shapes = [(size, size // num_arrays) for _ in range(num_arrays)]
    else:
        raise ValueError("Only axis 0 and 1 are supported")

    # Create arrays
    numpy_arrays = []
    for i, shape in enumerate(shapes):
        if datadist == "linspace":
            # Create arrays with linearly spaced values
            arr = np.linspace(i, i + 1, num=np.prod(shape), dtype=dtype).reshape(shape)
        else:
            arr = np.arange(i * np.prod(shape), (i + 1) * np.prod(shape), 1, dtype=dtype).reshape(shape)
        numpy_arrays.append(arr)

    # Calculate total data size in GB (4 bytes per int32)
    total_elements = sum(np.prod(shape) for shape in shapes)
    data_size_gb = total_elements * 4 / (1024**3)  # Convert bytes to GB

    # Time the concatenation
    start_time = time.time()
    result = np.concat(numpy_arrays, axis=axis)
    duration = time.time() - start_time

    return duration, result.shape, data_size_gb


def create_combined_plot(num_arrays, sizes, numpy_speeds_axis0, unaligned_speeds_axis0, aligned_speeds_axis0,
                         numpy_speeds_axis1, unaligned_speeds_axis1, aligned_speeds_axis1, output_dir="plots",
                         datadist="linspace", codec_str="LZ4"):
    """
    Create a figure with two side-by-side bar plots comparing the performance for both axes.

    Parameters:
    - sizes: List of array sizes
    - *_speeds_axis0: Lists of speeds (GB/s) for axis 0 concatenation
    - *_speeds_axis1: Lists of speeds (GB/s) for axis 1 concatenation
    - output_dir: Directory to save the plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set up the figure with two subplots side by side
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # Convert sizes to strings for the x-axis
    x_labels = [str(size) for size in sizes]
    x = np.arange(len(sizes))
    width = 0.25

    # Create bars for axis 0 plot
    rect1_axis0 = ax0.bar(x - width, numpy_speeds_axis0, width, label='NumPy', color='#1f77b4')
    rect2_axis0 = ax0.bar(x, unaligned_speeds_axis0, width, label='Blosc2 Unaligned', color='#ff7f0e')
    rect3_axis0 = ax0.bar(x + width, aligned_speeds_axis0, width, label='Blosc2 Aligned', color='#2ca02c')

    # Create bars for axis 1 plot
    rect1_axis1 = ax1.bar(x - width, numpy_speeds_axis1, width, label='NumPy', color='#1f77b4')
    rect2_axis1 = ax1.bar(x, unaligned_speeds_axis1, width, label='Blosc2 Unaligned', color='#ff7f0e')
    rect3_axis1 = ax1.bar(x + width, aligned_speeds_axis1, width, label='Blosc2 Aligned', color='#2ca02c')

    # Add labels and titles
    for ax, axis in [(ax0, 0), (ax1, 1)]:
        ax.set_xlabel('Array Size (N for NxN array)', fontsize=12)
        ax.set_title(f'Concatenation Performance for {num_arrays} arrays (axis={axis}) [{datadist}, {codec_str}]', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

        # Add legend inside each plot
        ax.legend(title="Concatenation Methods",
                 loc='upper left',
                 fontsize=12,
                 frameon=True,
                 facecolor='white',
                 edgecolor='black',
                 framealpha=0.8)

    # Add y-label only to the left subplot
    ax0.set_ylabel('Throughput (GB/s)', fontsize=12)

    # Add value labels on top of the bars
    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f} GB/s',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=8)

    autolabel(rect1_axis0, ax0)
    autolabel(rect2_axis0, ax0)
    autolabel(rect3_axis0, ax0)

    autolabel(rect1_axis1, ax1)
    autolabel(rect2_axis1, ax1)
    autolabel(rect3_axis1, ax1)

    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'concat_benchmark_combined.png'), dpi=100)
    plt.show()
    plt.close()

    print(f"Combined plot saved to {os.path.join(output_dir, 'concat_benchmark_combined.png')}")


def main():
    # Parameters
    sizes = [500, 1000, 2000, 4000, 10000]  #, 20000]  # Sizes of arrays to test
    num_arrays = 10
    dtype = np.float64  # Data type for arrays
    datadist = "linspace"  # Distribution of data in arrays
    codec = blosc2.Codec.LZ4
    codec_str = str(codec).split('.')[-1]
    print(f"{'=' * 70}")
    print(f"Blosc2 vs NumPy concatenation benchmark with {codec_str} codec")
    print(f"{'=' * 70}")


    # Lists to store results for both axes
    numpy_speeds_axis0 = []
    unaligned_speeds_axis0 = []
    aligned_speeds_axis0 = []
    numpy_speeds_axis1 = []
    unaligned_speeds_axis1 = []
    aligned_speeds_axis1 = []

    for axis in [0, 1]:
        print(f"\nConcatenating {num_arrays} arrays along axis {axis} with data distribution '{datadist}' ")
        print(f"{'Size':<8} {'NumPy (GB/s)':<14} {'Unaligned (GB/s)':<18} "
              f"{'Aligned (GB/s)':<16} {'Alig vs Unalig':<16} {'Alig vs NumPy':<16}")
        print(f"{'-' * 90}")

        for size in sizes:
            # Run the benchmarks
            numpy_time, numpy_shape, data_size_gb = run_numpy_benchmark(num_arrays, size, axis=axis, dtype=dtype)
            unaligned_time, shape1, _ = run_benchmark(num_arrays, size, aligned_chunks=False, axis=axis,
                                                      dtype=dtype, datadist=datadist, codec=codec)
            aligned_time, shape2, _ = run_benchmark(num_arrays, size, aligned_chunks=True, axis=axis,
                                                    dtype=dtype, datadist=datadist, codec=codec)

            # Calculate throughputs in GB/s
            numpy_speed = data_size_gb / numpy_time if numpy_time > 0 else float("inf")
            unaligned_speed = data_size_gb / unaligned_time if unaligned_time > 0 else float("inf")
            aligned_speed = data_size_gb / aligned_time if aligned_time > 0 else float("inf")

            # Store speeds in the appropriate list
            if axis == 0:
                numpy_speeds_axis0.append(numpy_speed)
                unaligned_speeds_axis0.append(unaligned_speed)
                aligned_speeds_axis0.append(aligned_speed)
            else:
                numpy_speeds_axis1.append(numpy_speed)
                unaligned_speeds_axis1.append(unaligned_speed)
                aligned_speeds_axis1.append(aligned_speed)

            # Calculate speedup ratios
            aligned_vs_unaligned = aligned_speed / unaligned_speed if unaligned_speed > 0 else float("inf")
            aligned_vs_numpy = aligned_speed / numpy_speed if numpy_speed > 0 else float("inf")

            # Print results
            print(f"{size:<10} {numpy_speed:<14.2f} {unaligned_speed:<18.2f} {aligned_speed:<16.2f} "
                  f"{aligned_vs_unaligned:>10.2f}x {aligned_vs_numpy:>10.2f}x")

            # Quick verification of result shape
            if axis == 0:
                expected_shape = (size, size)  # After concatenation along axis 0
            else:
                expected_shape = (size, size)  # After concatenation along axis 1

            # Verify shapes match
            shapes = [numpy_shape, shape1, shape2]
            if any(shape != expected_shape for shape in shapes):
                for i, shape_name in enumerate(["NumPy", "Blosc2 unaligned", "Blosc2 aligned"]):
                    if shapes[i] != expected_shape:
                        print(f"Warning: {shape_name} shape {shapes[i]} does not match expected {expected_shape}")

    print(f"{'=' * 70}")

    # Create the combined plot with both axes
    create_combined_plot(
        num_arrays,
        sizes,
        numpy_speeds_axis0, unaligned_speeds_axis0, aligned_speeds_axis0,
        numpy_speeds_axis1, unaligned_speeds_axis1, aligned_speeds_axis1,
        datadist=datadist, output_dir="plots", codec_str=codec_str,
    )


if __name__ == "__main__":
    main()
