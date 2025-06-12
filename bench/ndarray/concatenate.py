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


def run_benchmark(num_arrays=10, size=500, aligned_chunks=False, axis=0):
    """
    Benchmark blosc2.concatenate performance with different chunk alignments.

    Parameters:
    - num_arrays: Number of arrays to concatenate
    - size: Base size for array dimensions
    - aligned_chunks: Whether to use aligned chunk shapes
    - axis: Axis along which to concatenate (0 or 1)
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
    if aligned_chunks:
        # Aligned chunks: divisors of the shape dimensions
        chunk_shapes = [(shape[0] // 4, shape[1] // 4) for shape in shapes]
    else:
        # Unaligned chunks: not divisors of shape dimensions
        chunk_shapes = [(shape[0] // 4 + 1, shape[1] // 4 - 1) for shape in shapes]

    # Create arrays
    arrays = []
    for i, (shape, chunk_shape) in enumerate(zip(shapes, chunk_shapes)):
        arr = blosc2.arange(
            i * np.prod(shape), (i + 1) * np.prod(shape), 1, dtype="i4", shape=shape, chunks=chunk_shape
        )
        arrays.append(arr)

    # Time the concatenation
    start_time = time.time()
    result = blosc2.concatenate(arrays, axis=axis)
    duration = time.time() - start_time

    return duration, result.shape


def main():
    print(f"{'=' * 50}")
    print(f"Blosc2 concatenation benchmark")
    print(f"{'=' * 50}")

    # Parameters
    sizes = [400, 800, 1600, 3200]  # must be divisible by 4 for aligned chunks
    num_arrays = 10

    for axis in [0, 1]:
        print(f"\nConcatenating {num_arrays} arrays along axis {axis}")
        print(f"{'Size':<10} {'Unaligned (s)':<15} {'Aligned (s)':<15} {'Speedup':<10}")
        print(f"{'-' * 50}")

        for size in sizes:
            # Run both benchmarks
            unaligned_time, shape1 = run_benchmark(num_arrays, size, aligned_chunks=False, axis=axis)
            aligned_time, shape2 = run_benchmark(num_arrays, size, aligned_chunks=True, axis=axis)

            # Calculate speedup
            speedup = unaligned_time / aligned_time if aligned_time > 0 else float("inf")

            # Print results
            print(f"{size:<10} {unaligned_time:<15.4f} {aligned_time:<15.4f} {speedup:<10.2f}x")

            # Quick verification of result shape
            if axis == 0:
                expected_shape = (size, size)  # After concatenation along axis 0
            else:
                expected_shape = (size, size)  # After concatenation along axis 1
            if shape1 != expected_shape:
                print(
                    f"Warning: result shape unaligned {shape1} does not match expected shape {expected_shape}"
                )
            if shape2 != expected_shape:
                print(
                    f"Warning: result shape aligned {shape2} does not match expected shape {expected_shape}"
                )

    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
