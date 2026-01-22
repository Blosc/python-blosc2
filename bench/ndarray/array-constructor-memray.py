#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from time import time
import os

import numpy as np
import memray

import blosc2

N = 100_000_000


def info(a, t1):
    size = a.schunk.nbytes
    csize = a.schunk.cbytes
    print(
        f"Time: {t1:.3f} s - size: {size / 2 ** 30:.2f} GB ({size / t1 / 2 ** 30:.2f} GB/s)"
        f"\tStorage required: {csize / 2 ** 20:.2f} MB (cratio: {size / csize:.1f}x)"
    )


def run_benchmark():
    shape = (N,)
    shape = (100, 1000, 1000)
    print(f"*** Creating a blosc2 array with {N:_} elements (shape: {shape}) ***")
    t0 = time()
    #a = blosc2.arange(N, shape=shape, dtype=np.int32, urlpath="a.b2nd", mode="w")
    a = blosc2.linspace(0, 1, N, shape=shape, dtype=np.float64, urlpath="a.b2nd", mode="w")
    elapsed = time() - t0
    info(a, elapsed)
    return a


# Check if we're being tracked by memray
if not os.environ.get("MEMRAY_TRACKING", False):
    # Run the benchmark with memray tracking
    output_file = "array_constructor_memray.bin"
    print(f"Starting memray profiling. Results will be saved to {output_file}")

    with memray.Tracker(output_file):
        array = run_benchmark()

    print(f"\nMemray profiling completed. To view results, run:")
    print(f"memray flamegraph {output_file}")
    print(f"# or")
    print(f"memray summary {output_file}")
    print(f"# or")
    print(f"memray tree {output_file}")
else:
    # We're already being tracked by memray
    run_benchmark()
