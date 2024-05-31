#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import argparse
from time import time

import numpy as np

import blosc2


class MmapBenchmarking:
    def __init__(self, io_type: str, blosc_mode: str) -> None:
        self.io_type = io_type
        self.blosc_mode = blosc_mode
        self.mmap_mode_write = "w+" if self.io_type == "io_mmap" else None
        self.mmap_mode_read = "r" if self.io_type == "io_mmap" else None
        self.urlpath = "array.b2nd"
        self.n_chunks = 1000

    def __enter__(self):
        blosc2.remove_urlpath(self.urlpath)
        np.random.seed(42)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        blosc2.remove_urlpath(self.urlpath)

    def benchmark_writes(self) -> float:
        array = np.random.randn(self.n_chunks, 64, 64, 64).astype(np.float32)

        if self.blosc_mode == "schunk":
            chunksize = array[0].size * array.itemsize
            schunk = blosc2.SChunk(chunksize=chunksize, mmap_mode=self.mmap_mode_write, urlpath=self.urlpath)

            t0 = time()
            for c in range(self.n_chunks):
                schunk.append_data(array[c])
            t1 = time()
        elif self.blosc_mode == "ndarray":
            t0 = time()
            blosc2.asarray(array, chunks=(1, 64, 64, 64), urlpath=self.urlpath, mmap_mode=self.mmap_mode_write)
            t1 = time()
        else:
            raise ValueError(f"Unknown Blosc mode: {self.blosc_mode}")
        
        return t1 - t0

    def benchmark_reads(self, read_order: str = "sequential") -> float:
        obj_open = blosc2.open(self.urlpath, mmap_mode=self.mmap_mode_read)
        chunks_order = np.arange(self.n_chunks)
        if read_order == "random":
            np.random.shuffle(chunks_order)

        if self.blosc_mode == "schunk":
            t0 = time()
            for c in chunks_order:
                obj_open.decompress_chunk(c)
            t1 = time()
        elif self.blosc_mode == "ndarray":
            t0 = time()
            for c in chunks_order:
                obj_open[c]
            t1 = time()

        return t1 - t0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark memory-mapped IO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--io-type",
        required=True,
        type=str,
        choices=["io_file", "io_mmap"],
        help="Basic I/O type: default file operations (io_file) or memory-mapped files (io_mmap).",
    )
    parser.add_argument(
        "--blosc-mode",
        required=True,
        type=str,
        choices=["schunk", "ndarray"],
        help="Whether the data is written or read via the SChunk or ndarray interfaces.",
    )
    parser.add_argument(
        "--runs",
        required=False,
        type=int,
        default=10,
        help="Number of times the schunk is written/read and the aggregated time is calculated.",
    )
    
    args = parser.parse_args()

    with MmapBenchmarking(io_type=args.io_type, blosc_mode=args.blosc_mode) as bench:
        times_write = []
        for i in range(args.runs):
            print(f"Run {i+1}/{args.runs}", end="\r")
            times_write.append(bench.benchmark_writes())
        print(f"Time for writing the data with {args.io_type}: {min(times_write):.3f}s")

        for read_order in ["sequential", "random"]:
            times_read = []
            for i in range(args.runs):
                print(f"Run {i+1}/{args.runs}", end="\r")
                times_read.append(bench.benchmark_reads(read_order=read_order))
            print(f"Time for reading the data with {args.io_type} in {read_order} order: {min(times_read):.3f}s")
