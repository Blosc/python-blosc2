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
        self.shape = (self.n_chunks, 64, 64, 64)
        self.chunks = (1, 64, 64, 64)
        self.blocks = (1, 16, 64, 64)
        self.dtype = np.dtype(np.float32)
        self.size = np.prod(self.shape)
        self.nbytes = self.size * self.dtype.itemsize
        self.array = np.arange(self.size, dtype=self.dtype).reshape(self.shape)
        self.cparams = dict(typesize=self.dtype.itemsize, clevel=0)
        self.cdata = blosc2.asarray(self.array, chunks=self.chunks, blocks=self.blocks,
                                    cparams=self.cparams)

    def __enter__(self):
        blosc2.remove_urlpath(self.urlpath)
        np.random.seed(42)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        blosc2.remove_urlpath(self.urlpath)

    def benchmark_writes(self) -> float:
        array = self.array
        urlpath = None if self.io_type == "io_mem" else self.urlpath

        if self.blosc_mode == "schunk":
            chunksize = array[0].nbytes
            cparams = self.cparams | dict(blocksize=np.prod(self.blocks) * array.itemsize)
            schunk = blosc2.SChunk(chunksize=chunksize, cparams=cparams,
                                   mode="w", mmap_mode=self.mmap_mode_write,
                                   urlpath=urlpath)

            t0 = time()
            for c in range(self.n_chunks):
                schunk.append_data(array[c])
            t1 = time()
        elif self.blosc_mode == "ndarray":
            t0 = time()
            blosc2.asarray(array, chunks=self.chunks, blocks=self.blocks,
                           cparams=self.cparams, mode="w",
                           mmap_mode=self.mmap_mode_write, urlpath=urlpath)
            t1 = time()
        else:
            raise ValueError(f"Unknown Blosc mode: {self.blosc_mode}")

        return t1 - t0

    def benchmark_reads(self, read_order: str = "sequential") -> float:
        if self.io_type == "io_mem":
            cdata = self.cdata.schunk if self.blosc_mode == "schunk" else self.cdata
        else:
            cdata = blosc2.open(self.urlpath, mmap_mode=self.mmap_mode_read)

        chunks_order = np.arange(self.n_chunks)
        if read_order == "random":
            np.random.shuffle(chunks_order)

        if self.blosc_mode == "schunk":
            t0 = time()
            for c in chunks_order:
                cdata.decompress_chunk(c)
            t1 = time()
        elif self.blosc_mode == "ndarray":
            t0 = time()
            for c in chunks_order:
                _ = cdata[c]
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
        choices=["io_file", "io_mmap", "io_mem"],
        help="Basic I/O type: default file operations (io_file),"
             " memory-mapped files (io_mmap) or fully in-memory (io_mem).",
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
        min_time = min(times_write)
        speed = bench.nbytes / min_time / 2**30
        print(f"Time for writing the data with {args.io_type}: {min_time:.3f} s ({speed:.3f} GB/s)")

        for read_order in ["sequential", "random"]:
            times_read = []
            for i in range(args.runs):
                print(f"Run {i+1}/{args.runs}", end="\r")
                times_read.append(bench.benchmark_reads(read_order=read_order))
            min_time = min(times_read)
            speed = bench.nbytes / min_time / 2**30
            print(f"Time for reading the data with {args.io_type} in {read_order} order: {min_time:.3f} s"
                  f" ({speed:.3f} GB/s)")
