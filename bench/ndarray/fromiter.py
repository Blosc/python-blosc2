#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Benchmark for blosc2.fromiter() — Phase 3 performance baseline.

Covers the three Phase 3 tuning axes:

  1. Chunk buffer allocation / reuse
       Varies chunk shapes for a fixed total array size to expose allocation
       overhead per chunk and the cost of many small vs. few large chunks.

  2. Chunk traversal strategies
       Compares c_order=True (full in-memory buffer) vs c_order=False
       (streaming chunk-by-chunk) for the same multidimensional array.

  3. On-disk vs. in-memory targets
       Runs each case with and without a urlpath so that I/O overhead can be
       separated from construction overhead.

Usage::

    python bench/ndarray/fromiter.py               # default: in-memory only
    python bench/ndarray/fromiter.py --on-disk      # also run on-disk cases
    python bench/ndarray/fromiter.py --nreps 5      # more repetitions
    python bench/ndarray/fromiter.py --dtype float32
    python bench/ndarray/fromiter.py --help
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import time

import numpy as np

import blosc2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_iterator(total: int, dtype: np.dtype):
    """Return a fresh generator of *total* values cast to *dtype*."""
    # Use a generator so the iterable is one-shot (stress-tests the
    # implementation's single-pass contract).
    return (dtype.type(i % 1000) for i in range(total))


def measure(fn, nreps: int) -> tuple[float, float]:
    """Run *fn* *nreps* times and return (best, mean) wall-clock seconds."""
    times = []
    for _ in range(nreps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def array_info(a: blosc2.NDArray) -> str:
    nb = a.schunk.nbytes
    cb = a.schunk.cbytes
    return (
        f"{nb / 2**20:8.1f} MB uncompressed  "
        f"cratio {nb / cb:4.1f}x  "
        f"({cb / 2**20:.1f} MB on storage)"
    )


def print_result(label: str, best: float, mean: float, nbytes: int) -> None:
    gb = nbytes / 2**30
    print(
        f"  {label:<45s}  best {best:.3f}s ({gb / best:.2f} GB/s)"
        f"  mean {mean:.3f}s"
    )


def cleanup(urlpath: str | None) -> None:
    if urlpath is None:
        return
    if os.path.isdir(urlpath):
        shutil.rmtree(urlpath)
    elif os.path.exists(urlpath):
        os.remove(urlpath)


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------

def bench_chunk_sizes(dtype: np.dtype, nreps: int, on_disk: bool) -> None:
    """
    Section 1 — Chunk buffer allocation / reuse (optimisation A).

    Fixed total size, varying chunk shapes.  Exposes per-chunk allocation
    overhead: many tiny chunks vs. a few large chunks, and shows the impact
    of the page buffer on c_order=False.
    """
    print("\n" + "=" * 70)
    print("Section 1 — Chunk buffer allocation / reuse (opt A: page buffer)")
    print(f"  Fixed shape (1000, 1000), dtype={dtype}, nreps={nreps}")
    print("=" * 70)

    shape = (1000, 1000)
    total = math.prod(shape)
    nbytes = total * dtype.itemsize

    chunk_configs = [
        # (chunks, blocks, label)
        ((10, 10),    (5, 5),     "chunks=(10,10)    — many tiny"),
        ((50, 50),    (25, 25),   "chunks=(50,50)    — medium"),
        ((100, 100),  (50, 50),   "chunks=(100,100)  — medium-large"),
        ((200, 200),  (100, 100), "chunks=(200,200)  — large"),
        ((500, 500),  (250, 250), "chunks=(500,500)  — very large"),
        ((1000, 100), (500, 50),  "chunks=(1000,100) — full-row strip"),
        ((1000, 1000),(500, 500), "chunks=shape      — single chunk"),
    ]

    for order_label, c_order in (("c_order=True ", True), ("c_order=False", False)):
        print(f"\n  {order_label}")
        for chunks, blocks, clabel in chunk_configs:
            urlpath = "fromiter_bench.b2nd" if on_disk else None

            def run(c=chunks, b=blocks, u=urlpath, co=c_order):
                cleanup(u)
                blosc2.fromiter(
                    make_iterator(total, dtype),
                    shape=shape, dtype=dtype,
                    chunks=c, blocks=b,
                    c_order=co,
                    urlpath=u, mode="w" if u else None,
                )

            best, mean = measure(run, nreps)
            cleanup(urlpath)
            disk_tag = " [disk]" if on_disk else ""
            print_result(f"{clabel}{disk_tag}", best, mean, nbytes)


def bench_corder(dtype: np.dtype, nreps: int, on_disk: bool) -> None:
    """
    Section 2 — Chunk traversal strategies: c_order=True vs c_order=False.

    Runs the same shapes/chunk configs with both orderings so that the
    trade-off between in-memory buffering and streaming chunk fill is visible.
    """
    print("\n" + "=" * 70)
    print("Section 2 — Chunk traversal: c_order=True vs c_order=False")
    print(f"  dtype={dtype}, nreps={nreps}")
    print("=" * 70)

    cases = [
        # (shape, chunks, blocks, label)
        ((500, 500),        (50, 50),    (25, 25),   "2-D  (500,500)   chunks=(50,50)"),
        ((200, 200, 200),   (20, 20, 20),(10, 10, 10),"3-D  (200,200,200) chunks=(20,20,20)"),
        ((50, 50, 50, 50),  (10, 10, 10, 10),(5,5,5,5),"4-D  (50,50,50,50) chunks=(10,10,10,10)"),
    ]

    for shape, chunks, blocks, label in cases:
        total = math.prod(shape)
        nbytes = total * dtype.itemsize
        print(f"\n  {label}  [{nbytes / 2**20:.1f} MB]")

        for order_label, c_order in (("c_order=True ", True), ("c_order=False", False)):
            for disk_label, use_disk in (("in-memory", False), ("on-disk  ", True)):
                if use_disk and not on_disk:
                    continue
                urlpath = "fromiter_bench.b2nd" if use_disk else None

                def run(s=shape, c=chunks, b=blocks, u=urlpath, co=c_order):
                    cleanup(u)
                    blosc2.fromiter(
                        make_iterator(total, dtype),
                        shape=s, dtype=dtype,
                        chunks=c, blocks=b,
                        c_order=co,
                        urlpath=u, mode="w" if u else None,
                    )

                best, mean = measure(run, nreps)
                cleanup(urlpath)
                print_result(f"  {order_label}  {disk_label}", best, mean, nbytes)


def bench_ondisk_vs_memory(dtype: np.dtype, nreps: int) -> None:
    """
    Section 3 — On-disk vs. in-memory targets.

    Side-by-side comparison for a large-ish array so that I/O overhead
    is clearly separated from construction cost.
    """
    print("\n" + "=" * 70)
    print("Section 3 — On-disk vs. in-memory")
    print(f"  dtype={dtype}, nreps={nreps}")
    print("=" * 70)

    shape = (2000, 2000)
    chunks = (200, 200)
    blocks = (100, 100)
    total = math.prod(shape)
    nbytes = total * dtype.itemsize
    print(f"  shape={shape}  chunks={chunks}  [{nbytes / 2**20:.1f} MB]")

    for order_label, c_order in (("c_order=True ", True), ("c_order=False", False)):
        print(f"\n  {order_label}")
        for disk_label, urlpath in (("in-memory", None), ("on-disk  ", "fromiter_bench.b2nd")):

            def run(u=urlpath, co=c_order):
                cleanup(u)
                a = blosc2.fromiter(
                    make_iterator(total, dtype),
                    shape=shape, dtype=dtype,
                    chunks=chunks, blocks=blocks,
                    c_order=co,
                    urlpath=u, mode="w" if u else None,
                )
                return a

            best, mean = measure(run, nreps)
            cleanup(urlpath)
            print_result(f"  {disk_label}", best, mean, nbytes)


def bench_large(dtype: np.dtype, nreps: int, on_disk: bool) -> None:
    """
    Bonus — large array for headline throughput numbers.

    Includes the numpy fast path (optimisation C) when the iterable is
    already a numpy array, which completely bypasses Python iteration.
    """
    print("\n" + "=" * 70)
    print("Bonus — Large array headline throughput (opt C: numpy fast path)")
    print(f"  dtype={dtype}, nreps={nreps}")
    print("=" * 70)

    shape = (5000, 5000)
    chunks = (500, 500)
    blocks = (250, 250)
    total = math.prod(shape)
    nbytes = total * dtype.itemsize
    print(f"  shape={shape}  [{nbytes / 2**20:.0f} MB]")

    # NumPy baseline (pure Python generator)
    def np_run():
        np.fromiter(make_iterator(total, dtype), dtype=dtype, count=total).reshape(shape)

    best, mean = measure(np_run, nreps)
    print_result("  NumPy fromiter+reshape (generator baseline)", best, mean, nbytes)

    # blosc2 with generator
    for order_label, c_order in (("c_order=True ", True), ("c_order=False", False)):
        for disk_label, use_disk in (("in-memory", False), ("on-disk  ", True)):
            if use_disk and not on_disk:
                continue
            urlpath = "fromiter_bench_large.b2nd" if use_disk else None

            def run(s=shape, c=chunks, b=blocks, u=urlpath, co=c_order):
                cleanup(u)
                blosc2.fromiter(
                    make_iterator(total, dtype),
                    shape=s, dtype=dtype,
                    chunks=c, blocks=b,
                    c_order=co,
                    urlpath=u, mode="w" if u else None,
                )

            best, mean = measure(run, nreps)
            cleanup(urlpath)
            print_result(f"  blosc2 generator  {order_label} {disk_label}", best, mean, nbytes)

    # Optimisation C: numpy fast path — iterable is already an ndarray
    print()
    src = np.fromiter(make_iterator(total, dtype), dtype=dtype, count=total).reshape(shape)
    for disk_label, use_disk in (("in-memory", False), ("on-disk  ", True)):
        if use_disk and not on_disk:
            continue
        urlpath = "fromiter_bench_large.b2nd" if use_disk else None

        def run_np(s=shape, c=chunks, b=blocks, u=urlpath, arr=src):
            cleanup(u)
            blosc2.fromiter(arr, shape=s, dtype=dtype, chunks=c, blocks=b,
                            urlpath=u, mode="w" if u else None)

        best, mean = measure(run_np, nreps)
        cleanup(urlpath)
        print_result(f"  blosc2 ndarray fast path         {disk_label}", best, mean, nbytes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dtype", default="float64", help="NumPy dtype (default: float64)")
    p.add_argument("--nreps", type=int, default=3, help="Repetitions per measurement (default: 3)")
    p.add_argument(
        "--on-disk",
        action="store_true",
        default=False,
        help="Also run on-disk cases (writes temporary .b2nd files)",
    )
    p.add_argument("--section", type=int, default=0,
                   help="Run only section N (1-3 + bonus=4); 0 = all (default: 0)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = np.dtype(args.dtype)
    nreps = args.nreps
    on_disk = args.on_disk

    print(f"\nblosc2.fromiter() benchmark — dtype={dtype}  nreps={nreps}  on_disk={on_disk}")
    print(f"blosc2 version: {blosc2.__version__}")

    sections = {
        1: lambda: bench_chunk_sizes(dtype, nreps, on_disk),
        2: lambda: bench_corder(dtype, nreps, on_disk),
        3: lambda: bench_ondisk_vs_memory(dtype, nreps) if on_disk else print(
            "\nSection 3 skipped (use --on-disk to enable)"
        ),
        4: lambda: bench_large(dtype, nreps, on_disk),
    }

    if args.section == 0:
        for fn in sections.values():
            fn()
    else:
        sections[args.section]()

    print()


if __name__ == "__main__":
    main()
