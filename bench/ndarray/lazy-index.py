#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
"""
Profile and benchmark ``a[bool_array]`` on a blosc2 NDArray.

Compares the lazy path ``a[a < threshold][:]`` against the concrete
boolean-array path ``a[bool_arr]`` and breaks down where the time goes.

Usage::

    python bench/ndarray/lazy-index2.py

Optional flags::

    --ndim       Number of dimensions          (default: 2)
    --arr-size   Total number of elements      (default: 100_000_000)
    --threshold  Filter condition value        (default: 5)
"""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

import blosc2

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compute_shape(ndim: int, n_elements: int) -> tuple[int, ...]:
    d = int(round(n_elements ** (1.0 / ndim)))
    shape = [d] * ndim
    shape[0] = max(1, n_elements // int(np.prod(shape[1:])))
    return tuple(shape)


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------


def profile_lazy_index(ndim, arr_size, threshold):
    print(f"{'=' * 60}")
    print(f"ndim={ndim}, arr-size={arr_size:_}, threshold={threshold}")
    print(f"{'=' * 60}")
    print()

    shape = _compute_shape(ndim, arr_size)
    n_elements = np.prod(shape)

    # --- create array ----------------------------------------------------
    t0 = perf_counter()
    a = blosc2.arange(0, n_elements, shape=shape)
    t_create = perf_counter() - t0
    print(f"Array shape:         {shape}")
    print(f"Total elements:      {n_elements:_}")
    print(f"Uncompressed size:   {a.nbytes / 1e9:.2f} GB")
    print(f"Chunks:              {a.chunks}")
    print(f"Number of chunks:    {a.schunk.nchunks}")
    print(f"Create time:         {t_create:.3f}s")
    print()

    # --- path 1: a[a < threshold][:]  (lazy expression) ------------------
    t0 = perf_counter()
    result = a[a < threshold][:]
    t_lazy = perf_counter() - t0

    # --- path 2: bool_array = (a < threshold).compute() ; a[bool_array] --
    t0 = perf_counter()
    bool_arr = (a < threshold).compute()
    t_bool_compute = perf_counter() - t0

    t0 = perf_counter()
    result2 = a[bool_arr]
    t_concrete = perf_counter() - t0

    t_total_bool = t_bool_compute + t_concrete

    print(f"{'--- Path comparison ---':^50}")
    print(f"{'Path':<35} {'Time (ms)':<15}")
    print(f"{'-' * 50}")
    print(f"{'a[a < threshold][:]  (lazy)':<35} {t_lazy * 1000:<15.1f}")
    print("")
    print(f"{'  (a<threshold).compute()':<35} {t_bool_compute * 1000:<15.1f}")
    print(f"{'  a[bool_arr]  (concrete)':<35} {t_concrete * 1000:<15.1f}")
    print(f"{'  total (bool path)':<35} {t_total_bool * 1000:<15.1f}")
    print()

    assert np.array_equal(result, result2), "Results must match"

    # --- per-chunk breakdown ----------------------------------------------
    nchunks = a.schunk.nchunks
    chunks = a.chunks
    import numexpr

    # Decompress one chunk
    t0 = perf_counter()
    for i in range(min(nchunks, 10)):
        raw = a.schunk.decompress_chunk(i)
        _ = np.frombuffer(raw, dtype=a.dtype).reshape(a.chunks)
    t_dec = (perf_counter() - t0) / min(nchunks, 10)

    # Decompress + numexpr eval "(x < threshold)"
    t0 = perf_counter()
    for i in range(min(nchunks, 10)):
        raw = a.schunk.decompress_chunk(i)
        chunk = np.frombuffer(raw, dtype=a.dtype).reshape(a.chunks)
        _ = numexpr.evaluate("(x < threshold)", {"x": chunk, "threshold": threshold})
    t_dec_ne = (perf_counter() - t0) / min(nchunks, 10)

    # Slice numpy bool array + decompress + gather
    slices_ = []
    for i in range(nchunks):
        coords = np.unravel_index(i, tuple(np.ceil(np.array(shape) / np.array(chunks)).astype(int)))
        s = tuple(
            slice(c * ch, min((c + 1) * ch, shape[d])) for d, (c, ch) in enumerate(zip(coords, chunks))
        )
        slices_.append(s)

    t0 = perf_counter()
    for _ in range(min(nchunks, 10)):
        for s in slices_:
            mask_chunk = bool_arr[s]
            data_chunk = a[s]
            _ = data_chunk[mask_chunk]
    t_bool_gather = (perf_counter() - t0) / min(nchunks, 10) / nchunks

    # Decompress + eval + gather (lazy path per-chunk)
    t0 = perf_counter()
    for i in range(min(nchunks, 10)):
        raw = a.schunk.decompress_chunk(i)
        chunk = np.frombuffer(raw, dtype=a.dtype).reshape(a.chunks)
        mask = numexpr.evaluate("(x < threshold)", {"x": chunk, "threshold": threshold})
        _ = chunk[mask]
    t_dec_ne_gather = (perf_counter() - t0) / min(nchunks, 10)

    print(f"{'--- Per-chunk breakdown (×{nchunks} chunks) ---':^65}")
    print(f"{'Operation':<40} {'per chunk':<12} {'×' + str(nchunks) + ' total':<15}")
    print(f"{'-' * 67}")
    print(f"{'decompress_chunk':<40} {t_dec * 1e6:>8.0f} µs  {t_dec * nchunks * 1000:>8.1f} ms")
    print(
        f"{'decompress + numexpr eval':<40} {t_dec_ne * 1e6:>8.0f} µs  {t_dec_ne * nchunks * 1000:>8.1f} ms"
    )
    print(
        f"{'slice bool + decompress + gather':<40} {t_bool_gather * 1e6:>8.0f} µs  {t_bool_gather * nchunks * 1000:>8.1f} ms"
    )
    print(
        f"{'decompress + eval + gather (lazy)':<40} {t_dec_ne_gather * 1e6:>8.0f} µs  {t_dec_ne_gather * nchunks * 1000:>8.1f} ms"
    )
    print()

    # --- hotspot analysis ------------------------------------------------
    print(f"{'--- Hotspot analysis ---':^50}")
    print()
    print(f"The lazy path (a[a<{threshold}][:]) fuses the comparison into the")
    print("chunk evaluation, calling numexpr on the decompressed chunk data.")
    print()
    print("The concrete boolean path (a[bool_arr]) was previously ~8× slower")
    print("because NDArray.__getitem__ called process_key() which invokes")
    print(f"np.nonzero() on the boolean array, scanning all {n_elements:_} elements")
    print("and allocating index arrays — work that was immediately discarded.")
    print()
    print("With the fix (bool array check moved before process_key), the")
    print("boolean path now takes the same fast LazyExpr route as the lazy path.")
    print()

    print(f"{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print()
    print(f"  Query (lazy):                     a[a < {threshold}][:]")
    print(f"  Query (concrete):                 a[bool_arr] with bool_arr = (a<{threshold}).compute()")
    print(
        f"  Matching elements:                {result.size} / {n_elements:_} ({result.size / n_elements * 100:.5f}%)"
    )
    print(f"  Lazy path time:                   {t_lazy * 1000:.1f} ms")
    print(f"  Concrete path time:               {t_concrete * 1000:.1f} ms")
    print(f"  Ratio (concrete/lazy):            {t_concrete / t_lazy:.1f}x")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Profile concrete boolean array indexing")
    p.add_argument("--ndim", type=int, default=2, help="Number of dimensions (default: 2)")
    p.add_argument(
        "--arr-size", type=int, default=100_000_000, help="Total number of elements (default: 100_000_000)"
    )
    p.add_argument("--threshold", type=float, default=5, help="Filter threshold value (default: 5)")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"blosc2 version: {blosc2.__version__}")
    print(f"numpy  version: {np.__version__}")
    print(f"C-Blosc2 version: {blosc2.blosclib_version}")
    print()
    profile_lazy_index(args.ndim, args.arr_size, args.threshold)
    print("Done!")


if __name__ == "__main__":
    main()
