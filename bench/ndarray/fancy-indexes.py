#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
"""
Benchmark fancy indexing with a boolean array vs. a list of flat indices
(coords) on an in-memory blosc2.NDArray.

All approaches select the same elements (determined by the same set of
random flat indices), so the comparison reflects the overhead of each path.

Usage::

    python bench/ndarray/fancy-indexes.py --ndim 3 --arr-size 100000000

Optional flags::

    --ndim       Number of dimensions          (default: 3)
    --arr-size   Total number of elements      (default: 100_000_000)
    --max-idx    Maximum number of indices     (default: 100_000)
    --output     Save plot to PNG (optional, no display if set)
    --profile-mem  Measure peak memory instead of time

Benchmarked paths
------------------

* ``bool mask`` — ``a[bool_mask]`` with automatic sparse/dense detection.
* ``coord list`` — ``blosc2.take(a, coord_list, axis=None)[:]``
  (sparse-element gather via ``b2nd_get_sparse_cbuffer``).
* ``mask→coords`` — ``np.flatnonzero(bool_mask)`` + sparse gather.
* ``lazy expr`` — ``a[a < threshold][:]``, the idiomatic lazy-expression
  path (now auto-optimized internally via miniexpr + sparse take).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time as _time
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import psutil

import blosc2

# ---------------------------------------------------------------------------
# plot style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 14,
    "figure.dpi": 150,
    "savefig.dpi": 150,
})
plt.style.use("seaborn-v0_8-paper")

COLORS = {
    "bool mask": "#1f77b4",
    "coord list": "#ff7f0e",
    "mask→coords": "#2ca02c",
    "lazy expr": "#d62728",
}
MARKERS = {
    "bool mask": "o",
    "coord list": "s",
    "mask→coords": "^",
    "lazy expr": "D",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _compute_shape(ndim: int, n_elements: int) -> tuple[int, ...]:
    """Roughly-cubic shape with the given number of dimensions."""
    d = int(round(n_elements ** (1.0 / ndim)))
    shape = [d] * ndim
    shape[0] = max(1, n_elements // int(np.prod(shape[1:])))
    return tuple(shape)


def _peak_memory(func, *args, **kwargs):
    """Return RSS memory increase (MB) after *func(*args, **kwargs)."""
    proc = psutil.Process()
    before = proc.memory_info().rss
    peak = [before]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            rss = proc.memory_info().rss
            if rss > peak[0]:
                peak[0] = rss
            _time.sleep(0.001)

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    result = func(*args, **kwargs)
    stop.set()
    t.join(timeout=0.1)

    after = proc.memory_info().rss
    _ = result  # keep alive to count retained output
    delta_peak = (peak[0] - before) / (1024 * 1024)
    delta_after = (after - before) / (1024 * 1024)
    return max(delta_peak, delta_after)


def _make_bool_mask(shape, flat_indices):
    """Build a boolean array of *shape* with True at *flat_indices*."""
    mask = np.zeros(np.prod(shape), dtype=np.bool_)
    mask[flat_indices] = True
    return mask.reshape(shape)


# ---------------------------------------------------------------------------
# array creation
# ---------------------------------------------------------------------------

def create_array(shape):
    """Create an in-memory blosc2 linspace array."""
    n_elements = np.prod(shape)
    print(f"Shape: {shape}  |  n_elements: {n_elements:_}  "
          f"|  dtype: float64  |  total: {n_elements * 8 / 1e9:.2f} GB")
    t0 = perf_counter()
    a = blosc2.linspace(0.0, 1.0, int(n_elements), shape=shape)
    t = perf_counter() - t0
    print(f"blosc2.linspace created in {t:.2f}s  "
          f"cratio={a.schunk.cratio:.1f}x  "
          f"cbytes={a.schunk.cbytes / 1e6:.1f} MB")
    print()
    return a


# ---------------------------------------------------------------------------
# benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(a, ndim, max_idx=100_000, n_runs=3, profile_mem=False):
    """Compare bool-mask, coord-list, mask→coords, and lazy-expr indexing."""
    n_elements = a.size
    max_idx = min(max_idx, n_elements)

    n_indices_list = np.unique(
        np.logspace(0, np.log10(max(1, max_idx)), num=12, dtype=np.int64)
    )
    print(f"Index counts: {n_indices_list.tolist()}")
    if profile_mem:
        print("(memory-profiling mode, 1 run per point)")
    print()

    rng = np.random.default_rng(42)
    results = {"bool mask": [], "coord list": [], "mask→coords": [], "lazy expr": []}
    actual_counts = []

    for n_idx in n_indices_list:
        flat_idx = np.unique(rng.integers(0, n_elements, size=int(n_idx)))
        n_actual = len(flat_idx)

        bool_mask = _make_bool_mask(a.shape, flat_idx)
        coord_list = flat_idx.tolist()

        # Lazy-expr threshold: use selectivity to get ~n_actual matches
        # (linspace is uniform on [0, 1], so a < n_actual / n_elements)
        threshold = n_actual / n_elements if n_actual > 0 else 0.0

        if profile_mem:
            def _bool():
                return a[bool_mask]

            def _coords():
                return blosc2.take(a, coord_list, axis=None)[:]

            def _mask_to_coords():
                idx = np.flatnonzero(bool_mask)
                return blosc2.take(a, idx, axis=None)[:]

            def _lazy():
                return a[a < threshold][:]

            mem_bool = _peak_memory(_bool)
            mem_coords = _peak_memory(_coords)
            mem_m2c = _peak_memory(_mask_to_coords)
            mem_lazy = _peak_memory(_lazy)

            results["bool mask"].append(mem_bool)
            results["coord list"].append(mem_coords)
            results["mask→coords"].append(mem_m2c)
            results["lazy expr"].append(mem_lazy)
            print(
                f"  n_indices={n_actual:>7}: "
                f"bool_mask={mem_bool:.1f} MB  "
                f"coord_list={mem_coords:.1f} MB  "
                f"mask→coords={mem_m2c:.1f} MB  "
                f"lazy_expr={mem_lazy:.1f} MB"
            )
        else:
            # --- bool mask ---
            times_bool = []
            for _ in range(n_runs):
                t0 = perf_counter()
                _ = a[bool_mask]
                times_bool.append(perf_counter() - t0)
            t_bool = np.min(times_bool)

            # --- coord list ---
            times_coords = []
            for _ in range(n_runs):
                t0 = perf_counter()
                _ = blosc2.take(a, coord_list, axis=None)[:]
                times_coords.append(perf_counter() - t0)
            t_coords = np.min(times_coords)

            # --- mask → coords ---
            times_m2c = []
            for _ in range(n_runs):
                t0 = perf_counter()
                idx = np.flatnonzero(bool_mask)
                _ = blosc2.take(a, idx, axis=None)[:]
                times_m2c.append(perf_counter() - t0)
            t_m2c = np.min(times_m2c)

            # --- lazy expr ---
            times_lazy = []
            for _ in range(n_runs):
                t0 = perf_counter()
                _ = a[a < threshold][:]
                times_lazy.append(perf_counter() - t0)
            t_lazy = np.min(times_lazy)

            results["bool mask"].append(t_bool)
            results["coord list"].append(t_coords)
            results["mask→coords"].append(t_m2c)
            results["lazy expr"].append(t_lazy)
            print(
                f"  n_indices={n_actual:>7}: "
                f"bool_mask={t_bool:.5f}s  "
                f"coord_list={t_coords:.5f}s  "
                f"mask→coords={t_m2c:.5f}s  "
                f"lazy_expr={t_lazy:.5f}s"
            )

        actual_counts.append(n_actual)

    return np.array(actual_counts), results


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

def plot_results(n_indices, results, ndim, arr_size, output, profile_mem=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, times in results.items():
        ax.plot(
            n_indices, times, color=COLORS[label], marker=MARKERS[label],
            label=label, linewidth=2, markersize=7,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of selected elements")
    if not profile_mem:
        ax.set_yscale("log")
    ax.set_ylabel("Peak memory (MB)" if profile_mem else "Time (s)")
    title = (
        f"Bool mask vs coord list fancy indexing — "
        f"ndim={ndim}, arr-size={arr_size:_}"
    )
    if profile_mem:
        title += " (memory)"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if output:
        fig.savefig(output)
        print(f"\nPlot saved to {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark bool-mask fancy indexing vs coord-list sparse read"
    )
    p.add_argument(
        "--ndim", type=int, default=3,
        help="Number of dimensions (default: 3)",
    )
    p.add_argument(
        "--arr-size", type=int, default=100_000_000,
        help="Total number of elements (default: 100_000_000)",
    )
    p.add_argument(
        "--max-idx", type=int, default=100_000,
        help="Maximum number of indices to test (default: 100_000)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Save plot to this path (PNG). If omitted, display interactively.",
    )
    p.add_argument(
        "--profile-mem", action="store_true",
        help="Measure peak memory (MB) instead of timing.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print(f"blosc2 version: {blosc2.__version__}")
    print(f"numpy  version: {np.__version__}")
    print(f"C-Blosc2 version: {blosc2.blosclib_version}")
    print()

    shape = _compute_shape(args.ndim, args.arr_size)
    print(f"Using ndim={args.ndim}, arr-size={args.arr_size:_}  ->  shape={shape}")

    a = create_array(shape)

    n_indices, results = run_benchmark(
        a, args.ndim, max_idx=args.max_idx, profile_mem=args.profile_mem
    )

    plot_results(
        n_indices, results, args.ndim, args.arr_size,
        args.output, profile_mem=args.profile_mem,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
