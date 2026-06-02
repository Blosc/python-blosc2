#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################
"""
Benchmark ``take()`` / fancy indexing across numpy, blosc2, zarr, and h5py.

Usage::

    python bench/ndarray/take.py --ndim 2 --arr-size 100000000 --output take_2d.png

The script creates an array of *arr-size* elements with *ndim* dimensions,
then measures the time to gather a log-spaced range of random indices
(1 – 100 K).  numpy is kept in-memory; blosc2, zarr and h5py use on-disk
storage so the benchmark reflects I/O behaviour of compressed backends.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
import time as _time
from pathlib import Path

import h5py
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import psutil
import zarr
from zarr.codecs import BloscCodec, BytesCodec

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

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _compute_shape(ndim: int, n_elements: int) -> tuple[int, ...]:
    """Roughly-cubic shape with the given number of dimensions."""
    d = int(round(n_elements ** (1.0 / ndim)))
    shape = [d] * ndim
    # tweak the first dimension so total elements ≈ n_elements
    shape[0] = max(1, n_elements // int(np.prod(shape[1:])))
    return tuple(shape)

# ---------------------------------------------------------------------------
# array creation
# ---------------------------------------------------------------------------

def _chunks(shape):
    """Chunk shape used by all backends (~1/4 of each dimension)."""
    return tuple(max(s // 4, 1) for s in shape)


def create_arrays(shape, dtype=np.float64, del_source=False):
    """Create arrays for all four libraries in a shared temp directory."""
    n_elements = np.prod(shape)
    data = np.arange(n_elements, dtype=dtype).reshape(shape)

    tmpdir = Path(tempfile.mkdtemp(prefix="take_bench_"))
    chunks = _chunks(shape)

    # --- blosc2 ---------------------------------------------------------
    t0 = time.time()
    b2path = tmpdir / "data.b2nd"
    a_b2 = blosc2.asarray(data, chunks=chunks, urlpath=str(b2path),
                           cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5})
    print(f"Shape: {shape}  |  n_elements: {n_elements:_}  "
          f"|  itemsize: {data.itemsize}  |  total: {data.nbytes / 1e9:.2f} GB")
    print(f"Chunks: {chunks}  |  Blocks: {a_b2.blocks}")
    print(f"Tmp dir: {tmpdir}")
    print(f"blosc2 created in {time.time() - t0:.2f}s  "
          f"cratio={a_b2.schunk.cratio:.1f}x  "
          f"cbytes={a_b2.schunk.cbytes / 1e6:.1f} MB")
    print()

    # --- numpy ----------------------------------------------------------
    a_np = data.copy()

    # --- zarr -----------------------------------------------------------
    t0 = time.time()
    zpath = tmpdir / "data.zarr"
    a_z = zarr.open_array(str(zpath), mode="w", shape=shape, dtype=dtype, chunks=chunks,
                           codecs=[BytesCodec(),
                                   BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")])
    a_z[:] = data
    print(f"zarr  created in {time.time() - t0:.2f}s")

    # --- h5py ----------------------------------------------------------
    t0 = time.time()
    h5path = tmpdir / "data.h5"
    h5f = h5py.File(str(h5path), "w")
    a_h5 = h5f.create_dataset("data", data=data, chunks=chunks,
                               **hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=1))
    print(f"h5py  created in {time.time() - t0:.2f}s")
    print()

    if del_source:
        del data

    return a_b2, a_np, a_z, a_h5, tmpdir


# ---------------------------------------------------------------------------
# benchmark runner
# ---------------------------------------------------------------------------



def _peak_memory(func, *args, **kwargs):
    """Return RSS memory increase (MB) after *func(*args, **kwargs).

    The output of *func* is held alive during measurement so its
    allocations are reflected in the post-call RSS.
    Returns the maximum of two measurements:
    1. Peak RSS observed by a background sampler (catches transient C malloc).
    2. Post-call RSS delta (catches retained output arrays).
    """
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
    _ = result  # keep alive so retained output is counted
    delta_peak = (peak[0] - before) / (1024 * 1024)
    delta_after = (after - before) / (1024 * 1024)
    return max(delta_peak, delta_after)


def _select_indices(rng, size, n_indices):
    """Return a sorted, unique 1-D int64 array of ~*n_indices* random indices.

    Indices are sorted and deduplicated so that h5py (which requires
    strictly increasing order) can participate fairly."""
    idx = np.unique(rng.integers(0, size, size=n_indices, dtype=np.int64))
    return idx


def run_benchmark(a_b2, a_np, a_z, a_h5, ndim, n_runs=3, sparse=False,
                  profile_mem=False):
    """Run the fancy-indexing benchmark for a range of index counts."""
    shape = a_np.shape
    size = a_np.size if sparse else shape[0]  # flat size for sparse, axis-0 for orthogonal
    max_indices = min(100_000, size)

    n_indices_list = np.unique(
        np.logspace(0, np.log10(max(1, max_indices)), num=12, dtype=np.int64)
    )
    print(f"Index counts: {n_indices_list.tolist()}")

    if profile_mem:
        print("(memory-profiling mode, 1 run per point)")
        print()

    rng = np.random.default_rng(42)

    results = {
        "numpy": [],
        "blosc2": [],
        "zarr": [],
        "h5py": [],
    }
    actual_counts = []

    for n_idx in n_indices_list:
        idx = _select_indices(rng, size, int(n_idx))
        n_actual = len(idx)  # may be less after dedup

        if profile_mem:
            # --- memory profiling ---------------------------------------
            if sparse:
                # zarr/h5py lack sparse gather — measure full-read + np.take
                def _b2():
                    return blosc2.take(a_b2, idx, axis=None)[:]
                def _np():
                    return np.take(a_np, idx, axis=None)
                def _zarr():
                    return np.take(a_z[:], idx, axis=None)
                def _h5():
                    return np.take(a_h5[:], idx, axis=None)
            else:
                def _b2():
                    return blosc2.take(a_b2, idx, axis=0)[:]
                def _np():
                    return np.take(a_np, idx, axis=0)
                def _zarr():
                    if ndim == 1:
                        return a_z.oindex[(idx,)]
                    sel = (idx,) + (slice(None),) * (ndim - 1)
                    return a_z.oindex[sel]
                def _h5():
                    sel = (idx.tolist(),) + (slice(None),) * (ndim - 1)
                    return a_h5[sel]

            results["numpy"].append(_peak_memory(_np))
            results["blosc2"].append(_peak_memory(_b2))
            results["zarr"].append(_peak_memory(_zarr))
            results["h5py"].append(_peak_memory(_h5))

            print(
                f"  n_indices={n_actual:>7}: "
                f"numpy={results['numpy'][-1]:.1f} MB  "
                f"blosc2={results['blosc2'][-1]:.1f} MB  "
                f"zarr={results['zarr'][-1]:.1f} MB  "
                f"h5py={results['h5py'][-1]:.1f} MB"
            )
            actual_counts.append(n_actual)
            continue
        if sparse:
            # --- sparse path (axis=None, flat element gather) -------------
            # numpy
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = a_np.flat[idx]
                elapsed.append(time.perf_counter() - t0)
            results["numpy"].append(np.min(elapsed))

            # blosc2 — uses b2nd_get_sparse_cbuffer
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = blosc2.take(a_b2, idx, axis=None)[:]
                elapsed.append(time.perf_counter() - t0)
            results["blosc2"].append(np.min(elapsed))

            # zarr — no native sparse; full read + numpy.take
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = np.take(a_z[:], idx, axis=None)
                elapsed.append(time.perf_counter() - t0)
            results["zarr"].append(np.min(elapsed))

            # h5py — no native sparse; full read + numpy.take
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = np.take(a_h5[:], idx, axis=None)
                elapsed.append(time.perf_counter() - t0)
            results["h5py"].append(np.min(elapsed))
        else:
            # --- orthogonal path (axis=0, row/slab selection) -------------
            # numpy
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = a_np[idx]
                elapsed.append(time.perf_counter() - t0)
            results["numpy"].append(np.min(elapsed))

            # blosc2 — __getitem__ → _try_sparse_fancy_index → _take_numpy
            elapsed = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = a_b2[idx]
                elapsed.append(time.perf_counter() - t0)
            results["blosc2"].append(np.min(elapsed))

            # zarr
            elapsed = []
            if ndim == 1:
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = a_z.oindex[(idx,)]
                    elapsed.append(time.perf_counter() - t0)
            else:
                sel = (idx,) + (slice(None),) * (ndim - 1)
                for _ in range(n_runs):
                    t0 = time.perf_counter()
                    _ = a_z.oindex[sel]
                    elapsed.append(time.perf_counter() - t0)
            results["zarr"].append(np.min(elapsed))

            # h5py
            elapsed = []
            sel = (idx.tolist(),) + (slice(None),) * (ndim - 1)
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = a_h5[sel]
                elapsed.append(time.perf_counter() - t0)
            results["h5py"].append(np.min(elapsed))

        print(
            f"  n_indices={n_actual:>7}: "
            f"numpy={results['numpy'][-1]:.4f}s  "
            f"blosc2={results['blosc2'][-1]:.4f}s  "
            f"zarr={results['zarr'][-1]:.4f}s  "
            f"h5py={results['h5py'][-1]:.4f}s"
        )
        actual_counts.append(n_actual)

    return np.array(actual_counts), results


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

COLORS = {"numpy": "#1f77b4", "blosc2": "#ff7f0e", "zarr": "#2ca02c", "h5py": "#d62728"}
MARKERS = {"numpy": "o", "blosc2": "s", "zarr": "^", "h5py": "D"}


def plot_results(n_indices, results, ndim, arr_size, output, sparse=False, profile_mem=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, times in results.items():
        ax.plot(
            n_indices, times, color=COLORS[label], marker=MARKERS[label],
            label=label, linewidth=2, markersize=7,
        )

    ax.set_xscale("log")
    if not profile_mem:
        ax.set_yscale("log")
    ax.set_xlabel("Number of indices")
    ax.set_ylabel("Peak memory (MB)" if profile_mem else "Time (s)")
    mode = "sparse" if sparse else "fancy-indexing"
    suffix = " — memory" if profile_mem else ""
    ax.set_title(f"{mode} benchmark{suffix} — ndim={ndim}, arr-size={arr_size:_}")
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
    p = argparse.ArgumentParser(description="Benchmark take() across numpy/blosc2/zarr/h5py")
    p.add_argument("--ndim", type=int, default=1, help="Number of dimensions (default: 1)")
    p.add_argument(
        "--arr-size", type=int, default=100_000_000,
        help="Total number of elements (default: 100M)",
    )
    p.add_argument("--output", type=str, default=None,
                   help="Path to save the plot (PNG). If omitted, the plot is shown.")
    p.add_argument("--sparse", action="store_true",
                   help="Use axis=None (flat element gather via b2nd_get_sparse_cbuffer).")
    p.add_argument("--profile-mem", action="store_true",
                   help="Measure peak memory (MB) per library (tracemalloc).  Skips numpy.")
    return p.parse_args()


def main():
    args = parse_args()
    shape = _compute_shape(args.ndim, args.arr_size)
    dtype = np.float64

    a_b2, a_np, a_z, a_h5, tmpdir = create_arrays(shape, dtype,
                                                      del_source=args.profile_mem)

    try:
        n_indices, results = run_benchmark(a_b2, a_np, a_z, a_h5, args.ndim,
                                            sparse=args.sparse,
                                            profile_mem=args.profile_mem)
        plot_results(n_indices, results, args.ndim, args.arr_size, args.output,
                     sparse=args.sparse, profile_mem=args.profile_mem)
    finally:
        # Cleanup temp files
        if tmpdir.exists():
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
