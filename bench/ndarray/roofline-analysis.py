#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compute with different arithmetic intensities on NumPy/numexpr and blosc2
# This supports both in-memory and on-disk modes.  In-memory mode is the
# default.  If you want to run in on-disk mode, run this script with the
# command line argument "disk".

import math
import os
import pprint
import shutil
import sys
from time import time

import blosc2
import numexpr as ne
import numpy as np

dtype = np.float32


def numexpr_to_npy(func: str, la: list[np.ndarray], urlpath: str | None) -> np.ndarray:
    """
    Compute `func(a)` using numexpr.

    If `urlpath` is None, compute in-memory and return an ndarray.
    Otherwise, store the result as an on-disk .npy memmap and return it.
    """
    a, b, c = la
    if urlpath is None:
        out = np.empty_like(a)
    else:
        out = np.lib.format.open_memmap(urlpath, mode="w+", dtype=a.dtype, shape=a.shape)
    ne.evaluate(func, out=out, local_dict={"a": a, "b": b, "c": c})
    return out


def compute_example(
    la: list,
    large_la: list,
    intensity: str,
    cparams: blosc2.CParams,
    mem_mode: bool,
) -> dict[str, float]:
    """
    Run a computation for a given intensity on either NumPy/numexpr (ndarray)
    or blosc2 (NDArray), in-memory or on-disk depending on `mem_mode`.
    """
    t0 = time()
    is_numpy = isinstance(large_la[0], np.ndarray)
    np_out_path = None if mem_mode else "result_array.npy"
    res_out_path = None if mem_mode else "result_array.b2nd"

    # --- Elementwise intensities ------------------------------------------------
    if intensity == "very low":
        a, b, c = large_la
        nios = 4
        intensity_val = 2 / nios
        if is_numpy:
            res = numexpr_to_npy("a + b + c", [a, b, c], np_out_path)
        else:
            res = a + b + c

    elif intensity == "low":
        a, b, c = large_la
        nios = 4
        intensity_val = 22 / nios
        if is_numpy:
            res = numexpr_to_npy("sqrt(a + 2 * b + (c / 2)) ** 1.2", [a, b, c], np_out_path)
        else:
            res = np.sqrt(a + 2 * b + (c / 2)) ** 1.2

    elif intensity == "medium":
        a, b, c = large_la
        nios = 4
        intensity_val = 147 / nios
        expr = "exp(sqrt((sin(a) ** 2 + (cos(b) + arctan(c)) ** 3) * (1 + sin(b) ** 2 + cos(c) ** 2)))"
        if is_numpy:
            res = numexpr_to_npy(expr, [a, b, c], np_out_path)
        else:
            res = np.exp(np.sqrt((np.sin(a) ** 2 + (np.cos(b) + np.arctan(c)) ** 3) * (1 + np.sin(b) ** 2 + np.cos(c) ** 2)))

    # --- Matmul intensities -----------------------------------------------------
    elif intensity.startswith("matmul"):
        a, b, c = la
        nios = 3

        # Select submatrix based on intensity level
        scale = {"matmul2": 1, "matmul1": 2, "matmul0": 10}[intensity]
        n = shape[0] // scale

        if is_numpy:
            if scale > 1:
                a = a[n:n + n, n:n + n]
                b = b[n:n + n, n:n + n]
            tmp = np.matmul(a, b)
            if np_out_path is None:
                res = tmp
            else:
                res = np.lib.format.open_memmap(np_out_path, mode="w+", dtype=tmp.dtype, shape=tmp.shape)
                res[...] = tmp
                del tmp
        else:
            if scale > 1:
                a = a.slice((slice(n, n + n), slice(n, n + n)))
                b = b.slice((slice(n, n + n), slice(n, n + n)))
            res = blosc2.matmul(a, b, cparams=cparams, urlpath=res_out_path, mode="w" if not mem_mode else None)

        intensity_val = int((2 * res.shape[0]) / nios)
    else:
        raise ValueError(f"Invalid intensity: {intensity}")

    # --- Final stats ------------------------------------------------------------
    print(f"Intensity = {intensity_val}", end=", ")
    if hasattr(res, "compute"):
        res = res.compute(cparams=cparams, urlpath=res_out_path, mode="w" if not mem_mode else None)

    elapsed = time() - t0
    nelem_compute = res.size
    gflops = intensity_val * nelem_compute / elapsed / 1e9
    bw = nelem_compute * np.dtype(dtype).itemsize * nios / (elapsed * 1e9)
    print(f"Time = {elapsed:.2f}s, GFLOPS = {gflops:.2f}, Mem/disk BW = {bw:.2f} GB/s")

    return {"GFLOPS": gflops, "Intensity": intensity_val, "Time": elapsed}


def create_memmap_linspace(path: str, shape: tuple, dtype) -> np.ndarray:
    """Create a memmap array filled with linspace values chunk-by-chunk."""
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    total_elems = math.prod(shape)
    nelem = math.prod(shape[1:])

    for start in range(0, shape[0]):
        offset = start * nelem
        n_chunk = nelem
        chunk = np.linspace(offset / (total_elems - 1), (offset + n_chunk - 1) / (total_elems - 1), n_chunk, dtype=dtype).reshape((1,) + shape[1:])
        arr[start:start + 1, ...] = chunk

    return arr


def setup_arrays(mem_mode: bool):
    """Setup NumPy and blosc2 arrays for all backends."""
    global shape, large_shape, nelem, large_nelem

    if mem_mode:
        shape = (15_000, 15_000)
        large_shape = (2,) + shape
    else:
        # shape = (30_000, 30_000)
        shape = (15_000, 15_000)
        large_shape = (60,) + shape

    nelem = math.prod(shape)
    large_nelem = math.prod(large_shape)
    print(f"Shape: {shape}, Large shape: {large_shape}")

    # --- NumPy arrays ---
    if mem_mode:
        a_np = np.linspace(0, 1, nelem, dtype=dtype).reshape(shape)
        t0 = time()
        large_a_np = np.linspace(0, 1, large_nelem, dtype=dtype).reshape(large_shape)
        print(f"Large numpy array creation = {time() - t0:.2f} s")
        lops_np = [a_np, a_np.copy(), a_np.copy()]
        large_lops_np = [large_a_np, large_a_np.copy(), large_a_np.copy()]
    else:
        t0 = time()
        a_np = np.lib.format.open_memmap("a_array.npy", mode="w+", dtype=dtype, shape=shape)
        a_np[...] = np.linspace(0, 1, nelem, dtype=dtype).reshape(shape)
        print(f"Numpy memmap creation = {time() - t0:.2f} s")

        t0 = time()
        large_a_np = create_memmap_linspace("large_a_array.npy", large_shape, dtype)
        print(f"Large numpy memmap creation = {time() - t0:.2f} s")

        for src, dst in [("a_array.npy", "b_array.npy"), ("a_array.npy", "c_array.npy"),
                         ("large_a_array.npy", "large_b_array.npy"), ("large_a_array.npy", "large_c_array.npy")]:
            shutil.copy(src, dst)

        lops_np = [a_np, np.lib.format.open_memmap("b_array.npy", mode="r"), np.lib.format.open_memmap("c_array.npy", mode="r")]
        large_lops_np = [large_a_np, np.lib.format.open_memmap("large_b_array.npy", mode="r"),
                         np.lib.format.open_memmap("large_c_array.npy", mode="r")]

    return lops_np, large_lops_np, a_np


def setup_blosc2_backend(a_np, mem_mode: bool, cparams: blosc2.CParams, suffix: str = ""):
    """Setup blosc2 arrays (compressed or non-compressed)."""
    def make_path(name):
        return f"{name}{suffix}.b2nd" if not mem_mode else None

    if mem_mode:
        b2a = blosc2.asarray(a_np, cparams=cparams)
        t0 = time()
        large_b2a = blosc2.linspace(0, 1, large_nelem, dtype=dtype, shape=large_shape, cparams=cparams)
        print(f"Large array creation = {time() - t0:.2f} s")
        lops = [b2a, b2a.copy(cparams=cparams), b2a.copy(cparams=cparams)]
        large_lops = [large_b2a, blosc2.copy(large_b2a, cparams=cparams), blosc2.copy(large_b2a, cparams=cparams)]
    else:
        b2a = blosc2.asarray(a_np, cparams=cparams, urlpath=make_path("a_array"), mode="w")
        t0 = time()
        large_b2a = blosc2.linspace(0, 1, large_nelem, dtype=dtype, shape=large_shape, cparams=cparams,
                                    urlpath=make_path("large_a_array"), mode="w")
        print(f"Large array creation = {time() - t0:.2f} s")

        for src, dst in [(f"a_array{suffix}.b2nd", f"b_array{suffix}.b2nd"), (f"a_array{suffix}.b2nd", f"c_array{suffix}.b2nd"),
                         (f"large_a_array{suffix}.b2nd", f"large_b_array{suffix}.b2nd"),
                         (f"large_a_array{suffix}.b2nd", f"large_c_array{suffix}.b2nd")]:
            shutil.copy(src, dst)

        lops = [b2a, blosc2.open(make_path("b_array"), mode="r"), blosc2.open(make_path("c_array"), mode="r")]
        large_lops = [large_b2a, blosc2.open(make_path("large_b_array"), mode="r"),
                      blosc2.open(make_path("large_c_array"), mode="r")]

    print(f"large_b2a.cratio = {large_b2a.cratio:.2f}, b2a.cratio = {b2a.cratio:.2f}")
    return lops, large_lops


def cleanup_disk_files():
    patterns = ["a_array", "b_array", "c_array", "large_a_array", "large_b_array", "large_c_array", "result_array"]
    for pattern in patterns:
        for ext in [".npy", ".b2nd", "_nc.b2nd"]:
            try:
                os.unlink(pattern + ext)
            except FileNotFoundError:
                pass



def main() -> None:
    mem_mode = not (len(sys.argv) > 1 and sys.argv[1] == "disk")
    print(f"Running in {'in-memory' if mem_mode else 'on-disk'} mode")

    intensities = ["very low", "low", "medium", "matmul0", "matmul1", "matmul2"]
    cparams = blosc2.CParams(codec=blosc2.Codec.LZ4) if mem_mode else blosc2.CParams()

    # Setup arrays
    lops_np, large_lops_np, a_np = setup_arrays(mem_mode)

    # Run benchmarks for each backend
    results = {}
    backends = [
        ("numpy/numexpr", lops_np, large_lops_np, cparams),
        ("blosc2", *setup_blosc2_backend(a_np, mem_mode, cparams), cparams),
        ("blosc2-nocomp", *setup_blosc2_backend(a_np, mem_mode, blosc2.CParams(clevel=0), "_nc"), blosc2.CParams(clevel=0)),
    ]

    for name, lops, large_lops, cp in backends:
        print(f"\n*** {name}")
        results[name] = {}
        for intensity in intensities:
            results[name][intensity] = compute_example(lops, large_lops, intensity, cp, mem_mode)

    pprint.pprint(results)

    if not mem_mode:
        cleanup_disk_files()


if __name__ == "__main__":
    main()
