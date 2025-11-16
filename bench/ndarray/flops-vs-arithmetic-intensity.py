#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

# Compute with different arithmetic intensities on NumPy/numexpr and blosc2
# This supports both in-memory and on-disk modes.  On-disk mode is the
# default.  If you want to run in-memory mode, run this script with the
# command line argument "mem".

import math
import os
import pprint
import sys
from time import time

import blosc2
import numexpr as ne
import numpy as np

dtype = np.float32


def numexpr_to_npy(func: str, a: np.ndarray, urlpath: str | None) -> np.ndarray:
    """
    Compute `func(a)` using numexpr.

    If `urlpath` is None, compute in-memory and return an ndarray.
    Otherwise, store the result as an on-disk .npy memmap and return it.
    """
    out: np.ndarray
    if urlpath is None:
        out = np.empty_like(a)
    else:
        out = np.lib.format.open_memmap(urlpath, mode="w+", dtype=a.dtype, shape=a.shape)
    ne.evaluate(func, out=out, local_dict={"a": a})
    return out


def compute_example(
    a,
    large_a,
    intensity: str,
    cparams: blosc2.CParams,
    mem_mode: bool,
) -> dict[str, float]:
    """
    Run a computation for a given intensity on either NumPy/numexpr (ndarray)
    or blosc2 (NDArray), in-memory or on-disk depending on `mem_mode`.
    """
    t0 = time()

    np_out_path = None if mem_mode else "result_array.npy"
    b2_out_path = None if mem_mode else "result_array.b2nd"

    # --- Elementwise intensities ------------------------------------------------
    if intensity == "very low":
        a = large_a
        intensity_val = 1 / 2  # 1 FLOP / (2 elems)
        if isinstance(a, np.ndarray):
            b = numexpr_to_npy("a + 1", a, np_out_path)
        else:
            b = a + 1

    elif intensity == "low":
        a = large_a
        intensity_val = 22 / 2  # 22 FLOPs / (2 elems)
        if isinstance(a, np.ndarray):
            b = numexpr_to_npy(
                "sqrt(a + 2 * a + (a / 2)) ** 1.2",
                a,
                np_out_path,
            )
        else:
            b = np.sqrt(a + 2 * a + (a / 2)) ** 1.2

    elif intensity == "medium":
        a = large_a
        intensity_val = 147 / 2
        expr = (
            "exp(sqrt((sin(a) ** 2 + (cos(a) + arctan(a)) ** 3) * "
            "(1 + sin(a) ** 2 + cos(a) ** 2)))"
        )
        if isinstance(a, np.ndarray):
            b = numexpr_to_npy(expr, a, np_out_path)
        else:
            b = np.exp(
                np.sqrt(
                    np.sin(a) ** 2
                    + (np.cos(a) + np.arctan(a)) ** 3
                )
                * (1 + np.sin(a) ** 2 + np.cos(a) ** 2)
            )

    # --- Matmul intensities -----------------------------------------------------
    elif intensity.startswith("matmul"):
        # Choose submatrix `c`
        if isinstance(a, np.ndarray):
            if intensity == "matmul2":
                c = a
            elif intensity == "matmul1":
                n = shape[0] // 2
                c = a[n:n + n, n:n + n]
            elif intensity == "matmul0":
                n = shape[0] // 10
                c = a[n:n + n, n:n + n]
            else:
                raise ValueError(f"Invalid intensity: {intensity}")

            tmp = np.matmul(c, c)
            if np_out_path is None:
                b = tmp
            else:
                b = np.lib.format.open_memmap(
                    np_out_path, mode="w+", dtype=tmp.dtype, shape=tmp.shape
                )
                b[...] = tmp
                del tmp
        else:
            if intensity == "matmul2":
                c = a
            elif intensity == "matmul1":
                n = shape[0] // 2
                # c = blosc2.asarray(a[n:n + n, n:n + n])
                c = a.slice((slice(n, n + n), slice(n, n + n)))
            elif intensity == "matmul0":
                n = shape[0] // 10
                # c = blosc2.asarray(a[n:n + n, n:n + n])
                c = a.slice((slice(n, n + n), slice(n, n + n)))
            else:
                raise ValueError(f"Invalid intensity: {intensity}")

            b = blosc2.matmul(
                c,
                c,
                cparams=cparams,
                urlpath=b2_out_path,
                mode="w" if not mem_mode else None,
            )

        intensity_val = int((2 * c.shape[0]) / 3)

    else:
        raise ValueError(f"Invalid intensity: {intensity}")

    # --- Final stats ------------------------------------------------------------
    print(f"Intensity = {intensity_val}", end=", ")
    if hasattr(b, "compute"):
        b = b.compute(
            cparams=cparams,
            urlpath=b2_out_path,
            mode="w" if not mem_mode else None,
        )

    elapsed = time() - t0
    print(f"Time = {elapsed:.2f}s", end=", ")
    nelem_compute = b.size
    gflops = intensity_val * nelem_compute / elapsed / 1e9
    print(f"GFLOPS = {gflops:.2f}", end=", ")
    bw = nelem_compute * b.dtype.itemsize * 2 / (elapsed * 1e9)
    print(f"Mem/disk BW = {bw:.2f} GB/s")

    return {"GFLOPS": gflops, "Intensity": intensity_val, "Time": elapsed}


def setup_numpy_arrays(mem_mode: bool):
    """
    Return (a, large_a) for NumPy/numexpr backend.

    If mem_mode is True, use in-memory ndarrays.
    Otherwise, use on-disk memmaps.
    """
    global shape, large_shape, nelem, large_nelem
    if mem_mode:
        shape = (15_000, 15_000)
        large_shape = (5,) + shape
    else:
        shape = (30_000, 30_000)
        large_shape = (50,) + shape
    nelem = math.prod(shape)
    large_nelem = math.prod(large_shape)

    print("Shape:", shape)
    print("Large shape:", large_shape)

    if mem_mode:
        a_np = np.linspace(0, 1, nelem, dtype=dtype).reshape(shape)
        t0 = time()
        large_a_np = np.linspace(0, 1, large_nelem, dtype=dtype).reshape(large_shape)
        print(f"Large numpy array creation = {time() - t0:.2f} s")
        return a_np, large_a_np

    t0 = time()
    np_a_path = "a_array.npy"
    a_np = np.lib.format.open_memmap(np_a_path, mode="w+", dtype=dtype, shape=shape)
    a_np[...] = np.linspace(0, 1, nelem, dtype=dtype).reshape(shape)
    print(f"Numpy memmap creation = {time() - t0:.2f} s")

    t0 = time()
    large_np_a_path = "large_a_array.npy"
    large_a_np = np.lib.format.open_memmap(
        large_np_a_path, mode="w+", dtype=dtype, shape=large_shape
    )

    chunk_size = 1
    total_elems = large_nelem
    for start in range(0, large_shape[0], chunk_size):
        stop = min(start + chunk_size, large_shape[0])
        n_chunk = (stop - start) * nelem
        offset = start * nelem
        chunk = np.linspace(
            offset / (total_elems - 1),
            (offset + n_chunk - 1) / (total_elems - 1),
            n_chunk,
            dtype=dtype,
        ).reshape((stop - start,) + shape)
        large_a_np[start:stop, ...] = chunk
    print(f"Large numpy memmap creation = {time() - t0:.2f} s")

    return a_np, large_a_np


def setup_blosc2_arrays(a_np, mem_mode: bool, cparams: blosc2.CParams):
    """
    Return (b2a, large_b2a) for a compressed blosc2 backend.

    If mem_mode is True, arrays are in-memory. Otherwise they are on-disk.
    """
    if mem_mode:
        b2a = blosc2.asarray(a_np, cparams=cparams)
        t0 = time()
        large_b2a = blosc2.linspace(
            0, 1, large_nelem, dtype=dtype, shape=large_shape, cparams=cparams
        )
        print(f"Large array creation = {time() - t0:.2f} s")
    else:
        b2a = blosc2.asarray(a_np, cparams=cparams, urlpath="a_array.b2nd", mode="w")
        t0 = time()
        large_b2a = blosc2.linspace(
            0,
            1,
            large_nelem,
            dtype=dtype,
            shape=large_shape,
            cparams=cparams,
            urlpath="large_array.b2nd",
            mode="w",
        )
        print(f"Large array creation = {time() - t0:.2f} s")

    print(f"large_b2a.cratio = {large_b2a.cratio:.2f}, b2a.cratio = {b2a.cratio:.2f}")
    return b2a, large_b2a


def setup_blosc2_nocomp_arrays(large_b2a, a_np, mem_mode: bool):
    """
    Return (b2a_nc, large_b2a_nc) for a non-compressed blosc2 backend.
    """
    no_compr_cparams = blosc2.CParams(clevel=0)
    if mem_mode:
        b2a_nc = blosc2.asarray(a_np, cparams=no_compr_cparams)
        t0 = time()
        large_b2a_nc = blosc2.copy(large_b2a, cparams=no_compr_cparams)
        print(f"Large array creation = {time() - t0:.2f} s")
    else:
        b2a_nc = blosc2.asarray(
            a_np, cparams=no_compr_cparams, urlpath="a_array.b2nd", mode="w"
        )
        t0 = time()
        large_b2a_nc = blosc2.copy(
            large_b2a,
            cparams=no_compr_cparams,
            urlpath="large_array2.b2nd",
            mode="w",
        )
        print(f"Large array creation = {time() - t0:.2f} s")

    print(
        f"large_b2a_nc.cratio = {large_b2a_nc.cratio:.2f}, "
        f"b2a_nc.cratio = {b2a_nc.cratio:.2f}"
    )
    return b2a_nc, large_b2a_nc, no_compr_cparams


def cleanup_disk_files():
    for fname in (
        "a_array.npy",
        "large_a_array.npy",
        "a_array.b2nd",
        "large_array.b2nd",
        "large_array2.b2nd",
        "result_array.b2nd",
        "result_array.npy",
    ):
        try:
            os.unlink(fname)
        except FileNotFoundError:
            pass


def main() -> None:
    mem_mode = len(sys.argv) > 1 and sys.argv[1] == "mem"

    results = {"numpy/numexpr": {}, "blosc2": {}, "blosc2-nocomp": {}}
    intensities = ["very low", "low", "medium", "matmul0", "matmul1", "matmul2"]

    cparams = blosc2.CParams()
    if mem_mode:
        print("Running in-memory mode (mem)")
        cparams = blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=1)

    # --- NumPy/numexpr backend --------------------------------------------------
    a_np, large_a_np = setup_numpy_arrays(mem_mode)
    print(f"*** Numpy/numexpr ({'in-memory' if mem_mode else 'memmap on-disk'})")
    for intensity in intensities:
        results["numpy/numexpr"][intensity] = compute_example(
            a_np, large_a_np, intensity, cparams, mem_mode
        )

    # --- blosc2 compressed backend ---------------------------------------------
    print(f"*** Blosc2 array (compressed, {'in-memory' if mem_mode else 'on-disk'})")
    b2a, large_b2a = setup_blosc2_arrays(a_np, mem_mode, cparams)
    for intensity in intensities:
        results["blosc2"][intensity] = compute_example(
            b2a, large_b2a, intensity, cparams, mem_mode
        )

    # --- blosc2 non-compressed backend -----------------------------------------
    print(f"*** Blosc2 array (no compressed, {'in-memory' if mem_mode else 'on-disk'})")
    b2a_nc, large_b2a_nc, no_compr_cparams = setup_blosc2_nocomp_arrays(
        large_b2a, a_np, mem_mode
    )
    for intensity in intensities:
        results["blosc2-nocomp"][intensity] = compute_example(
            b2a_nc, large_b2a_nc, intensity, no_compr_cparams, mem_mode
        )

    pprint.pprint(results)

    if not mem_mode:
        cleanup_disk_files()


if __name__ == "__main__":
    main()
