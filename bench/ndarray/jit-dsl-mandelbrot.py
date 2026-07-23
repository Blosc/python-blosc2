#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Compares a NumPy-vectorized Mandelbrot escape-time kernel against the same
# kernel run through @blosc2.jit's DSL (control-flow) dispatch route, directly
# on NumPy operands.  Tracing would silently drop the per-pixel loop/break, so
# jit compiles the whole function with miniexpr instead; see
# doc/guides/optimization_tips.md ("Let @blosc2.jit compile control flow
# instead of tracing it").
#
# Return paths are equalized (both calls end in a plain NumPy array): any
# non-None jit() kwarg flips the return from `retval[()]` to `.compute()`,
# which would otherwise skew the comparison.

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np

import blosc2


@blosc2.jit
def mandelbrot_jit(cr, ci, max_iter):
    zr = 0.0
    zi = 0.0
    n = 0
    for _i in range(max_iter):
        if zr * zr + zi * zi > 4.0:
            break
        new_zr = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = new_zr
        n = n + 1
    return n


def mandelbrot_numpy(cr, ci, max_iter):
    zr = np.zeros_like(cr)
    zi = np.zeros_like(ci)
    n = np.zeros(cr.shape, dtype=np.int64)
    active = np.ones(cr.shape, dtype=bool)
    for _i in range(max_iter):
        mag = zr * zr + zi * zi
        active &= mag <= 4.0
        new_zr = zr * zr - zi * zi + cr
        new_zi = 2 * zr * zi + ci
        zr = np.where(active, new_zr, zr)
        zi = np.where(active, new_zi, zi)
        n = np.where(active, n + 1, n)
    return n


def _grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    cr = np.linspace(-2.0, 1.0, width, dtype=np.float64)[None, :] * np.ones((height, 1))
    ci = np.linspace(-1.5, 1.5, height, dtype=np.float64)[:, None] * np.ones((1, width))
    return cr, ci


def _bench(fn, reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(description="NumPy vs @blosc2.jit (DSL route) Mandelbrot benchmark.")
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()

    cr, ci = _grid(args.height, args.width)
    print(f"grid: {args.height}x{args.width}, max_iter={args.max_iter}")

    ref = mandelbrot_numpy(cr, ci, args.max_iter)
    got = mandelbrot_jit(cr, ci, args.max_iter)
    assert np.array_equal(ref, got), "jit DSL result does not match the NumPy reference"

    numpy_med = _bench(lambda: mandelbrot_numpy(cr, ci, args.max_iter), args.reps, args.warmup)
    jit_med = _bench(lambda: mandelbrot_jit(cr, ci, args.max_iter), args.reps, args.warmup)

    print(f"numpy vectorized:      {numpy_med:.6f} s")
    print(f"blosc2.jit (DSL route): {jit_med:.6f} s")
    print(f"speedup: {numpy_med / jit_med:.2f}x")


if __name__ == "__main__":
    main()
