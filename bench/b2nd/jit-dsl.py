#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import statistics
import tempfile
import time

import blosc2
import numpy as np


@blosc2.dsl_kernel
def k_dsl(x, y):
    acc = x
    i = 0
    while i < 2:
        if i == 0:
            acc = acc + y
        else:
            acc = np.where(acc < y, acc + i, acc - i)
        i = i + 1
    return acc


@blosc2.dsl_kernel
def k_heavy_dsl(x, y, niter):
    acc = x
    i = 0
    while i < niter:
        t = np.sin(acc * 1.001 + y * 0.123)
        u = np.cos(acc * 0.777 - y * 0.211)
        v = np.exp(t * 0.25) - np.log(np.abs(u) + 1.0)
        p = np.sin(v * 0.731 + acc * 0.071)
        q = np.cos(v * 0.379 - y * 0.053)
        r = np.exp((p - q) * 0.17) - np.log(np.abs(p + q) + 1.0)
        w = np.sin((r + v) * 0.11) + np.cos((r - v) * 0.07)
        delta = v + r + w
        acc = np.where((acc < y), (acc + delta), (acc - delta))
        i = i + 1
    return acc


@blosc2.dsl_kernel
def k_arith_loop_dsl(x, y, niter):
    acc = x
    i = 0
    while i < niter:
        # Arithmetic-only recurrence intended to stress loop codegen.
        a1 = acc * 0.913 + y * 0.087
        a2 = a1 * 0.731 + acc * 0.269
        a3 = a2 * 0.619 + a1 * 0.381
        a4 = a3 * 0.541 + a2 * 0.459
        a5 = a4 * 0.503 + a3 * 0.497
        acc = (acc * 0.97) + (a5 * 0.03) + (i * 0.0000001)
        i = i + 1
    return acc


@blosc2.dsl_kernel
def mandelbrot_dsl(cr, ci, max_iter):
    zr = cr * 0.0
    zi = ci * 0.0
    i = 0
    while i < max_iter:
        zr2 = ((zr * zr) - (zi * zi)) + cr
        zi2 = (((zr * zi) * 2.0) + ci)
        zr = zr2
        zi = zi2
        i = i + 1
    # Mandelbrot-like iterate z <- z^2 + c (returns final magnitude proxy).
    return ((zr * zr) + (zi * zi))


def _bench_cold_warm(fn, reps: int, warmup: int) -> tuple[float, float, float]:
    # First invocation: captures JIT compile/runtime setup cost when present.
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0

    # Optional warmup happens after first call, so "cold" remains representative.
    for _ in range(warmup):
        fn()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return cold, statistics.median(times), min(times)


def _fmt(v: float) -> str:
    return f"{v:.6f}"


@contextlib.contextmanager
def _fresh_tmpdir(enabled: bool):
    if not enabled:
        yield
        return
    old_tmpdir = os.environ.get("TMPDIR")
    tmpdir = tempfile.mkdtemp(prefix="me-jit-bench-")
    os.environ["TMPDIR"] = tmpdir
    try:
        yield
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark JIT modes for expressions, reductions and DSL kernels.")
    parser.add_argument("--n", type=int, default=100_000, help="Array length.")
    parser.add_argument("--reps", type=int, default=2, help="Measured repetitions per workload/mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per workload/mode.")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"), help="Input dtype.")
    parser.add_argument("--clevel", type=int, default=1, help="Compression level for input arrays.")
    parser.add_argument("--heavy-iters", type=int, default=16, help="Iterations for the heavy DSL kernel.")
    parser.add_argument("--arith-iters", type=int, default=512, help="Iterations for the arithmetic loop DSL kernel.")
    parser.add_argument("--mandelbrot-iters", type=int, default=50, help="Iterations for Mandelbrot DSL kernel.")
    parser.add_argument(
        "--compiler",
        default="auto",
        choices=("auto", "tcc", "cc"),
        help="JIT backend override: auto (default), tcc, or cc.",
    )
    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="Use a fresh TMPDIR per workload/mode row so cold_s includes actual JIT build cost.",
    )
    parser.add_argument("--trace", action="store_true", help="Print reminder for ME_DSL_TRACE usage.")
    args = parser.parse_args()

    if args.trace:
        print("Tip: run with ME_DSL_TRACE=1 for backend/JIT diagnostics.")

    dtype = np.dtype(args.dtype)
    jit_backend = None if args.compiler == "auto" else args.compiler
    cparams = blosc2.CParams(clevel=args.clevel, codec=blosc2.Codec.LZ4)

    print(f"Building inputs: n={args.n:,}, dtype={dtype}, clevel={args.clevel}")
    a = blosc2.linspace(0.0, 1.0, args.n, dtype=dtype)
    b = blosc2.linspace(1.0, 2.0, args.n, dtype=dtype, cparams=cparams)
    cr = blosc2.linspace(-2.0, 1.0, args.n, dtype=dtype, cparams=cparams)
    ci = blosc2.linspace(-1.5, 1.5, args.n, dtype=dtype, cparams=cparams)

    modes = [("auto", None), ("on", True), ("off", False)]
    rows = []

    for mode_name, jit in modes:
        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.sin(a + 0.5).compute(jit=jit, jit_backend=jit_backend), args.reps, args.warmup
            )
        rows.append(("compute_expr", mode_name, cold, med, best))

        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.sin(a + 0.5).sum(jit=jit, jit_backend=jit_backend), args.reps, args.warmup
            )
        rows.append(("reduce_sum", mode_name, cold, med, best))

        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.lazyudf(k_dsl, (a, b), dtype=dtype, jit=jit, jit_backend=jit_backend).compute(),
                args.reps,
                args.warmup,
            )
        rows.append(("lazyudf_dsl", mode_name, cold, med, best))

        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.lazyudf(
                    k_heavy_dsl,
                    (a, b, args.heavy_iters),
                    dtype=dtype,
                    jit=jit,
                    jit_backend=jit_backend,
                ).compute(),
                args.reps,
                args.warmup,
            )
        rows.append(("lazyudf_heavy", mode_name, cold, med, best))

        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.lazyudf(
                    k_arith_loop_dsl,
                    (a, b, args.arith_iters),
                    dtype=dtype,
                    jit=jit,
                    jit_backend=jit_backend,
                ).compute(),
                args.reps,
                args.warmup,
            )
        rows.append(("udf_arith", mode_name, cold, med, best))

        with _fresh_tmpdir(args.fresh_cache):
            cold, med, best = _bench_cold_warm(
                lambda: blosc2.lazyudf(
                    mandelbrot_dsl,
                    (cr, ci, args.mandelbrot_iters),
                    dtype=dtype,
                    jit=jit,
                    jit_backend=jit_backend,
                ).compute(),
                args.reps,
                args.warmup,
            )
        rows.append(("mandelbrot_dsl", mode_name, cold, med, best))

    warm_baseline = {}
    cold_baseline = {}
    for workload, mode_name, cold, med, _best in rows:
        if mode_name == "off":
            warm_baseline[workload] = med
            cold_baseline[workload] = cold

    print(f"\nbackend: {args.compiler}")
    print("workload        mode    cold_s  warm_med_s   best_s   warm_speedup   cold_speedup")
    print("-----------------------------------------------------------------------------------")
    for workload, mode_name, cold, med, best in rows:
        warm_base = warm_baseline.get(workload)
        cold_base = cold_baseline.get(workload)
        warm_speedup = (warm_base / med) if warm_base else 1.0
        cold_speedup = (cold_base / cold) if cold_base else 1.0
        print(
            f"{workload:<14} {mode_name:<5} {_fmt(cold):>8}   {_fmt(med):>8}   {_fmt(best):>8}   "
            f"{warm_speedup:>8.3f}x   {cold_speedup:>8.3f}x"
        )


if __name__ == "__main__":
    main()
