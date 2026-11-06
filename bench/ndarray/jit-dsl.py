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

import numpy as np

import blosc2

where = np.where
sin = np.sin
cos = np.cos
exp = np.exp
log = np.log


@blosc2.menudet
def k_menudet(x, y):
    acc = x
    i = 0
    while i < 2:
        if i == 0:
            acc = acc + y
        else:
            acc = where(acc < y, acc + i, acc - i)
        i = i + 1
    return acc


@blosc2.menudet
def k_heavy_menudet(x, y, niter: np.int64):
    acc = x
    i = 0
    while i < niter:
        t = sin(acc * 1.001 + y * 0.123)
        u = cos(acc * 0.777 - y * 0.211)
        v = exp(t * 0.25) - log(abs(u) + 1.0)
        p = sin(v * 0.731 + acc * 0.071)
        q = cos(v * 0.379 - y * 0.053)
        r = exp((p - q) * 0.17) - log(abs(p + q) + 1.0)
        w = sin((r + v) * 0.11) + cos((r - v) * 0.07)
        delta = v + r + w
        acc = where((acc < y), (acc + delta), (acc - delta))
        i = i + 1
    return acc


@blosc2.menudet
def k_arith_loop_menudet(x, y, niter: np.int64):
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


def make_mandelbrot(max_iter):
    # A typed captured bound lets the native proof admit block-parallel execution.
    # This benchmarks bounded for loops, not runtime while-loop/cap semantics.
    iterations = np.int64(max_iter)

    @blosc2.menudet
    def mandelbrot_menudet(cr, ci):
        zr = cr * 0.0
        zi = ci * 0.0
        for _i in range(iterations):
            zr2 = ((zr * zr) - (zi * zi)) + cr
            zi2 = ((zr * zi) * 2.0) + ci
            zr = zr2
            zi = zi2
        # Mandelbrot-like iterate z <- z^2 + c (returns final magnitude proxy).
        return (zr * zr) + (zi * zi)

    return mandelbrot_menudet


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
def _fresh_tmpdir(enabled: bool, mandelbrot):
    if not enabled:
        yield
        return
    old_tmpdir = os.environ.get("TMPDIR")
    tmpdir = tempfile.mkdtemp(prefix="me-jit-bench-")
    os.environ["TMPDIR"] = tmpdir
    try:
        # Menudet keeps specialized kernels in memory as well as compiler files.
        for author in (k_menudet, k_heavy_menudet, k_arith_loop_menudet, mandelbrot):
            author._kernels.clear()
        with blosc2.jit_options(cache_dir=tmpdir):
            yield
    finally:
        if old_tmpdir is None:
            os.environ.pop("TMPDIR", None)
        else:
            os.environ["TMPDIR"] = old_tmpdir
        shutil.rmtree(tmpdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JIT modes for expressions, reductions and Menudet kernels."
    )
    parser.add_argument("--n", type=int, default=10_000, help="Array length.")
    parser.add_argument("--reps", type=int, default=2, help="Measured repetitions per workload/mode.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per workload/mode.")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"), help="Input dtype.")
    parser.add_argument("--clevel", type=int, default=1, help="Compression level for input arrays.")
    parser.add_argument(
        "--heavy-iters", type=int, default=16, help="Iterations for the heavy Menudet kernel."
    )
    parser.add_argument(
        "--arith-iters", type=int, default=512, help="Iterations for the arithmetic loop Menudet kernel."
    )
    parser.add_argument(
        "--mandelbrot-iters",
        type=int,
        default=50,
        help="Captured iteration bound for Mandelbrot Menudet kernel.",
    )
    parser.add_argument(
        "--compiler",
        default="auto",
        choices=("auto", "tcc", "cc"),
        help="JIT backend override: auto (default), tcc, or cc.",
    )
    parser.add_argument(
        "--mode",
        default="all",
        choices=("off", "on", "auto", "all"),
        help="JIT mode rows to benchmark: off, on, auto, or all (default).",
    )
    parser.add_argument(
        "--fresh-cache",
        action="store_true",
        help="Use fresh compiler/specialization caches per workload/mode row, including JIT build cost.",
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
    mandelbrot = make_mandelbrot(args.mandelbrot_iters)

    mode_map = {
        "auto": ("auto", None),
        "on": ("on", True),
        "off": ("off", False),
    }
    modes = (
        [mode_map["auto"], mode_map["on"], mode_map["off"]] if args.mode == "all" else [mode_map[args.mode]]
    )
    rows = []

    for mode_name, jit in modes:
        with blosc2.jit_options(jit=jit, jit_backend=jit_backend):
            rows.extend(bench_mode(a, b, cr, ci, mandelbrot, args, mode_name))

    warm_baseline = {}
    cold_baseline = {}
    for workload, mode_name, cold, med, _best in rows:
        if mode_name == "off":
            warm_baseline[workload] = med
            cold_baseline[workload] = cold

    print(f"\nbackend: {args.compiler}")
    print("workload            mode    cold_s  warm_med_s   best_s   warm_speedup   cold_speedup")
    print("-----------------------------------------------------------------------------------")
    for workload, mode_name, cold, med, best in rows:
        warm_base = warm_baseline.get(workload)
        cold_base = cold_baseline.get(workload)
        warm_speedup = f"{(warm_base / med):>8.3f}x" if warm_base else f"{'n/a':>8}"
        cold_speedup = f"{(cold_base / cold):>8.3f}x" if cold_base else f"{'n/a':>8}"
        print(
            f"{workload:<18} {mode_name:<5} {_fmt(cold):>8}   {_fmt(med):>8}   {_fmt(best):>8}   "
            f"{warm_speedup}   {cold_speedup}"
        )


def bench_mode(a, b, cr, ci, mandelbrot, args, mode_name):
    rows = []
    workloads = (
        ("compute_expr", lambda: blosc2.sin(a + 0.5).compute()),
        ("reduce_sum", lambda: blosc2.sin(a + 0.5).sum()),
        ("menudet", lambda: k_menudet(a, b).compute()),
        ("menudet_heavy", lambda: k_heavy_menudet(a, b, np.int64(args.heavy_iters)).compute()),
        ("menudet_arith", lambda: k_arith_loop_menudet(a, b, np.int64(args.arith_iters)).compute()),
        ("mandelbrot_menudet", lambda: mandelbrot(cr, ci).compute()),
    )
    for name, compute in workloads:
        with _fresh_tmpdir(args.fresh_cache, mandelbrot):
            cold, med, best = _bench_cold_warm(compute, args.reps, args.warmup)
        rows.append((name, mode_name, cold, med, best))
    return rows


if __name__ == "__main__":
    main()
