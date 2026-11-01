#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare miniexpr-disabled/enabled expressions and public Menudet kernels.

All routes materialize NDArrays with the same storage settings. Preparation and
first compute are reported separately; warm timings include allocation and
compression, but not validation or conversion to NumPy. First compute is not a
guaranteed cold compilation (native/disk caches may already be populated).
"""

import argparse
import contextlib
import importlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

import blosc2

lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
where = np.where
sin = np.sin
cos = np.cos
tanh = np.tanh
sqrt = np.sqrt
exp = np.exp
expm1 = np.expm1
log = np.log
log1p = np.log1p
abs = np.abs


@blosc2.menudet
def kernel_loop1(x: np.float32, y: np.float32) -> np.float32:
    acc = 0.0
    for i in range(1):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.menudet
def kernel_loop2(x: np.float32, y: np.float32) -> np.float32:
    acc = 0.0
    for i in range(2):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.menudet
def kernel_loop4(x: np.float32, y: np.float32) -> np.float32:
    acc = 0.0
    for i in range(4):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.menudet
def kernel_loop4_heavy(x: np.float32, y: np.float32) -> np.float32:
    acc = 0.0
    for i in range(4):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1) + (tmp * tmp) * 0.05
    return acc


@blosc2.menudet
def kernel_nested2(x: np.float32, y: np.float32) -> np.float32:
    acc = 0.0
    for i in range(2):
        for j in range(2):
            if (i + j) % 2 == 0:
                tmp = where(x < y, y + i + j, x - i - j)
            else:
                tmp = where(x > y, x + i + j, y - i - j)
            acc = acc + tmp * (i + j + 1)
    return acc


def expr_for_steps(steps: int) -> str:
    terms = []
    for i in range(steps):
        if i % 2 == 0:
            terms.append(f"where(x < y, y + {i}, x - {i}) * {i + 1}")
        else:
            terms.append(f"where(x > y, x + {i}, y - {i}) * {i + 1}")
    return " + ".join(terms)


def expr_for_steps_heavy(steps: int) -> str:
    terms = []
    for i in range(steps):
        if i % 2 == 0:
            term = f"where(x < y, y + {i}, x - {i})"
        else:
            term = f"where(x > y, x + {i}, y - {i})"
        terms.append(f"{term} * {i + 1} + ({term} * {term}) * 0.05")
    return " + ".join(terms)


def expr_nested2() -> str:
    terms = []
    for i in range(2):
        for j in range(2):
            if (i + j) % 2 == 0:
                term = f"where(x < y, y + {i + j}, x - {i + j})"
            else:
                term = f"where(x > y, x + {i + j}, y - {i + j})"
            terms.append(f"{term} * {i + j + 1}")
    return " + ".join(terms)


def expr_transcendentals() -> str:
    return "log(exp(x) + tanh(x) + log1p(abs(x)) + sqrt(abs(x)) + expm1(x))"


def expr_transcend1() -> str:
    return "log(exp(x))"


def expr_transcend2() -> str:
    return "tanh(x)"


def expr_transcend3() -> str:
    return "log1p(abs(x))"


def expr_sincos_identity() -> str:
    return "sin(x) ** 2 + cos(x) ** 2"


@blosc2.menudet
def kernel_transcend1(x: np.float32) -> np.float32:
    return log(exp(x))


@blosc2.menudet
def kernel_transcend2(x: np.float32) -> np.float32:
    return tanh(x)


@blosc2.menudet
def kernel_transcend3(x: np.float32) -> np.float32:
    return log1p(abs(x))


@blosc2.menudet
def kernel_transcend4(x: np.float32) -> np.float32:
    return log(exp(x) + tanh(x) + log1p(abs(x)) + sqrt(abs(x)) + expm1(x))


@blosc2.menudet
def kernel_sincos_identity(x: np.float32) -> np.float32:
    return sin(x) ** 2 + cos(x) ** 2


@contextlib.contextmanager
def miniexpr_enabled(enabled: bool):
    old = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = enabled
    try:
        yield
    finally:
        lazyexpr_mod.try_miniexpr = old


def time_it(fn, niter=3):
    samples = []
    out = None
    for _ in range(niter):
        del out
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        samples.append(dt)
    return min(samples), out, samples


def bench_case(name, kernel, expr, inputs, *, repeat=3):
    a = inputs["x"]
    storage = {"chunks": a.chunks, "blocks": a.blocks, "cparams": a.cparams}
    # NumPy is an independent reference, not one of the timed engines.
    expected = eval(expr, {"__builtins__": {}, **vars(np)}, {k: v[:] for k, v in inputs.items()})
    gb = (sum(v.nbytes for v in inputs.values()) + expected.nbytes) / 1e9
    row = {
        "case": name,
        "preparation_s": {},
        "first_compute_s": {},
        "samples_s": {},
        "execution_reports": {},
    }

    for route, enabled in (("baseline", False), ("lazyexpr", True), ("menudet", True)):
        with miniexpr_enabled(enabled):
            start = time.perf_counter()
            if route == "menudet":
                specialized = kernel.specialize(**inputs)
                lazy = specialized.lazy(inputs)
                row["has_jit"] = specialized.has_jit
            else:
                lazy = blosc2.lazyexpr(expr, inputs)
            row["preparation_s"][route] = time.perf_counter() - start
            start = time.perf_counter()
            result = lazy.compute(**storage)
            row["first_compute_s"][route] = time.perf_counter() - start
            row["execution_reports"][route] = getattr(lazy, "_source_execution_report", None) or getattr(
                lazy, "_native_execution_report", None
            )
            assert isinstance(result, blosc2.NDArray)
            assert result.shape == a.shape
            assert result.dtype == a.dtype, f"{name}/{route}: {result.dtype} != {a.dtype}"
            # Typed native intermediates and NumPy weak scalars round differently;
            # the heavy loop also cancels near zero. Allow 32 float32 eps there.
            np.testing.assert_allclose(
                result[:], expected, rtol=1e-5, atol=32 * np.finfo(a.dtype).eps, err_msg=route
            )
            # No timing is reported for a route until it passes correctness.
            del result
            elapsed, result, samples = time_it(lambda lazy=lazy: lazy.compute(**storage), niter=repeat)
            row["samples_s"][route] = samples
            row[route] = elapsed
            row[f"{route}_gbps"] = gb / elapsed
            del result
    return row


def table_formatter():
    headers = [
        "Case",
        "Base ms",
        "Base GB/s",
        "Expr ms",
        "Expr GB/s",
        "Menudet ms",
        "Menudet GB/s",
        "Expr/Base",
        "Menudet/Base",
    ]
    widths = [
        12,
        len(headers[1]),
        len(headers[2]),
        len(headers[3]),
        len(headers[4]),
        len(headers[5]),
        len(headers[6]),
        len(headers[7]),
        len(headers[8]),
    ]
    align_right = {1, 2, 3, 4, 5, 6, 7, 8}
    fmt_parts = []
    for i, w in enumerate(widths):
        align = ">" if i in align_right else "<"
        fmt_parts.append(f"{{:{align}{w}}}")
    fmt = "|".join(fmt_parts)
    sep = "+".join("-" * w for w in widths)
    return headers, fmt, sep


def format_row(row):
    base = row["baseline"] * 1000
    expr = row["lazyexpr"] * 1000
    menudet = row["menudet"] * 1000
    return [
        row["case"],
        f"{base:.2f}",
        f"{row['baseline_gbps']:.2f}",
        f"{expr:.2f}",
        f"{row['lazyexpr_gbps']:.2f}",
        f"{menudet:.2f}",
        f"{row['menudet_gbps']:.2f}",
        f"{row['baseline'] / row['lazyexpr']:.2f}x",
        f"{row['baseline'] / row['menudet']:.2f}x",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcend", action="store_true", help="Run only the transcendental lazyexpr cases"
    )
    parser.add_argument("--size", type=int, default=1_000, help="Side length of the square inputs")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--compiler", choices=("tcc", "cc"), default="tcc")
    parser.add_argument("--no-jit", action="store_true", help="Use the native interpreter")
    parser.add_argument("--output", type=Path, help="Save metadata and raw per-case results as JSON")
    args = parser.parse_args()
    if min(args.size, args.repeat, args.threads) < 1:
        parser.error("size, repeat and threads must be positive")
    blosc2.set_nthreads(args.threads)

    n = args.size
    dtype = np.float32
    cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

    a = blosc2.linspace(0, 1, n * n, shape=(n, n), dtype=dtype, cparams=cparams)
    b = blosc2.linspace(1, 0, n * n, shape=(n, n), dtype=dtype, cparams=cparams)

    cases = [
        ("loop1", kernel_loop1, expr_for_steps(1)),
        ("loop2", kernel_loop2, expr_for_steps(2)),
        ("loop4", kernel_loop4, expr_for_steps(4)),
        ("loop4_heavy", kernel_loop4_heavy, expr_for_steps_heavy(4)),
        ("nested2", kernel_nested2, expr_nested2()),
    ]

    transcendental_cases = [
        ("transcend1", kernel_transcend1, expr_transcend1()),
        ("transcend2", kernel_transcend2, expr_transcend2()),
        ("transcend3", kernel_transcend3, expr_transcend3()),
        ("transcend4", kernel_transcend4, expr_transcendentals()),
        ("sincos_id", kernel_sincos_identity, expr_sincos_identity()),
    ]

    metadata = {
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "blosc2": blosc2.__version__,
        "numpy": np.__version__,
        "c_blosc2": blosc2.blosclib_version,
        "shape": a.shape,
        "dtype": a.dtype.str,
        "chunks": a.chunks,
        "blocks": a.blocks,
        "cparams": repr(a.cparams),
        "threads": args.threads,
        "compiler_requested": args.compiler,
        "jit_requested": not args.no_jit,
        "repeat": args.repeat,
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ME_", "BLOSC_ME_")) or k in ("CC", "CFLAGS")
        },
        "protocol": "one first compute + best of repeat warm computes; allocation/compression included",
        "input_layout": "C-order linspace x=[0,1], y=[1,0]; no random seed",
    }
    rows = []
    with blosc2.jit_options(jit=not args.no_jit, jit_backend=args.compiler):
        metadata["execution_options"] = {k: str(v) for k, v in blosc2.jit_config.execution_options().items()}
        print("Base: miniexpr disabled; Expr: miniexpr enabled (fallback allowed). Warm best times.")
        for title, group, inputs in (
            ("Loop cases", [] if args.transcend else cases, {"x": a, "y": b}),
            ("Transcendental cases", transcendental_cases, {"x": a}),
        ):
            if not group:
                continue
            print(f"\n{title}", flush=True)
            headers, fmt, sep = table_formatter()
            print(fmt.format(*headers), flush=True)
            print(sep, flush=True)
            for name, kernel, expr in group:
                row = bench_case(name, kernel, expr, inputs, repeat=args.repeat)
                rows.append(row)
                print(fmt.format(*format_row(row)), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
