#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import contextlib
import argparse
import time

import numpy as np

import blosc2
import importlib

lazyexpr_mod = importlib.import_module("blosc2.lazyexpr")
where = np.where


@blosc2.dsl_kernel
def kernel_loop1(x, y):
    acc = 0.0
    for i in range(1):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.dsl_kernel
def kernel_loop2(x, y):
    acc = 0.0
    for i in range(2):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.dsl_kernel
def kernel_loop4(x, y):
    acc = 0.0
    for i in range(4):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1)
    return acc


@blosc2.dsl_kernel
def kernel_loop4_heavy(x, y):
    acc = 0.0
    for i in range(4):
        if i % 2 == 0:
            tmp = where(x < y, y + i, x - i)
        else:
            tmp = where(x > y, x + i, y - i)
        acc = acc + tmp * (i + 1) + (tmp * tmp) * 0.05
    return acc


@blosc2.dsl_kernel
def kernel_nested2(x, y):
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


@contextlib.contextmanager
def miniexpr_enabled(enabled: bool):
    old = lazyexpr_mod.try_miniexpr
    lazyexpr_mod.try_miniexpr = enabled
    try:
        yield
    finally:
        lazyexpr_mod.try_miniexpr = old


def time_it(fn, niter=3):
    best = None
    for _ in range(niter):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return best, out


def bench_case(name, kernel, expr, a, b, dtype, gb):
    if kernel.dsl_source is None:
        raise RuntimeError(f"DSL extraction failed for {name}")

    with miniexpr_enabled(False):
        lazy_expr_base = blosc2.lazyexpr(expr, {"x": a, "y": b})
        res_base = lazy_expr_base.compute()
        base_time, _ = time_it(lambda: lazy_expr_base.compute())

    with miniexpr_enabled(True):
        lazy_expr_fast = blosc2.lazyexpr(expr, {"x": a, "y": b})
        _ = lazy_expr_fast.compute()
        expr_time, _ = time_it(lambda: lazy_expr_fast.compute())

        lazy_dsl = blosc2.lazyudf(kernel, (a, b), dtype=dtype)
        res_dsl = lazy_dsl.compute()
        dsl_time, _ = time_it(lambda: lazy_dsl.compute())

    np.testing.assert_allclose(res_dsl[...], res_base[...], rtol=1e-5, atol=2e-6)

    return {
        "case": name,
        "baseline": base_time,
        "lazyexpr": expr_time,
        "dsl": dsl_time,
        "baseline_gbps": gb / base_time,
        "lazyexpr_gbps": gb / expr_time,
        "dsl_gbps": gb / dsl_time,
    }


def table_formatter():
    headers = [
        "Case",
        "Base ms",
        "Base GB/s",
        "Expr ms",
        "Expr GB/s",
        "DSL ms",
        "DSL GB/s",
        "Expr/Base",
        "DSL/Base",
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
    dsl = row["dsl"] * 1000
    return [
        row["case"],
        f"{base:.2f}",
        f"{row['baseline_gbps']:.2f}",
        f"{expr:.2f}",
        f"{row['lazyexpr_gbps']:.2f}",
        f"{dsl:.2f}",
        f"{row['dsl_gbps']:.2f}",
        f"{row['baseline'] / row['lazyexpr']:.2f}x",
        f"{row['baseline'] / row['dsl']:.2f}x",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transcend", action="store_true", help="Run only the transcendental lazyexpr cases")
    args = parser.parse_args()

    n = 10_000
    dtype = np.float32
    cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)

    a = blosc2.linspace(0, 1, n * n, shape=(n, n), dtype=dtype, cparams=cparams)
    b = blosc2.linspace(1, 0, n * n, shape=(n, n), dtype=dtype, cparams=cparams)
    gb = a.nbytes * 3 / 1e9

    cases = [
        ("loop1", kernel_loop1, expr_for_steps(1)),
        ("loop2", kernel_loop2, expr_for_steps(2)),
        ("loop4", kernel_loop4, expr_for_steps(4)),
        ("loop4_heavy", kernel_loop4_heavy, expr_for_steps_heavy(4)),
        ("nested2", kernel_nested2, expr_nested2()),
    ]

    transcendental_cases = [
        ("transcend1", expr_transcend1()),
        ("transcend2", expr_transcend2()),
        ("transcend3", expr_transcend3()),
        ("transcend4", expr_transcendentals()),
        ("sincos_id", expr_sincos_identity()),
    ]

    if not args.transcend:
        headers, fmt, sep = table_formatter()
        print(fmt.format(*headers), flush=True)
        print(sep, flush=True)
        for name, kernel, expr in cases:
            row = bench_case(name, kernel, expr, a, b, dtype, gb)
            print(fmt.format(*format_row(row)), flush=True)

    if not args.transcend:
        print()
    print("Transcendental lazyexpr cases", flush=True)
    print("Case        |Base ms|Base GB/s|Expr ms|Expr GB/s|Expr/Base", flush=True)
    print("------------+-------+---------+-------+---------+---------", flush=True)
    with miniexpr_enabled(False):
        for name, expr in transcendental_cases:
            lazy_expr_base = blosc2.lazyexpr(expr, {"x": a})
            res_base = lazy_expr_base.compute()
            base_time, _ = time_it(lambda: lazy_expr_base.compute())

            with miniexpr_enabled(True):
                lazy_expr_fast = blosc2.lazyexpr(expr, {"x": a})
                res_fast = lazy_expr_fast.compute()
                expr_time, _ = time_it(lambda: lazy_expr_fast.compute())

            np.testing.assert_allclose(res_fast[...], res_base[...], rtol=1e-5, atol=2e-6)
            print(
                f"{name:<12}|"
                f"{base_time * 1000:>7.2f}|"
                f"{(a.nbytes * 2 / 1e9) / base_time:>9.2f}|"
                f"{expr_time * 1000:>7.2f}|"
                f"{(a.nbytes * 2 / 1e9) / expr_time:>9.2f}|"
                f"{base_time / expr_time:>8.2f}x",
                flush=True,
            )


if __name__ == "__main__":
    main()
