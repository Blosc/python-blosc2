#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare normal LazyExpr execution and public Menudet kernels.

All routes materialize NDArrays with the same storage settings. Positive
--warmups counts select prepared-engine warm medians over --samples runs.
--warmups 0 isolates expression/compiler caches before every sample and includes
engine construction/compilation. Inputs, imports, cache isolation, and validation
are not timed; CPU/OS caches are not flushed. Allocation/compression are timed.
--jit tcc|cc requests a backend; omission inherits environment/default policy.
--math-backend system|sleef|accelerate selects the math library for both routes;
omission inherits the current Python/native policy. A requested backend that the
build lacks fails loudly rather than silently using another library.

Examples::

    python bench/ndarray/dsl-kernel-bench.py --jit cc --warmups 3 --samples 3
    python bench/ndarray/dsl-kernel-bench.py --jit cc --warmups 0 --samples 3
    python bench/ndarray/dsl-kernel-bench.py --math-backend sleef
"""

import argparse
import importlib
import json
import os
import platform
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from statistics import median

import numpy as np

import blosc2

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


@blosc2.menudet(jit=False)
def kernel_math_probe(x: np.float64) -> np.float64:
    """float64 so the probe never shares a signature with the measured kernels."""
    return sin(x) + exp(x)


def require_math_backend(parser, backend):
    """Fail before printing any table when the build lacks the requested library."""
    if backend is None:
        return
    try:
        kernel_math_probe.specialize(np.zeros(2, dtype=np.float64))
    except blosc2.MenudetError as error:
        parser.error(f"--math-backend {backend}: {error}")


@contextmanager
def cold_preparation(author):
    """Isolate preparation/compiler caches as in bench/menudet-engines.py."""
    native = importlib.import_module("blosc2.native_graph")
    lazy = importlib.import_module("blosc2.lazyexpr")
    graph = importlib.import_module("blosc2.expression_graph")
    native._native_plan_cache.clear()
    for function in (
        native._native_first_syntax,
        native._native_text_expression,
        native.compile_plan,
        native.resident_map_kernel,
        native.configured_map_kernel,
        lazy._evaluate_recipe_names,
        graph.parse_expression,
        graph.expression_syntax,
    ):
        function.cache_clear()
    lazy._evaluate_recipe_cache.clear()
    author._kernels.clear()
    keys = ("ME_DSL_JIT_CACHE_DIR", "ME_DSL_JIT_POS_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    with tempfile.TemporaryDirectory(prefix="menudet-cold-") as directory:
        os.environ.update(ME_DSL_JIT_CACHE_DIR=directory, ME_DSL_JIT_POS_CACHE="0")
        try:
            with blosc2.jit_options(cache_dir=directory):
                yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def time_it(fn, *, samples=3, warmups=3, cold_factory=None, author=None):
    for _ in range(warmups):
        out = fn()
        del out
    timings = []
    out = None
    for _ in range(samples):
        del out
        with cold_preparation(author) if warmups == 0 else nullcontext():
            t0 = time.perf_counter()
            out = cold_factory() if warmups == 0 else fn()
            dt = time.perf_counter() - t0
        timings.append(dt)
    return median(timings), out, timings


def execution_mode(report, has_jit=None):
    """Return the engine a route actually ran, never just the requested policy."""
    backend = (report or {}).get("backend")
    if backend in ("menudet-jit", "menudet-interpreter"):
        return "jit" if backend == "menudet-jit" else "interp"
    if has_jit is not None:
        return "jit" if has_jit else "interp"
    return backend or "other"


def bench_case(name, kernel, expr, inputs, *, samples=3, warmups=3):
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
        "execution_modes": {},
        "mode": "preparation-cold" if warmups == 0 else "warm",
    }

    for route in ("lazyexpr", "menudet"):

        def prepare(route=route):
            if route == "menudet":
                specialized = kernel.specialize(**inputs)
                row["has_jit"] = specialized.has_jit
                return specialized.lazy(inputs)
            return blosc2.lazyexpr(expr, inputs)

        start = time.perf_counter()
        lazy = prepare()
        row["preparation_s"][route] = time.perf_counter() - start
        start = time.perf_counter()
        result = lazy.compute(**storage)
        row["first_compute_s"][route] = time.perf_counter() - start
        row["execution_reports"][route] = getattr(lazy, "_source_execution_report", None) or getattr(
            lazy, "_native_execution_report", None
        )

        def record_mode(route=route):
            report = row["execution_reports"].get(route)
            # Menudet reports JIT support through its specialized kernel, not
            # through the storage execution report.
            row["execution_modes"][route] = execution_mode(
                report, row.get("has_jit") if route == "menudet" else None
            )

        record_mode()
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

        def cold_run(prepare=prepare, route=route):
            cold_lazy = prepare()
            result = cold_lazy.compute(**storage)
            row["execution_reports"][route] = getattr(
                cold_lazy, "_source_execution_report", None
            ) or getattr(cold_lazy, "_native_execution_report", None)
            record_mode()
            return result

        elapsed, result, timings = time_it(
            lambda lazy=lazy: lazy.compute(**storage),
            samples=samples,
            warmups=warmups,
            cold_factory=cold_run,
            author=kernel,
        )
        row["samples_s"][route] = timings
        row[route] = elapsed
        row[f"{route}_gbps"] = gb / elapsed
        del result
    return row


def table_formatter():
    headers = [
        "Case",
        "LazyExpr ms",
        "LazyExpr GB/s",
        "LazyExpr mode",
        "Menudet ms",
        "Menudet GB/s",
        "Menudet mode",
        "Menudet speedup",
    ]
    widths = [12, *(len(header) for header in headers[1:])]
    align_right = set(range(1, len(headers)))
    fmt_parts = []
    for i, w in enumerate(widths):
        align = ">" if i in align_right else "<"
        fmt_parts.append(f"{{:{align}{w}}}")
    fmt = "|".join(fmt_parts)
    sep = "+".join("-" * w for w in widths)
    return headers, fmt, sep


def format_row(row):
    expr = row["lazyexpr"] * 1000
    menudet = row["menudet"] * 1000
    modes = row.get("execution_modes", {})
    return [
        row["case"],
        f"{expr:.2f}",
        f"{row['lazyexpr_gbps']:.2f}",
        modes.get("lazyexpr", "?"),
        f"{menudet:.2f}",
        f"{row['menudet_gbps']:.2f}",
        modes.get("menudet", "?"),
        f"{row['lazyexpr'] / row['menudet']:.2f}x",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--transcend", action="store_true", help="Run only the transcendental lazyexpr cases"
    )
    parser.add_argument("--size", type=int, default=4_000, help="Side length of the square inputs")
    parser.add_argument("--samples", type=int, default=3, help="timed repetitions; report the median")
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="untimed repetitions; 0 selects preparation-cold samples including compilation",
    )
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    jit_group = parser.add_mutually_exclusive_group()
    jit_group.add_argument("--jit", choices=("tcc", "cc"), help="JIT backend (default: inherit policy)")
    jit_group.add_argument("--no-jit", action="store_true", help="Use the native interpreter")
    parser.add_argument(
        "--math-backend",
        choices=("system", "sleef", "accelerate"),
        help="math library for both routes (default: inherit Python/native policy)",
    )
    parser.add_argument("--output", type=Path, help="Save metadata and raw per-case results as JSON")
    args = parser.parse_args()
    if min(args.size, args.samples, args.threads) < 1 or args.warmups < 0:
        parser.error("size, samples and threads must be positive; warmups must be non-negative")
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
        "routes": ["lazyexpr", "menudet"],
        "speedup": "lazyexpr_s / menudet_s",
        "execution_modes": "per-route engine actually selected: jit, interp, or other",
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
        "jit_backend_requested": args.jit,
        "jit_requested": False if args.no_jit else True if args.jit else None,
        "math_backend_requested": args.math_backend,
        "samples": args.samples,
        "warmups": args.warmups,
        "mode": "preparation-cold" if args.warmups == 0 else "warm",
        "statistic": "median",
        "qualification_timings": "preparation_s and first_compute_s are diagnostics excluded from sample medians",
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("ME_", "BLOSC_ME_")) or k in ("CC", "CFLAGS")
        },
        "protocol": (
            "median of cache-isolated samples including preparation/compilation and allocation/compression"
            if args.warmups == 0
            else "qualification + untimed warmups + median of prepared-engine samples; allocation/compression included"
        ),
        "input_layout": "C-order linspace x=[0,1], y=[1,0]; no random seed",
    }
    rows = []
    execution = {"jit": False} if args.no_jit else {"jit": True, "jit_backend": args.jit} if args.jit else {}
    if args.math_backend is not None:
        execution["math_backend"] = args.math_backend
    with blosc2.jit_options(**execution):
        require_math_backend(parser, args.math_backend)
        metadata["execution_options"] = {k: str(v) for k, v in blosc2.jit_config.execution_options().items()}
        mode = "Preparation-cold" if args.warmups == 0 else "Warm"
        print(f"LazyExpr vs Menudet. {mode} medians include allocation/compression.")
        if args.warmups == 0:
            print(
                "Cold samples include preparation/compilation; inputs are prebuilt; CPU/OS caches are not flushed."
            )
        print("Menudet speedup = LazyExpr time / Menudet time; NumPy is an untimed correctness reference.")
        print(
            "mode = engine actually selected per route (jit or interp); 'other' means neither native path."
        )
        print(f"Miniexpr math backend request: {args.math_backend or 'runtime default'}")
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
                row = bench_case(name, kernel, expr, inputs, samples=args.samples, warmups=args.warmups)
                rows.append(row)
                print(fmt.format(*format_row(row)), flush=True)
    if args.output:
        args.output.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
