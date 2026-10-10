#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################
"""Measure portable JIT coverage, correctness, cold load and warm execution.

Run from an environment built with the experimental Menudet native revision::

    python bench/jit-coverage.py --items 4096 --repeat 5
    python bench/jit-coverage.py --case 'loop/*' --backends tcc clang
    python bench/jit-coverage.py --json results.json --require-jit

This benchmarks production PortableKernel artifacts on NumPy buffers, not
compression, lazy scheduling or legacy @blosc2.jit dispatch. Cases intentionally
include safe fallbacks. Cold timing includes artifact load/JIT preparation, not
source export; existing compiler caches can make it a cached load, not a fresh
compile. Warm timing includes the Python adapter and output allocation. A prepared
kernel's has_jit flag identifies the route for these direct, nonempty evaluations.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import blosc2


@dataclass(frozen=True)
class Case:
    name: str
    body: str
    inputs: dict[str, str] = field(default_factory=lambda: {"x": "float64", "y": "float64"})
    output: str = "float64"
    constants: dict = field(default_factory=dict)
    cardinality: str = "elementwise"
    data: str = "normal"
    masked: bool = False
    expect_jit: bool = True

    def source(self):
        arguments = ", ".join([*self.inputs, *self.constants])
        return f"def k({arguments}):\n    " + self.body.replace("\n", "\n    ") + "\n"


FLOAT = {"x": "float64", "y": "float64"}
INT = {"x": "int32", "y": "int32"}
CASES = [
    Case("arithmetic/float64", "return x * 2 + y"),
    Case("arithmetic/float32", "return x * 2 + y", {"x": "float32", "y": "float32"}, "float32"),
    Case("arithmetic/int32", "return x * 2 + y", INT, "int32"),
    Case("arithmetic/mod", "return x % 7", INT, "int32"),
    Case("arithmetic/shift", "return x << 2", INT, "int32"),
    Case("arithmetic/floordiv", "return x // 3.0"),
    Case("arithmetic/power", "return x ** 2.5", data="positive"),
    Case("compare/int64-exact", "return x < y", {"x": "int64", "y": "int64"}, "bool", data="exact"),
    Case("compare/mixed-domains", "return x < y", {"x": "int64", "y": "uint64"}, "bool", data="exact"),
    Case("flow/locals", "z = x * 2\nreturn z + y"),
    Case("flow/conditional-return", "if x > 0:\n    return x\nreturn -x"),
    Case("flow/elif", "if int(x) > 0:\n    return x\nelif int(y) > 0:\n    return y\nreturn 0.0"),
    Case("flow/where", "return where(x > 0, int(y), 0)", output="int64", data="lazy-cast"),
    Case("flow/and", "return x > 0 and int(y) > 0", output="bool", data="lazy-cast"),
    Case("flow/or", "return x <= 0 or int(y) > 0", output="bool", data="lazy-cast"),
    Case("loop/range", "s = 0.0\nfor i in range(16):\n    s = s + x\nreturn s"),
    Case("loop/negative-range", "s = 0.0\nfor i in range(15, -1, -2):\n    s = s + x\nreturn s"),
    Case("loop/while", "s = 0.0\ni = 0\nwhile i < 16:\n    s = s + x\n    i = i + 1\nreturn s"),
    Case(
        "loop/break", "s = 0.0\nfor i in range(16):\n    if s > y:\n        break\n    s = s + x\nreturn s"
    ),
    Case(
        "loop/continue",
        "s = 0.0\nfor i in range(16):\n    if i < 3:\n        continue\n    s = s + x\nreturn s",
    ),
    Case("loop/return", "for i in range(16):\n    if x > 0:\n        return x\nreturn y"),
    Case("loop/nested", "s = 0.0\nfor i in range(4):\n    for j in range(4):\n        s = s + x\nreturn s"),
    Case(
        "loop/nested-exits",
        "s = 0.0\nfor i in range(4):\n    for j in range(4):\n        if j == 1:\n"
        "            continue\n        if j == 3:\n            break\n        s = s + x\nreturn s",
    ),
    Case(
        "loop/while-continue",
        "s = 0.0\ni = 0\nwhile i < 16:\n    i = i + 1\n    if i < 3:\n"
        "        continue\n    s = s + x\nreturn s",
    ),
    Case(
        "loop/mandelbrot",
        "zr = 0.0\nzi = 0.0\nn = 0\nfor i in range(limit):\n"
        "    if zr * zr + zi * zi > 4.0:\n        break\n"
        "    new_zr = zr * zr - zi * zi + x\n    zi = 2 * zr * zi + y\n"
        "    zr = new_zr\n    n = n + 1\nreturn n",
        output="int64",
        constants={"limit": 64},
        data="mandelbrot",
    ),
    *[
        Case(f"cast/{cast}", f"return {cast}(x)", output=cast, data="cast")
        for cast in ("int8", "uint8", "int32", "uint32", "int64", "uint64")
    ],
    *[
        Case(f"integer-math/{name}", f"return {name}(x)", INT, "int32")
        for name in ("abs", "sign", "square", "floor", "ceil", "trunc", "round", "real", "imag", "conj")
    ],
    *[
        Case(f"integer-math/{name}", f"return {expression}", INT, "int32", data="combinatorial")
        for name, expression in (
            ("factorial", "fac(x)"),
            ("ncr", "ncr(x, y)"),
            ("npr", "npr(x, y)"),
            ("power", "x ** y"),
        )
    ],
    Case("math/unary", "return sin(x) + cos(y)"),
    Case("math/binary", "return hypot(x, y)"),
    Case("math/ldexp", "return ldexp(x, 2)"),
    Case("math/fma", "return fma(x, y, 1.0)"),
    Case("math/fp-status", "return sqrt(x)"),
    Case("weak/integer-leaf", "return x + c", {"x": "int8"}, "int8", {"c": 7}),
    Case("weak/computed", "return x + (c + 1)", {"x": "int8"}, "int8", {"c": 7}),
    Case("weak/floating-cast", "return x + int8(c)", {"x": "int8"}, "int8", {"c": 7.75}),
    Case("mask/arithmetic", "return x * 2 + y", masked=True),
    Case("mask/checked-cast", "return int64(x)", output="int64", data="masked-cast", masked=True),
    Case("mask/loops", "s = 0.0\nfor i in range(16):\n    s = s + x\nreturn s", masked=True),
    *[
        Case(f"reduction/{name}", f"return {body}", cardinality="block_scalar", data="positive")
        for name, body in (
            ("sum-direct", "block_sum(x)"),
            ("sum-map", "block_sum(x * 2)"),
            ("prod-direct", "block_prod(x)"),
            ("prod-map", "block_prod(x + y)"),
            ("sum-math", "block_sum(sin(x))"),
        )
    ],
    Case("reduction/masked-map", "return block_sum(x * 2)", cardinality="block_scalar", masked=True),
    Case(
        "fallback/integer-reduction",
        "return block_sum(x)",
        INT,
        "int64",
        cardinality="block_scalar",
        expect_jit=False,
    ),
    Case(
        "fallback/multiple-reductions",
        "return block_sum(x) + block_sum(y)",
        cardinality="block_scalar",
        expect_jit=False,
    ),
    Case(
        "fallback/scalar-statements",
        "s = block_sum(x)\nreturn s + 1",
        cardinality="block_scalar",
        expect_jit=False,
    ),
]


def operands(case, count):
    index = np.arange(count)
    x = (index % 31 - 15) / 8
    y = (index % 17 - 8) / 4
    if case.data == "positive":
        # Near-one products avoid making overflow the only measured workload.
        x = 1 + (index % 7 - 3) * 1e-5
        y = (index % 5 - 2) * 1e-5
    elif case.data == "cast":
        x = (index % 127) + 0.75
    elif case.data == "combinatorial":
        x = index % 8
        y = np.minimum(index % 3, x)
    elif case.data == "mandelbrot":
        width = max(1, int(np.sqrt(count)))
        x = -2 + 3 * (index % width) / max(1, width - 1)
        y = -1.5 + 3 * (index // width) / max(1, (count - 1) // width)
    elif case.data == "exact":
        x = np.resize(np.array([-(2**63), -1, 0, 2**53, 2**63 - 1], dtype="int64"), count)
        values = (
            [0, 2**64 - 1, 0, 2**53 + 1, 2**64 - 1]
            if case.inputs["y"] == "uint64"
            else [-(2**63), 0, -1, 2**53 + 1, 2**63 - 1]
        )
        y = np.resize(np.array(values, dtype=case.inputs["y"]), count)
    elif case.data == "lazy-cast":
        y = np.where(x > 0, 2.75, np.nan)
    elif case.data == "masked-cast":
        x = np.where(index % 3 != 0, 2.75, np.nan)
    values = {"x": x, "y": y}
    arrays = {name: np.asarray(values[name], dtype=dtype) for name, dtype in case.inputs.items()}
    mask = np.asarray(index % 3 != 0) if case.masked else None
    return arrays, mask


@contextmanager
def backend_environment(backend, compiler):
    changes = {"ME_DSL_JIT_COMPILER": "tcc" if backend == "tcc" else "cc"}
    if compiler:
        changes["CC"] = compiler
    previous = {key: os.environ.get(key) for key in changes}
    os.environ.update(changes)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def compiler_for(backend, args):
    if backend == "tcc":
        return None  # Bundled libtcc need not provide a tcc executable on PATH.
    requested = getattr(args, backend)
    candidates = (
        (requested,)
        if requested
        else (("gcc-16", "gcc-15", "gcc-14", "gcc") if backend == "gcc" else ("clang",))
    )
    for candidate in candidates:
        path = shutil.which(candidate)
        if not path:
            continue
        version = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
        description = (version.stdout + version.stderr).lower()
        # macOS /usr/bin/gcc is Clang, not a separate GCC backend.
        if (backend == "gcc" and "clang" not in description) or (
            backend == "clang" and "clang" in description
        ):
            return path
    return None


def check_result(actual, expected, mask, cardinality):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        raise AssertionError(
            f"result signature differs: {actual.dtype}/{actual.shape}, {expected.dtype}/{expected.shape}"
        )
    if mask is not None and cardinality == "elementwise":
        # Nonparticipating output lanes have unspecified contents.
        actual, expected = actual[mask], expected[mask]
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == "f":
        participating = ~np.isnan(expected)
        # Keep signed-zero and rounding checks; NaN payload/sign is not portable.
        if actual[participating].tobytes() != expected[participating].tobytes():
            raise AssertionError("non-NaN floating result bits differ")


def measure(kernel, arrays, mask, repeat):
    def evaluate():
        return kernel.evaluate_block(arrays, valid_mask=mask)

    evaluate()  # Warm-up is outside reported timings.
    samples = []
    for _ in range(repeat):
        start = time.perf_counter_ns()
        evaluate()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"best_ms": min(samples), "median_ms": statistics.median(samples), "samples_ms": samples}


def benchmark_reference(case, arrays, mask, repeat):
    artifact = blosc2.DSLKernel.from_source(case.source()).export(
        case.inputs,
        case.output,
        constants=case.constants,
        version="1.1",
        cardinality=case.cardinality,
    )
    start = time.perf_counter_ns()
    reference = blosc2.PortableKernel.from_json(artifact, jit=False)
    cold = (time.perf_counter_ns() - start) / 1e6
    expected, status = reference.evaluate_block(arrays, valid_mask=mask, return_status=True)
    return (
        artifact,
        expected,
        status,
        {
            "route": "interpreter",
            "cold_load_ms": cold,
            "fp_status": status,
            **measure(reference, arrays, mask, repeat),
        },
    )


def benchmark_backend(case, artifact, arrays, mask, expected, expected_status, backend, compiler, args):
    if backend != "tcc" and compiler is None:
        return {"route": "unavailable"}
    try:
        with backend_environment(backend, compiler):
            start = time.perf_counter_ns()
            kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
            cold = (time.perf_counter_ns() - start) / 1e6
            actual, status = kernel.evaluate_block(arrays, valid_mask=mask, return_status=True)
            check_result(actual, expected, mask, case.cardinality)
            if status != expected_status:
                raise AssertionError(f"FP status differs: {status} != {expected_status}")
            route = "jit" if kernel.has_jit else "fallback"
            return {
                "route": route,
                "cold_load_ms": cold,
                "fp_status": status,
                **measure(kernel, arrays, mask, args.repeat),
            }
    except Exception as error:
        return {"route": "invalid", "error": str(error)}


def print_row(record, backends, color):
    reference = record["interpreter"]["median_ms"]
    medians = [reference]
    cells = [f"{reference:.3f} I"]
    for backend in backends:
        result = record["backends"][backend]
        if "median_ms" in result:
            ms = result["median_ms"]
            medians.append(ms)
            cells.append(f"{ms:.3f} {'JIT' if result['route'] == 'jit' else 'fb'} {reference / ms:.1f}x")
        else:
            medians.append(float("inf"))
            cells.append(result["route"])
    fastest = min(range(len(medians)), key=medians.__getitem__)
    rendered = []
    for index, cell in enumerate(cells):
        cell = f"{cell:>{14 if index == 0 else 24}s}"
        rendered.append(f"\033[1;32m{cell}\033[0m" if color and index == fastest else cell)
    print(f"{record['case']:34s}" + " ".join(rendered), flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=int, default=4096)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--backends", nargs="+", choices=("tcc", "gcc", "clang"), default=["tcc", "gcc", "clang"]
    )
    parser.add_argument("--gcc", help="GCC executable (auto-detected by default)")
    parser.add_argument("--clang", help="Clang executable (auto-detected by default)")
    parser.add_argument(
        "--case", action="append", default=[], help="case-name glob; repeat to select multiple families"
    )
    parser.add_argument("--list", action="store_true", help="list selected cases without loading kernels")
    parser.add_argument(
        "--require-jit",
        action="store_true",
        help="fail if an expected eligible case falls back or a backend is missing",
    )
    parser.add_argument("--json", type=Path, help="write timings, routes and environment metadata")
    args = parser.parse_args()
    if args.items < 1 or args.repeat < 1:
        parser.error("--items and --repeat must be positive")
    selected = [
        case
        for case in CASES
        if not args.case or any(fnmatch.fnmatchcase(case.name, pattern) for pattern in args.case)
    ]
    if not selected:
        parser.error("no cases match --case")
    if args.list:
        for case in selected:
            print(f"{case.name:34s} expected={'jit' if case.expect_jit else 'fallback'}")
        return 0
    backends = list(dict.fromkeys(args.backends))
    compilers = {backend: compiler_for(backend, args) for backend in backends}
    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "blosc2": blosc2.__version__,
        "numpy": np.__version__,
        "items": args.items,
        "repeat": args.repeat,
        "compilers": compilers,
        "environment": {
            key: os.environ.get(key)
            for key in ("CFLAGS", "ME_DSL_JIT_TCC_OPTIONS", "ME_DSL_JIT_CACHE_DIR", "ME_DSL_WHILE_MAX_ITERS")
        },
    }
    records = []
    failures = []
    totals = {backend: {"jit": 0, "fallback": 0, "unavailable": 0, "invalid": 0} for backend in backends}
    color = "NO_COLOR" not in os.environ and (sys.stdout.isatty() or os.environ.get("FORCE_COLOR") == "1")
    print(f"Portable JIT coverage: {len(selected)} cases, {args.items} items, {args.repeat} warm samples")
    print(
        "Cells: median ms [JIT/fb], speedup vs interpreter; cold load ms in JSON. Fastest valid median highlighted."
    )
    print(
        "C/JIT caches are preserved; cold load may reuse cached code. Nonparticipating lanes are not compared."
    )
    for backend, compiler in compilers.items():
        print(f"{backend}: {compiler or ('bundled/runtime libtcc' if backend == 'tcc' else 'unavailable')}")
    print(f"{'case':34s} {'interpreter':>14s}" + "".join(f"{backend:>24s}" for backend in backends))
    for case in selected:
        arrays, mask = operands(case, args.items)
        try:
            artifact, expected, expected_status, timing = benchmark_reference(
                case, arrays, mask, args.repeat
            )
        except Exception as error:
            raise RuntimeError(f"{case.name}: interpreter reference failed: {error}") from error
        record = {
            "case": case.name,
            "expected_jit": case.expect_jit,
            "source": case.source(),
            "inputs": case.inputs,
            "output": case.output,
            "constants": case.constants,
            "cardinality": case.cardinality,
            "masked": case.masked,
            "interpreter": timing,
            "backends": {},
        }
        for backend in backends:
            result = benchmark_backend(
                case, artifact, arrays, mask, expected, expected_status, backend, compilers[backend], args
            )
            if result["route"] == "invalid":
                failures.append(f"{case.name}/{backend}: {result['error']}")
            elif args.require_jit and case.expect_jit and result["route"] != "jit":
                failures.append(f"{case.name}/{backend}: expected JIT, got {result['route']}")
            totals[backend][result["route"]] += 1
            record["backends"][backend] = result
        print_row(record, backends, color)
        records.append(record)
    print("\nRoute counts (selected cases, including intentional fallbacks):")
    for backend, counts in totals.items():
        print(f"  {backend}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    if args.json:
        args.json.write_text(
            json.dumps(
                {"metadata": metadata, "cases": records, "route_counts": totals, "failures": failures},
                indent=2,
            )
            + "\n"
        )
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
