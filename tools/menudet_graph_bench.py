"""Isolated M5/M6 performance evidence, not a universal speed budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import blosc2
from blosc2.native_graph import compile_plan


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def worker(size, dtype, route, layout, reduction):
    x = np.linspace(0, 1, size, dtype=dtype)
    if layout == "strided":
        x = np.repeat(x, 2)[::2]
    # Keep comparison semantics common to every route.
    source = "x * 2 + 1"
    shape = (size,)
    expr = None
    compile_ms = None
    if route in {"native", "graph", "blosc2"}:
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(source, {"x": x})
    if route in {"native", "graph"}:
        compile_plan.cache_clear()
        plan, compile_ms = timed(expr.native_kernel)
    build = time.perf_counter_ns()
    if route == "native":
        bound = {name: value for name, value in expr.operands.items() if name in plan.input_dtypes}

        def call():
            return plan.evaluate_array(bound, reduction="sum" if reduction else None, return_report=True)
    elif route == "graph":

        def call():
            return (
                expr.sum(_require_native=True)
                if reduction
                else expr.compute(_require_native=True, _getitem=True)
            )
    elif route == "blosc2":

        def call():
            return expr.sum() if reduction else expr[:]
    elif route == "numpy":

        def call():
            return np.sum(x * 2 + 1) if reduction else x * 2 + 1
    elif route == "numexpr":
        import numexpr

        numexpr.set_num_threads(1)

        def call():
            return numexpr.evaluate("sum(x * 2 + 1)" if reduction else source, local_dict={"x": x})
    else:
        raise ValueError(route)
    setup_ms = (time.perf_counter_ns() - build) / 1e6
    first, first_ms = timed(call)
    warm = [timed(call)[1] for _ in range(5)]
    native_report = first[1] if route == "native" else getattr(expr, "_native_execution_report", None)
    values = first[0] if route == "native" else first
    reference = np.sum(x * 2 + 1) if reduction else x * 2 + 1
    np.testing.assert_allclose(values, reference, rtol=1e-3 if reduction else 1e-6)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "route": route,
        "size": size,
        "dtype": dtype,
        "layout": layout,
        "reduction": reduction,
        "compile_ms": compile_ms,
        "setup_ms": setup_ms,
        "first_ms": first_ms,
        "warm_median_ms": statistics.median(warm),
        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "native_report": native_report,
        "shape": shape,
        "backend": "portable-interpreter" if route in {"native", "graph"} else route,
        "threads": 1,
        "iterator_tile_items": 1024,
        "numpy": np.__version__,
        "package": blosc2.__file__,
    }


def supplement(scenario):
    from concurrent.futures import ThreadPoolExecutor
    from tempfile import TemporaryDirectory

    x = np.linspace(0, 1, 8192, dtype="float64")
    with TemporaryDirectory(dir=os.environ.get("MENUDET_WORK_DIR")) as directory:
        root = Path(directory)
        operands = {"x": x}
        expression = "x * 2 + 1"
        reference = x * 2 + 1
        if scenario == "mixed":
            operands = {"x": x.astype("float32"), "y": np.ones(1, dtype="float64")}
            expression = "x + y"
            reference = operands["x"] + operands["y"]
        if scenario in {"compressed", "persisted"}:
            operands["x"] = blosc2.asarray(x, urlpath=root / "input.b2nd", chunks=(1024,), blocks=(256,))
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, operands)
        plan, compile_ms = timed(expr.native_kernel)
        if scenario == "partial":

            def call():
                return expr.compute(slice(128, 2048, 2), _require_native=True, _getitem=True)

            reference = reference[128:2048:2]
        elif scenario == "persisted":
            bindings = {n: v for n, v in expr.operands.items() if n in plan.input_dtypes}
            portable = plan.lazy(bindings, partitions=(1024,))
            portable.save(root / "recipe.b2nd")
            loaded, load_ms = timed(lambda: blosc2.open(root / "recipe.b2nd", deserialize="safe"))

            def call():
                return loaded[:]
        elif scenario == "threaded":
            bindings = {n: v for n, v in expr.operands.items() if n in plan.input_dtypes}
            pool = ThreadPoolExecutor(max_workers=4)

            def call():
                return list(pool.map(lambda _: plan.evaluate_array(bindings), range(4)))[0]
        else:

            def call():
                return expr.compute(_require_native=True, _getitem=True)

        first, first_ms = timed(call)
        warm = [timed(call)[1] for _ in range(5)]
        np.testing.assert_array_equal(first, reference)
        if scenario == "threaded":
            pool.shutdown()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {
            "scenario": scenario,
            "compile_ms": compile_ms,
            "first_ms": first_ms,
            "warm_median_ms": statistics.median(warm),
            "load_ms": load_ms if scenario == "persisted" else None,
            "size": 8192,
            "threads": 4 if scenario == "threaded" else 1,
            "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
            "native_report": getattr(expr, "_native_execution_report", None),
            "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
            "python_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "note": "Four independent concurrent calls"
            if scenario == "threaded"
            else "Existing portable recipe block scheduler"
            if scenario == "persisted"
            else "Native logical scheduler",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--scenario", choices=["mixed", "partial", "compressed", "persisted", "threaded"])
    parser.add_argument("--supplement-existing", action="store_true")
    args = parser.parse_args()
    if args.scenario:
        print(json.dumps(supplement(args.scenario)))
        return
    if args.worker:
        size, dtype, route, layout, reduction = args.worker
        print(json.dumps(worker(int(size), dtype, route, layout, reduction == "sum")))
        return
    if args.report is None:
        parser.error("--report is required")
    if args.supplement_existing:
        report = json.loads(args.report.read_text())
        report["supplemental"] = []
        for scenario in ["mixed", "partial", "compressed", "persisted", "threaded"]:
            p = subprocess.run(
                [sys.executable, "-m", "tools.menudet_graph_bench", "--scenario", scenario],
                capture_output=True,
                text=True,
                check=True,
            )
            report["supplemental"].append(json.loads(p.stdout))
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        return
    rows = []
    cases = [
        (n, d, layout, r)
        for n in (8, 262144)
        for d in ("float32", "float64", "int64")
        for layout in ("C", "strided")
        for r in ("none", "sum")
    ]
    for index, (size, dtype, layout, reduction) in enumerate(cases):
        routes = ["native", "graph", "numpy", "numexpr", "blosc2"]
        if index % 2:
            routes.reverse()
        for route in routes:
            environment = {**os.environ, "BLOSC_NTHREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
            p = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tools.menudet_graph_bench",
                    "--worker",
                    str(size),
                    dtype,
                    route,
                    layout,
                    reduction,
                ],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
            rows.append(json.loads(p.stdout))
    args.report.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "rows": rows,
                "limitations": [
                    "Process peak RSS includes interpreter/imports/reference allocations.",
                    "One serial native thread; compressed input frontend not timed in this matrix.",
                    "Compilation measured for native plan; control-route parsing is included in first run.",
                    "No universal speed budget or speedup claim.",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
