"""Isolated actual-JIT comparisons; compilation excluded from warm execution."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import blosc2

EXPRESSIONS = {
    "affine": "x * 2 + y / 3 - 1",
    "polynomial": "(x + y) * (x - y)",
    "where": "where(x > 0, y / x, x * 2 + y)",
}


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def worker(route, workload, dtype, size, api):
    x = np.linspace(-2, 2, size, dtype=dtype)
    y = np.linspace(1, 2, size, dtype=dtype)
    bindings = {"x": x, "y": y}
    expression = EXPRESSIONS[workload]
    compile_ms = None
    has_jit = False
    if route in {"tcc", "gcc", "interpreter"}:
        os.environ["ME_DSL_JIT_COMPILER"] = "cc" if route == "gcc" else "tcc"
        if route == "gcc":
            os.environ["CC"] = os.environ.get(
                "MENUDET_GCC", shutil.which("gcc-16") or shutil.which("gcc") or "cc"
            )
        source = f"def k(x, y):\n    return {expression}\n"
        artifact = blosc2.DSLKernel.from_source(source).export(
            {"x": dtype, "y": dtype}, dtype, version="1.1"
        )
        kernel, compile_ms = timed(
            lambda: blosc2.PortableKernel.from_json(artifact, jit=route != "interpreter")
        )
        has_jit = kernel.has_jit
        if has_jit != (route != "interpreter"):
            raise RuntimeError(f"{route} backend eligibility/execution mismatch")

        def call():
            return kernel.evaluate(bindings) if api == "flat" else kernel.evaluate_array(bindings)
    elif route == "numpy":

        def call():
            if workload == "affine":
                return x * 2 + y / 3 - 1
            if workload == "polynomial":
                return (x + y) * (x - y)
            return np.where(x > 0, y / x, x * 2 + y)
    else:
        import numexpr

        numexpr.set_num_threads(1)

        def call():
            return numexpr.evaluate(expression, local_dict=bindings)

    first, first_ms = timed(call)
    warm = [timed(call)[1] for _ in range(3 if route == "interpreter" else 9)]
    reference = (
        x * 2 + y / 3 - 1
        if workload == "affine"
        else (x + y) * (x - y)
        if workload == "polynomial"
        else np.where(x > 0, y / x, x * 2 + y)
    )
    np.testing.assert_array_equal(first, reference)
    return {
        "route": route,
        "workload": workload,
        "dtype": dtype,
        "size": size,
        "api": api,
        "compile_ms": compile_ms,
        "first_ms": first_ms,
        "warm_median_ms": statistics.median(warm),
        "warm_samples_ms": warm,
        "has_jit": has_jit,
        "threads": 1,
        "tile_items": 1024 if api == "array" else None,
        "compiler": os.environ.get("CC") if route == "gcc" else route,
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", nargs=5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.worker:
        route, workload, dtype, size, api = args.worker
        with TemporaryDirectory(dir=os.environ["MENUDET_WORK_DIR"]) as cache:
            os.environ["ME_DSL_JIT_CACHE_DIR"] = cache
            print(json.dumps(worker(route, workload, dtype, int(size), api)))
        return
    if not args.report:
        parser.error("--report is required")
    rows = []
    for size in (8, 262144):
        for dtype in ("float32", "float64"):
            for workload in EXPRESSIONS:
                for route in ("interpreter", "tcc", "gcc", "numpy", "numexpr"):
                    for api in ("flat", "array") if route in {"tcc", "gcc"} else ("flat",):
                        p = subprocess.run(
                            [
                                sys.executable,
                                "-m",
                                "tools.menudet_jit_bench",
                                "--worker",
                                route,
                                workload,
                                dtype,
                                str(size),
                                api,
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        rows.append(json.loads(p.stdout))
    args.report.write_text(
        json.dumps(
            {
                "platform": platform.platform(),
                "numpy": np.__version__,
                "rows": rows,
                "limitations": [
                    "C-contiguous finite workloads; selected-lane diagnostics qualified separately.",
                    "No explicit SIMD added; GCC may autovectorize strict loops.",
                    "Native comparison bridge retains interpreter comparison semantics.",
                    "Fresh-process artifact compilation includes metadata inference, not export.",
                    "Compilation/first/warm separate; no process-wide speedup claim.",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
