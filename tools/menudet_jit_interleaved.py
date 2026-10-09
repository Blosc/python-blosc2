"""Compare warm JIT/array-engine timings with rotating order in fresh processes."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numexpr
import numpy as np

import blosc2
from tools.menudet_jit_bench import EXPRESSIONS, timed


def worker():  # noqa: C901 -- independent compile, warm-up, validation and sampling phases
    """Compilation and correctness checks are outside all execution samples."""
    numexpr.set_num_threads(1)
    x = np.linspace(-2, 2, 262144, dtype="float64")
    y = np.linspace(1, 2, 262144, dtype="float64")
    bindings = {"x": x, "y": y}
    compilers = {
        "gcc": os.environ.get("MENUDET_GCC", shutil.which("gcc-16") or shutil.which("gcc")),
        "clang": os.environ.get("MENUDET_CLANG", shutil.which("clang")),
    }
    if not all(compilers.values()):
        raise RuntimeError("Both GCC and Clang must be available")
    compiled = {}
    compile_rows = []
    for workload, expression in EXPRESSIONS.items():
        artifact = blosc2.DSLKernel.from_source(f"def k(x, y):\n    return {expression}\n").export(
            {"x": "float64", "y": "float64"}, "float64", version="1.1"
        )
        compiled[workload] = {}
        for backend in ("interpreter", "tcc", "gcc", "clang"):
            os.environ["ME_DSL_JIT_COMPILER"] = "cc" if backend in compilers else "tcc"
            if backend in compilers:
                os.environ["CC"] = compilers[backend]
            kernel, ms = timed(
                lambda a=artifact, b=backend: blosc2.PortableKernel.from_json(a, jit=b != "interpreter")
            )
            if kernel.has_jit != (backend != "interpreter"):
                raise RuntimeError(f"Unexpected JIT eligibility: {workload}/{backend}")
            compiled[workload][backend] = kernel
            compile_rows.append({"workload": workload, "backend": backend, "compile_ms": ms})

    def numpy_call(workload):
        if workload == "affine":
            return x * 2 + y / 3 - 1
        if workload == "polynomial":
            return (x + y) * (x - y)
        return np.where(x > 0, y / x, x * 2 + y)

    calls = {}
    for workload, expression in EXPRESSIONS.items():
        calls[workload] = {
            backend: lambda k=kernel: k.evaluate(bindings) for backend, kernel in compiled[workload].items()
        }
        calls[workload]["numpy"] = lambda w=workload: numpy_call(w)
        calls[workload]["numexpr"] = lambda e=expression: numexpr.evaluate(e, local_dict=bindings)
        expected = numpy_call(workload)
        # One untimed complete execution per backend warms code and engine caches.
        for call in calls[workload].values():
            np.testing.assert_array_equal(call(), expected)

    rows = []
    workloads = list(EXPRESSIONS)
    for round_number in range(30):
        # Rotate workload order as well as backend order; all backends see the
        # same buffers in the same process. The slow interpreter takes three
        # samples, matching the original benchmark's sample budget.
        for slot in range(len(workloads)):
            workload = workloads[(slot + round_number) % len(workloads)]
            backends = list(calls[workload])
            if round_number >= 3:
                backends.remove("interpreter")
            shift = round_number % len(backends)
            for position, backend in enumerate(backends[shift:] + backends[:shift]):
                result, ms = timed(calls[workload][backend])
                rows.append(
                    {
                        "workload": workload,
                        "backend": backend,
                        "round": round_number,
                        "position": position,
                        "ms": ms,
                    }
                )
                del result
    for workload in workloads:
        expected = numpy_call(workload)
        for call in calls[workload].values():
            np.testing.assert_array_equal(call(), expected)
    return {
        "samples": rows,
        "compile_rows": compile_rows,
        "compilers": compilers,
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--processes", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        with TemporaryDirectory(dir=os.environ["MENUDET_WORK_DIR"]) as cache:
            os.environ["ME_DSL_JIT_CACHE_DIR"] = cache
            print(json.dumps(worker()))
        return
    if args.report is None or args.processes < 1:
        parser.error("--report and a positive --processes are required")
    processes = []
    for _ in range(args.processes):
        result = subprocess.run(
            [sys.executable, "-m", "tools.menudet_jit_interleaved", "--worker"],
            capture_output=True,
            text=True,
            check=True,
        )
        processes.append(json.loads(result.stdout))
    summary = []
    for workload in EXPRESSIONS:
        for backend in ("interpreter", "tcc", "gcc", "clang", "numpy", "numexpr"):
            medians = [
                statistics.median(
                    r["ms"] for r in p["samples"] if r["workload"] == workload and r["backend"] == backend
                )
                for p in processes
            ]
            summary.append(
                {
                    "workload": workload,
                    "backend": backend,
                    "median_of_process_medians_ms": statistics.median(medians),
                    "process_medians_ms": medians,
                }
            )
    report = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numexpr": numexpr.__version__,
        "size": 262144,
        "dtype": "float64",
        "api": "flat",
        "numexpr_threads": 1,
        "technique": "Rotating workload/backend order; untimed warm-up; identical shared buffers",
        "samples_per_process": {"interpreter": 3, "other_backends": 30},
        "compiler_versions": {
            name: subprocess.check_output([compiler, "--version"], text=True)
            for name, compiler in processes[0]["compilers"].items()
        },
        "summary": summary,
        "processes": processes,
        "limitations": [
            "No core affinity or frequency tracing; not proof of the prior timing cause.",
            "Finite C-contiguous workloads; compilation excluded from warm timings.",
            "Interpreter has a smaller sample budget and participates in initial rounds only.",
        ],
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for workload in EXPRESSIONS:
        print(
            workload,
            {
                r["backend"]: round(r["median_of_process_medians_ms"], 3)
                for r in summary
                if r["workload"] == workload
            },
        )


if __name__ == "__main__":
    main()
