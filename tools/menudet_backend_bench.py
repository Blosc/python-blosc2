"""M1 sign-off baselines: verified full-DSL TCC/GCC controls and portable interpreter.

Every sample is a subprocess. Cold compilation, fresh-process disk reuse,
same-process recompilation, first evaluation and compiled warm execution are
separate. No user files/caches are removed; each trial gets an owned directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def worker(engine, dtype, count, repeats):
    import numexpr

    import blosc2

    x = np.arange(count, dtype=dtype)
    compile_ns = 0
    if engine == "integrated":
        artifact = json.dumps(
            {
                "schema_version": "1.0",
                "language": {"name": "miniexpr", "version": "1.0"},
                "requires": ["numeric"],
                "source": "def k(x):\n    return x * 2 + 1\n",
                "entry_point": "k",
                "inputs": [{"name": "x", "dtype": dtype}],
                "constants": [],
                "output": {"dtype": dtype, "contract": "elementwise"},
                "semantics": {"fp": "strict"},
                "context": {"ndim": 0},
                "metadata": {},
            }
        )
        start = time.perf_counter_ns()
        kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
        compile_ns = time.perf_counter_ns() - start
        if kernel.has_jit:
            raise AssertionError("Update baseline eligibility before benchmarking a new portable JIT")

        def evaluate():
            return kernel.evaluate({"x": x})
    elif engine == "numpy":

        def evaluate():
            return x * 2 + 1
    else:

        def evaluate():
            return numexpr.evaluate("x * 2 + 1", local_dict={"x": x})

    samples = []
    for _ in range(repeats + 1):
        start = time.perf_counter_ns()
        actual = evaluate()
        samples.append(time.perf_counter_ns() - start)
        np.testing.assert_array_equal(actual, x * 2 + 1)
        assert actual.dtype == x.dtype
    return {
        "requested": engine,
        "backend": "interpreter" if engine == "integrated" else engine,
        "dtype": dtype,
        "nitems": count,
        "compile_ns": compile_ns,
        "first_ns": samples[0],
        "warm_median_ns": int(np.median(samples[1:])),
        "numexpr_threads": numexpr.get_num_threads(),
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
    }


def run(args):
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    owned = Path(tempfile.mkdtemp(prefix="m1-baseline-", dir=root))
    log = owned / "compiler-invocations.txt"
    wrapper = owned / "gcc-wrapper"
    gcc_version = None
    if args.gcc:
        gcc_version = subprocess.check_output([str(args.gcc), "--version"], text=True).splitlines()[0]
        if "clang" in gcc_version.lower():
            raise ValueError("--gcc must name actual GCC, not the macOS gcc alias to Clang")
        wrapper.write_text(
            f'#!/bin/sh\nprintf \'%s\\n\' "$*" >> {shlex.quote(str(log))}\nexec {shlex.quote(str(args.gcc.resolve()))} "$@"\n'
        )
        wrapper.chmod(0o700)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("ME_DSL_") and key not in {"CC", "CFLAGS"}
    }
    environment.update(ME_DSL_FP_MODE="strict", ME_DSL_JIT="1", CC=str(wrapper), NUMEXPR_NUM_THREADS="1")
    backends = ["portable", "interpreter", "tcc", "integrated", "numpy", "numexpr"]
    if args.gcc:
        backends.append("cc")
    rows = []

    def compile_calls():
        return (
            sum("-dynamiclib" in line or "-shared" in line for line in log.read_text().splitlines())
            if log.exists()
            else 0
        )

    for trial in range(args.trials):
        for dtype in ("float32", "float64", "int64"):
            for count in (16, args.large_count):
                order = backends if trial % 2 == 0 else list(reversed(backends))
                for backend in order:
                    cache = owned / f"trial-{trial}-{dtype}-{count}-{backend}"
                    cache.mkdir()
                    env = {**environment, "ME_DSL_JIT_CACHE_DIR": str(cache)}
                    native = backend in {"portable", "interpreter", "tcc", "cc"}
                    command = (
                        [str(args.native_runner.resolve()), backend, dtype, str(count), str(args.repeats)]
                        if native
                        else [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            backend,
                            "--dtype",
                            dtype,
                            "--count",
                            str(count),
                            "--repeats",
                            str(args.repeats),
                        ]
                    )
                    for phase in ["cold", "disk_cache_reuse"] if backend == "cc" else ["cold"]:
                        before = compile_calls()
                        start = time.perf_counter_ns()
                        proc = subprocess.run(command, env=env, capture_output=True, text=True, check=True)
                        process_ns = time.perf_counter_ns() - start
                        row = json.loads(proc.stdout)
                        row.update(
                            trial=trial,
                            phase=phase,
                            process_ns=process_ns,
                            compiler_invocations=compile_calls() - before,
                            route="direct_native" if native else "python",
                        )
                        row["semantic_contract"] = (
                            "full-miniexpr-control"
                            if backend in {"interpreter", "tcc", "cc"}
                            else "menudet-draft-1.0-checked"
                            if backend in {"portable", "integrated"}
                            else "control"
                        )
                        if backend == "cc":
                            assert row["compiler_invocations"] == (1 if phase == "cold" else 0), row
                            row["compiler"] = gcc_version
                        rows.append(row)
    return {
        "schema_version": "menudet-backend-baseline-1",
        "numpy": np.__version__,
        "platform": platform.platform(),
        "native_revision": args.native_revision,
        "installed_native_revision": args.installed_native_revision,
        "native_runner_sha256": hashlib.sha256(args.native_runner.read_bytes()).hexdigest(),
        "gcc": gcc_version,
        "trials": args.trials,
        "repeats": args.repeats,
        "owned_cache_directory": str(owned),
        "order": "alternating forward/reverse subprocess order",
        "results": rows,
        "skips": ([] if args.gcc else [{"backend": "gcc", "reason": "No explicit GCC executable supplied"}])
        + [
            {
                "backend": "tcc",
                "phase": "disk_cache_reuse",
                "reason": "TCC compiles in memory; no persistent disk artifact cache",
            }
        ],
        "limitations": "Single host, affine positive-domain kernel only; native output preallocated, Python output allocated; no RSS/scaling claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["integrated", "numpy", "numexpr"])
    parser.add_argument("--dtype", default="float64")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--large-count", type=int, default=262144)
    parser.add_argument("--native-runner", type=Path)
    parser.add_argument("--gcc", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--native-revision")
    parser.add_argument("--installed-native-revision")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 101 or args.trials < 1:
        parser.error("Positive trial count and 1..101 repetitions required")
    if args.worker:
        print(json.dumps(worker(args.worker, args.dtype, args.count, args.repeats)))
    else:
        if not args.native_runner or not args.work_dir or not args.report:
            parser.error("--native-runner, --work-dir and --report are required")
        args.report.write_text(json.dumps(run(args), indent=2) + "\n")


if __name__ == "__main__":
    main()
