"""Repeat the native lowering-coverage benchmark across fresh processes.

This preserves the native harness's fixed order / best-of-warm methodology;
it measures coverage and interpreter speedups, not stable compiler rankings.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--processes", type=int, default=5)
    args = parser.parse_args()
    if args.processes < 1:
        parser.error("--processes must be positive")
    pattern = re.compile(r"^(.+?)\s+([\d.]+) I\s+([\d.]+) (JIT|fb)\s+([\d.]+) (JIT|fb)\s+([\d.]+) (JIT|fb)$")
    processes = []
    raw = []
    for _ in range(args.processes):
        with TemporaryDirectory(dir=os.environ["MENUDET_WORK_DIR"]) as cache:
            environment = dict(os.environ, ME_DSL_JIT_CACHE_DIR=cache)
            result = subprocess.run(
                [str(args.executable.resolve()), "65536", "7"],
                env=environment,
                capture_output=True,
                text=True,
                check=True,
            )
        if "value checks: all match interpreter" not in result.stdout:
            raise RuntimeError(result.stdout + result.stderr)
        rows = []
        for line in result.stdout.splitlines():
            match = pattern.match(line)
            if match:
                values = match.groups()
                rows.append(
                    {
                        "case": values[0],
                        "interpreter_ms": float(values[1]),
                        "tcc_ms": float(values[2]),
                        "tcc_route": values[3],
                        "gcc_ms": float(values[4]),
                        "gcc_route": values[5],
                        "clang_ms": float(values[6]),
                        "clang_route": values[7],
                    }
                )
        if len(rows) != 8:
            raise RuntimeError(f"Unexpected benchmark output:\n{result.stdout}")
        processes.append(rows)
        raw.append(result.stdout)
    summary = []
    for index, row in enumerate(processes[0]):
        entry = {"case": row["case"]}
        for backend in ("interpreter", "tcc", "gcc", "clang"):
            samples = [process[index][f"{backend}_ms"] for process in processes]
            entry[f"{backend}_ms"] = statistics.median(samples)
            entry[f"{backend}_range_ms"] = [min(samples), max(samples)]
            if backend != "interpreter":
                routes = {process[index][f"{backend}_route"] for process in processes}
                if len(routes) != 1:
                    raise RuntimeError(f"Inconsistent compilation: {row['case']}/{backend}")
                entry[f"{backend}_route"] = routes.pop()
        summary.append(entry)
    report = {
        "platform": platform.platform(),
        "nitems": 65536,
        "repeats": 7,
        "technique": "Median across fresh processes of native best-of-warm timings; fixed backend order",
        "limitations": [
            "Compilation excluded; not an interleaved compiler-ranking experiment.",
            "Native harness compares result bytes with the interpreter.",
        ],
        "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "linked_native_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.executable.resolve().parent.parent.glob("*miniexpr*")
            if path.suffix in (".dylib", ".so", ".dll")
        },
        "compiler_versions": {
            compiler: subprocess.check_output([compiler, "--version"], text=True)
            for compiler in ("gcc-16", "clang")
        },
        "summary": summary,
        "processes": processes,
        "raw_output": raw,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
