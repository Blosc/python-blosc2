"""Check exhaustive bulk-compiled corpus rows against the normal serial TCC route."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("corpus_directory", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    compilers = {
        "tcc": None,
        "gcc": os.environ.get("MENUDET_GCC", shutil.which("gcc-16")),
        "clang": os.environ.get("MENUDET_CLANG", shutil.which("clang")),
    }
    if not compilers["gcc"] or not compilers["clang"]:
        parser.error("GCC and Clang are required")
    results = []
    for corpus in ("arithmetic-v1.1.json", "functions-v1.1.json"):
        reference = None
        for backend, compiler in compilers.items():
            with TemporaryDirectory(dir=os.environ["MENUDET_WORK_DIR"]) as cache:
                environment = dict(
                    os.environ, ME_DSL_JIT_COMPILER="cc" if compiler else "tcc", ME_DSL_JIT_CACHE_DIR=cache
                )
                environment.pop("MENUDET_JIT_SERIAL", None)
                if compiler:
                    environment["CC"] = compiler
                start = time.perf_counter()
                process = subprocess.run(
                    [str(args.runner.resolve()), str((args.corpus_directory / corpus).resolve()), "on"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=60,
                )
                elapsed = time.perf_counter() - start
            rows = [json.loads(line) for line in process.stdout.splitlines()]
            assert all(row["reference_match"] for row in rows)
            if reference is None:
                reference = rows
            assert rows == reference, f"Complete conformance rows differ: {backend}/{corpus}"
            bulk = re.search(r"bulk JIT: kernels=(\d+) compiler_invocations=(\d+) status=ok", process.stderr)
            if compiler:
                assert bulk is not None
                assert int(bulk[2]) == 1
                assert int(bulk[1]) == sum(row["backend"] == "jit" for row in rows)
            result = {
                "corpus": corpus,
                "backend": backend,
                "cold_seconds": elapsed,
                "cases": len(rows),
                "routes": dict(Counter(row["backend"] for row in rows)),
                "all_rows_identical_to_serial_tcc": True,
                "external_compiler_invocations": int(bulk[2]) if bulk else 0,
                "stdout_sha256": hashlib.sha256(process.stdout.encode()).hexdigest(),
            }
            results.append(result)
            print(result, flush=True)
    report = {
        "method": "Fresh cache; all corpus rows compared exactly against serial TCC",
        "results": results,
        "compiler_versions": {
            backend: subprocess.check_output([compiler, "--version"], text=True)
            for backend, compiler in compilers.items()
            if compiler
        },
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
