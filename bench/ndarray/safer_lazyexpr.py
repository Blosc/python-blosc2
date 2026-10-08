"""Compare safe graphs with the trusted legacy evaluator in isolated processes.

Use --baseline-root with an immutable source snapshot and the matching compiled
extension for a historical control. Otherwise full mode is only a legacy-path
proxy. No timing assertions run in CI. Outputs are checked before timing.
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def worker(args):
    if args.package_root:
        sys.meta_path = [
            finder for finder in sys.meta_path if type(finder).__module__ != "_editable_skbc_blosc2"
        ]
        sys.path.insert(0, args.package_root)
    import cProfile
    import resource

    import numpy as np

    import blosc2

    if args.package_root:
        assert Path(blosc2.__file__).resolve().is_relative_to(Path(args.package_root).resolve())

    blosc2.set_nthreads(1)
    n = args.size
    shape = (max(1, n // 128), 128) if args.family == "axis" else (n,)
    x = np.linspace(1, 4, math.prod(shape), dtype=args.dtype).reshape(shape)
    y = np.ones_like(x)
    expressions = {
        "arithmetic": ("x * 2 + y", x * 2 + y),
        "math": ("sqrt(x) + sin(x) + y", np.sqrt(x) + np.sin(x) + y),
        "reduction": ("sum(sqrt(x))", np.sqrt(x).sum()),
        "center": ("x - mean(x)", x - x.mean()),
        "axis": ("x.mean(axis=0)", x.mean(axis=0)),
        "index": ("x[::2] + y[::2]", x[::2] + y[::2]),
        "open": ("sqrt(x) + y", np.sqrt(x) + y),
    }
    text, expected = expressions[args.family]
    with tempfile.TemporaryDirectory(dir=args.tempdir) as directory:
        operands = {
            "x": blosc2.asarray(
                x, urlpath=str(Path(directory) / "x.b2nd") if args.family == "open" else None
            ),
            "y": blosc2.asarray(
                y, urlpath=str(Path(directory) / "y.b2nd") if args.family == "open" else None
            ),
        }
        timings = []
        construction = []
        first_execution = []
        open_times = []
        profiler = cProfile.Profile() if args.profile_output else None
        for trial in range(args.repeats + 1):
            start = time.perf_counter()
            expr = (
                blosc2.lazyexpr(text, operands)
                if args.package_root
                else blosc2.lazyexpr(text, operands, evaluation=args.mode)
            )
            constructed = time.perf_counter() - start
            if args.family == "open":
                path = str(Path(directory) / "expr.b2nd")
                expr.save(path)
                start = time.perf_counter()
                expr = blosc2.open(path, deserialize=args.mode)
                opened = time.perf_counter() - start
                if trial:
                    open_times.append(opened)
            start = time.perf_counter()
            result = expr[()]
            first = time.perf_counter() - start
            np.testing.assert_allclose(result, expected, rtol=1e-5 if args.dtype == "float32" else 1e-12)
            start = time.perf_counter()
            if profiler is not None:
                profiler.enable()
            result = expr[()]
            if profiler is not None:
                profiler.disable()
            elapsed = time.perf_counter() - start
            if trial:
                construction.append(constructed)
                first_execution.append(first)
                timings.append(elapsed)
        if profiler is not None:
            profiler.dump_stats(args.profile_output)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(
            json.dumps(
                {
                    "family": args.family,
                    "mode": args.mode,
                    "size": n,
                    "dtype": args.dtype,
                    "construction_seconds": statistics.median(construction),
                    "first_execution_seconds": statistics.median(first_execution),
                    "repeated_execution_seconds": statistics.median(timings),
                    "execution_min_seconds": min(timings),
                    "execution_max_seconds": max(timings),
                    "open_seconds": statistics.median(open_times) if open_times else None,
                    "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                    "version": blosc2.__version__,
                    "platform": platform.platform(),
                    "package": blosc2.__file__,
                    "control": "historical source snapshot" if args.package_root else "current source",
                }
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=["safe", "full"], default="safe")
    parser.add_argument("--family", default="math")
    parser.add_argument("--families", default="arithmetic,math,reduction,center,axis,index,open")
    parser.add_argument("--size", type=int, default=100000)
    parser.add_argument("--sizes", default="1000,100000,1000000")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Independent process pairs; alternate execution order each round",
    )
    parser.add_argument("--tempdir")
    parser.add_argument("--package-root")
    parser.add_argument("--baseline-root")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-output",
        type=Path,
        help="Worker only: profile repeated execution, excluding construction/open/checks",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.rounds < 1:
        parser.error("--repeats and --rounds must be positive")
    if args.profile_output and not args.worker:
        parser.error("--profile-output requires --worker")
    if args.worker:
        worker(args)
        return
    records = []
    env = dict(os.environ, NUMEXPR_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for round_number in range(args.rounds):
        order = ("full", "safe") if round_number % 2 == 0 else ("safe", "full")
        for size in map(int, args.sizes.split(",")):
            for family in args.families.split(","):
                pair = {}
                for mode in order:
                    command = [
                        sys.executable,
                        __file__,
                        "--worker",
                        "--mode",
                        mode,
                        "--family",
                        family,
                        "--size",
                        str(size),
                        "--dtype",
                        args.dtype,
                        "--repeats",
                        str(args.repeats),
                    ]
                    if args.tempdir:
                        command += ["--tempdir", args.tempdir]
                    if mode == "full" and args.baseline_root:
                        command += ["--package-root", args.baseline_root]
                    completed = subprocess.run(command, env=env, check=True, text=True, capture_output=True)
                    record = json.loads(completed.stdout)
                    record["round"] = round_number
                    record["execution_order"] = list(order)
                    pair[mode] = record
                records.extend((pair["full"], pair["safe"]))
    ratios = []
    for full, safe in zip(records[::2], records[1::2], strict=True):
        ratios.append(
            {
                "family": safe["family"],
                "size": safe["size"],
                "round": safe["round"],
                "execution_ratio": safe["repeated_execution_seconds"] / full["repeated_execution_seconds"],
                "construction_ratio": safe["construction_seconds"] / full["construction_seconds"],
            }
        )
    report = {
        "records": records,
        "ratios": ratios,
        "geomean_execution_ratio": math.exp(statistics.mean(math.log(x["execution_ratio"]) for x in ratios)),
    }
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(output + "\n")
    else:
        print(output)


if __name__ == "__main__":
    main()
