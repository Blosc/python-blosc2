"""Materialized stage costs; run repeatedly in fresh processes for cold samples.

No compressed storage IO. RSS is process peak, not compiler-specific allocations.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np

import blosc2


def timed(function):
    start = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - start) / 1000


def revision(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def dirty(path):
    return bool(
        subprocess.check_output(["git", "-C", path, "diff", "HEAD", "--name-only"], text=True).strip()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dtype", choices=["float32", "float64", "int64"], default="float32")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    if args.rows < 1 or args.columns < 1 or args.repeats < 1:
        parser.error("dimensions and repeats must be positive")
    expression = "x - sum(x, axis=0)"
    plan, preparation = timed(
        lambda: blosc2.NativeGraph.from_expression(expression, {"x": args.dtype}, jit=args.jit)
    )
    schedule, specialization = timed(lambda: plan.specialize({"x": (args.dtype, (args.rows, args.columns))}))
    values = {"x": np.ones((args.rows, args.columns), dtype=args.dtype)}
    (_, report), first = timed(lambda: schedule.execute(values))
    warm = [timed(lambda: schedule.execute(values))[1] for _ in range(args.repeats)]
    values["x"].fill(2)
    (result, _), rebind = timed(lambda: schedule.execute(values))
    np.testing.assert_array_equal(result, np.full(values["x"].shape, 2 - 2 * args.rows, dtype=result.dtype))
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss *= 1 if platform.system() == "Darwin" else 1024
    except ImportError:
        rss = None
    print(
        json.dumps(
            {
                "python_revision": revision("."),
                "native_revision": revision("../miniexpr"),
                "python_tracked_changes": dirty("."),
                "native_tracked_changes": dirty("../miniexpr"),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "expression": expression,
                "dtype": args.dtype,
                "shape": values["x"].shape,
                "jit_requested": args.jit,
                "preparation_us": preparation,
                "specialization_us": specialization,
                "first_us": first,
                "warm_median_us": statistics.median(warm),
                "rebind_us": rebind,
                "plan": {
                    key: value
                    for key, value in plan.info().items()
                    if key not in {"inputs", "inferred_dtype"}
                },
                "schedule": {key: value for key, value in schedule.info().items() if key != "dtype"},
                "stages": [
                    {**schedule.stage_info(i), "dtype": str(schedule.stage_info(i)["dtype"])}
                    for i in range(plan.info()["stages"])
                ],
                "report": report,
                "output_bytes": result.nbytes,
                "process_peak_rss_bytes": rss,
                "note": "Warm imports, first plan cold; metadata excludes compiler/artifact allocations; RSS includes all imports and NumPy reference checking.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
