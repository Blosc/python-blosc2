"""Interleaved old export-twice/native JSON/native text preparation baseline.

Run in fresh subprocesses for cold samples. Execution timings exclude storage IO;
graph/native iterator memory must not be labeled whole-process or compressed IO.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np

import blosc2
from blosc2.dsl_kernel import DSLKernel


def old_prepare(dtype):
    author = DSLKernel.from_source("def k(x, y):\n    return x * 2 + y\n")
    signatures = {"x": dtype, "y": dtype}
    first = author.export(signatures, "float64", version="1.1", casting="unsafe")
    inferred = blosc2.PortableKernel.from_json(first).inferred_dtype
    return blosc2.PortableKernel.from_json(
        author.export(signatures, inferred, version="1.1", casting="unsafe")
    )


def timed(function):
    start = time.perf_counter_ns()
    result = function()
    return result, (time.perf_counter_ns() - start) / 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "float64", "int64"], default="float32")
    args = parser.parse_args()
    if args.items < 0 or args.repeats < 1:
        parser.error("items must be nonnegative and repeats positive")
    signature = {"x": args.dtype, "y": args.dtype}
    seed = blosc2.NativeGraph.from_expression("x * 2 + y", signature).to_json()
    timings = {"old_export_twice_us": [], "native_json_us": [], "native_text_us": []}
    routes = {
        "old_export_twice_us": lambda: old_prepare(args.dtype),
        "native_json_us": lambda: blosc2.NativeGraph.from_json(seed),
        "native_text_us": lambda: blosc2.NativeGraph.from_expression("x * 2 + y", signature),
    }
    for repeat in range(args.repeats):
        names = list(routes)
        names = names[repeat % 3 :] + names[: repeat % 3]
        for name in names:
            _, elapsed = timed(routes[name])
            timings[name].append(elapsed)
    plan = blosc2.NativeGraph.from_json(seed)
    metadata = {"x": (args.dtype, (args.items,)), "y": (args.dtype, ())}
    schedule, specialize = timed(lambda: plan.specialize(metadata))
    values = {"x": np.arange(args.items, dtype=args.dtype), "y": np.array(3, dtype=args.dtype)}
    (_, report), first = timed(lambda: schedule.execute(values))
    warm = [timed(lambda: schedule.execute(values))[1] for _ in range(args.repeats)]
    values["y"] = np.array(4, dtype=args.dtype)
    (result, _), rebind = timed(lambda: schedule.execute(values))
    np.testing.assert_array_equal(result, values["x"] * 2 + 4)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(
        json.dumps(
            {
                "python_revision": revision,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "items": args.items,
                "dtype": args.dtype,
                "repeats": args.repeats,
                "backend": "portable-jit" if plan.has_jit else "portable-interpreter",
                "preparation_median_us": {key: statistics.median(value) for key, value in timings.items()},
                "specialize_us": specialize,
                "first_us": first,
                "warm_median_us": statistics.median(warm),
                "rebind_us": rebind,
                "iterator": report,
                "note": "Process imports warm; no result cache; no compressed reads; compiler memory/RSS not measured.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
