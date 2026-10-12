"""Interleave warm LazyExpr and Menudet reads over identical resident inputs."""

import argparse
import json
import os
import platform
import random
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--inputs", choices=("blosc2", "numpy"), default="blosc2")
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    blosc2.set_nthreads(12)
    x, y = np.linspace(0, 1, 8_000_000), np.linspace(1, 2, 8_000_000)
    expected = x + (y - 1)
    inputs = (
        {"x": x, "y": y}
        if args.inputs == "numpy"
        else {
            name: blosc2.asarray(value, chunks=(1_000_000,), blocks=(15625,))
            for name, value in {"x": x, "y": y}.items()
        }
    )
    expressions = {"lazyexpr": blosc2.lazyexpr("x + (y - 1)", inputs), "menudet": adjusted(**inputs)}
    assert expressions["menudet"].kernel.has_jit
    for expression in expressions.values():
        for _ in range(5):
            np.testing.assert_array_equal(expression[:], expected)
    times = {name: [] for name in expressions}
    generator = random.Random(42)
    for _ in range(args.samples):
        order = list(expressions)
        generator.shuffle(order)
        for name in order:
            start = perf_counter()
            result = expressions[name][:]
            times[name].append((perf_counter() - start) * 1000)
            del result
    timings = {
        name: {
            "median_ms": round(median(values), 4),
            "p25_ms": round(float(np.percentile(values, 25)), 4),
            "p75_ms": round(float(np.percentile(values, 75)), 4),
        }
        for name, values in times.items()
    }
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "extension": blosc2.blosc2_ext.__file__,
                "compiler": os.environ.get("ME_DSL_JIT_COMPILER", "default"),
                "cc": os.environ.get("CC", "default"),
                "threads": blosc2.nthreads,
                "samples_per_method": args.samples,
                "items": x.size,
                "dtype": str(x.dtype),
                "expression": "x + (y - 1)",
                "input_kind": args.inputs,
                "chunks": None if args.inputs == "numpy" else inputs["x"].chunks,
                "blocks": None if args.inputs == "numpy" else inputs["x"].blocks,
                "input_cparams": None if args.inputs == "numpy" else repr(inputs["x"].schunk.cparams),
                "timings": timings,
                "lazyexpr_over_menudet": round(median(times["lazyexpr"]) / median(times["menudet"]), 4),
                "menudet_route": expressions["menudet"]._source_execution_report,
                "lazyexpr_native_report": getattr(expressions["lazyexpr"], "_native_execution_report", None),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
