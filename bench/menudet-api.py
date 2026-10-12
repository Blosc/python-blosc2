"""Compare warm LazyExpr/Menudet value reads; retain explicit backend evidence."""

import argparse
import json
import os
import platform
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


@blosc2.menudet
def add_adjusted(x, y):
    return x + (y - 1)


def measure(operation, samples):
    first = perf_counter()
    values = operation()
    first_ms = (perf_counter() - first) * 1000
    times = []
    for _ in range(samples):
        start = perf_counter()
        values = operation()
        times.append((perf_counter() - start) * 1000)
    return values, first_ms, median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()
    if args.samples <= 0 or any(size <= 0 for size in args.items):
        parser.error("items and samples must be positive")
    rows = []
    for size in args.items:
        x, y = np.linspace(0, 1, size), np.linspace(1, 2, size)
        expected = x + (y - 1)
        for storage in ("numpy", "blosc2"):
            inputs = {"x": x, "y": y}
            if storage == "blosc2":
                inputs = {name: blosc2.asarray(value) for name, value in inputs.items()}
            # Include first-call binding/preparation in the first-read figure;
            # warm reads below retain the same bound lazy objects.
            for api in ("LazyExpr", "Menudet"):
                start = perf_counter()
                lazy = (
                    blosc2.lazyexpr("x + (y - 1)", inputs) if api == "LazyExpr" else add_adjusted(**inputs)
                )
                binding_ms = (perf_counter() - start) * 1000
                values, first_ms, warm_ms = measure(lambda lazy=lazy: lazy[:], args.samples)
                np.testing.assert_array_equal(values, expected)
                route = (
                    getattr(lazy, "_native_execution_report", {}).get("backend", "legacy")
                    if api == "LazyExpr"
                    else "portable-jit"
                    if lazy.kernel.has_jit
                    else "portable-interpreter"
                )
                rows.append(
                    {
                        "storage": storage,
                        "items": size,
                        "api": api,
                        "route": route,
                        "binding_plus_first_ms": round(binding_ms + first_ms, 3),
                        "warm_median_ms": round(warm_ms, 3),
                    }
                )
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "numpy": np.__version__,
                "blosc2": blosc2.__version__,
                "blosc2_threads": blosc2.nthreads,
                "jit_compiler": os.environ.get("ME_DSL_JIT_COMPILER", "runtime default"),
                "samples": args.samples,
                "dtype": "float64",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
