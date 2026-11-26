"""Compare the opt-in C-Blosc2 resident minimum with the graph adapter.

Run with the shared-core build and a local C-Blosc2 exposing the resident pilot.
Preparation is uncached; warmed execution and ordinary compute are measured
separately. Timings are microsecond medians from interleaved runs on one thread.
This deliberately covers only contiguous int64 min(input), not mapped graphs.
"""

import argparse
import statistics
import time

import numpy as np

import blosc2
from blosc2.graph_plan import NativeGraph


def medians(first, second, repeat):
    first()
    second()
    times = [[], []]
    for run in range(repeat):
        for index in (run % 2, 1 - run % 2):
            function = (first, second)[index]
            start = time.perf_counter_ns()
            result = function()
            times[index].append((time.perf_counter_ns() - start) / 1000)
            del result
    return [statistics.median(samples) for samples in times]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[4097, 1000003])
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=51)
    args = parser.parse_args()
    if min(*args.items, args.tile, args.repeat) <= 0:
        parser.error("items, tile and repeat must be positive")
    if not getattr(blosc2.blosc2_ext, "_b2nd_min_i64_available", lambda: False)():
        parser.error("requires the C-Blosc2 resident minimum pilot")
    previous = blosc2.set_nthreads(1)
    try:
        print(f"{'items':>10} {'phase':<12} {'pilot us':>12} {'adapter us':>12}")
        for count in args.items:
            values = np.random.default_rng(42).integers(-(2**62), 2**62, size=count)
            inputs = {"x": values}
            metadata = {"x": (values.dtype, values.shape)}

            def prepare_pilot(count=count):
                return blosc2.blosc2_ext._ResidentMinI64Plan(count, args.tile)

            def prepare_adapter(metadata=metadata):
                return NativeGraph.from_expression("min(x)", {"x": "int64"}, jit=False).specialize(
                    metadata, args.tile
                )

            pilot = prepare_pilot()
            adapter = prepare_adapter()
            actual, report = pilot.execute(values)
            expected, reference = adapter.execute(inputs)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            assert actual == values.min()
            assert report["fp_flags"] == reference["fp_flags"] == 0
            assert report["zero_copy_tiles"] == reference["zero_copy_tiles"]
            assert report["temporary_bytes"] == reference["temporary_bytes"] == 0
            with blosc2.expression_evaluation("safe"):
                expr = blosc2.lazyexpr("min(x)", inputs)

            def compute_pilot(expr=expr):
                return expr.compute(jit=False, _native_tile_items=args.tile)

            def compute_adapter(expr=expr):
                return expr.compute(jit=False, _native_tile_items=args.tile, _require_native=True)

            compute_pilot()
            assert expr._native_execution_report["route"] == "c-blosc2-resident-min"
            for phase, first, second in [
                ("prepare", prepare_pilot, prepare_adapter),
                (
                    "execute",
                    lambda p=pilot, v=values: p.execute(v),
                    lambda a=adapter, i=inputs: a.execute(i),
                ),
                ("compute", compute_pilot, compute_adapter),
            ]:
                a, b = medians(first, second, args.repeat)
                print(f"{count:10d} {phase:<12} {a:12.2f} {b:12.2f}")
    finally:
        blosc2.set_nthreads(previous)


if __name__ == "__main__":
    main()
