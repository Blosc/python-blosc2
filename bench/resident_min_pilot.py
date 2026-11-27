"""Compare the opt-in C-Blosc2 resident minimum with the graph adapter.

Run with the shared-core build and a local C-Blosc2 exposing the resident pilot.
Preparation is uncached; adapter specialization, warmed execution and ordinary
compute are measured separately. Pilot preparation includes geometry while
adapter preparation does not. Timings are microsecond medians from interleaved
runs on one thread. Direct handles explicitly select pilot and adapter; automatic
compute dispatch is not an enduring old/new implementation selector.
This covers contiguous int64 min(input) and two simple mapped reductions.
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
    parser.add_argument("--expression", choices=["min(x)", "min(x + 1)", "min(x * 2)"], default="min(x)")
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
                if args.expression == "min(x)":
                    return blosc2.blosc2_ext._ResidentMinI64Plan(count, args.tile)
                source = b"def mapped(x):\n    return x + 1\n"
                if args.expression == "min(x * 2)":
                    source = b"def mapped(x):\n    return x * 2\n"
                return blosc2.blosc2_ext._ResidentMapMinI64Plan(source, count, args.tile)

            def prepare_adapter(metadata=metadata):
                return NativeGraph.from_expression(
                    args.expression, {"x": "int64"}, jit=False, _executor="adapter"
                )

            pilot = prepare_pilot()
            adapter_plan = prepare_adapter()
            adapter = adapter_plan.specialize(metadata, args.tile)
            actual, report = pilot.execute(values)
            expected, reference = adapter.execute(inputs)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            numpy_values = values
            if args.expression == "min(x + 1)":
                numpy_values = values + 1
            elif args.expression == "min(x * 2)":
                numpy_values = values * 2
            assert actual == numpy_values.min()
            assert report["fp_flags"] == reference["fp_flags"] == 0
            assert report["zero_copy_tiles"] == reference["zero_copy_tiles"]
            assert report["temporary_bytes"] == reference["temporary_bytes"]
            with blosc2.expression_evaluation("safe"):
                expr = blosc2.lazyexpr(args.expression, inputs)

            def compute_pilot(expr=expr):
                return expr.compute(jit=False, _native_tile_items=args.tile)

            def compute_adapter(expr=expr):
                return expr.compute(jit=False, _native_tile_items=args.tile, _require_native=True)

            compute_pilot()
            assert expr._native_execution_report["route"].startswith(
                ("c-blosc2-resident-", "c-blosc2-typed-resident")
            )
            # Unlike preparation scopes are deliberately labelled. There is no
            # independent specialization operation in the specialized pilot.
            a, b = medians(prepare_pilot, prepare_adapter, args.repeat)
            print(f"{count:10d} {'prep+geom/prep':<12} {a:12.2f} {b:12.2f}")
            _, b = medians(
                lambda: None, lambda p=adapter_plan, m=metadata: p.specialize(m, args.tile), args.repeat
            )
            print(f"{count:10d} {'specialize':<12} {'n/a':>12} {b:12.2f}")
            for phase, first, second in [
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
