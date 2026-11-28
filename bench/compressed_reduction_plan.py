"""Track ordered compressed SUM decode amplification, not just sink throughput.

Resident and compressed execution share a typed plan and strict logical order.
Preparation and compression are outside timing. Extra decode budget admits up
to eight LRU chunk slots per operand/worker: report actual allocation and traffic.
This local benchmark is not the ARM64/x86-64 production performance gate.
"""

import argparse
import math
import statistics
import sys
import sysconfig
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, help="Isolated installed wheel (use Python -S)")
    parser.add_argument("--shape", type=int, nargs=2, default=(1001, 1003))
    parser.add_argument("--chunks", type=int, nargs=2, default=(128, 128))
    parser.add_argument("--blocks", type=int, nargs=2, default=(32, 32))
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--cache-budgets", type=int, nargs="+", default=(1, 8))
    args = parser.parse_args()
    if args.package:
        args.package = args.package.resolve()
        sys.path[:0] = [str(args.package), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    import numpy as np

    import blosc2
    from blosc2.graph_plan import NativeGraph

    if args.package:
        assert Path(blosc2.__file__).is_relative_to(args.package)
        assert Path(blosc2.blosc2_ext.__file__).is_relative_to(args.package)
    if min(*args.shape, *args.chunks, *args.blocks, args.tile, args.repeat, *args.cache_budgets) <= 0:
        parser.error("shapes, tile, repeat and cache-budget multiples must be positive")
    if not getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)():
        parser.error("requires the C-Blosc2 typed executor")
    # Dyadic values make the exact serial reference independent of NumPy's sum tree.
    values = (np.arange(math.prod(args.shape), dtype="float64") % 113 / 8).reshape(args.shape)
    array = blosc2.asarray(values, chunks=tuple(args.chunks), blocks=tuple(args.blocks), contiguous=False)
    graph = NativeGraph.from_expression("sum(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    schedule = graph.specialize({"x": (values.dtype, values.shape)}, args.tile)
    expected, reference = schedule.execute({"x": values})
    physical_chunks = math.prod(
        (size + chunk - 1) // chunk for size, chunk in zip(array.shape, array.chunks, strict=True)
    )
    cache_bytes = math.prod(array.ext_chunks) * values.dtype.itemsize
    print(f"shape={tuple(args.shape)} chunks={array.chunks} blocks={array.blocks} tile={args.tile}")
    print(
        f"{'budget B':>12} {'resident ms':>12} {'storage ms':>12} {'cache B':>12} "
        f"{'decodes':>10} {'chunks':>8} {'amp':>8} {'decoded B':>14}"
    )
    for multiple in args.cache_budgets:
        budget = multiple * cache_bytes

        def execute_storage(decode_budget=budget):
            actual, report = schedule.execute_arrays({"x": array}, workers=1, max_decode_bytes=decode_budget)
            assert actual.dtype == expected.dtype
            assert actual.shape == expected.shape
            assert actual.tobytes() == expected.tobytes()
            assert report["fp_flags"] == reference["fp_flags"]
            return report

        report = execute_storage()  # Warm caches/compiler before timing.
        resident_times, storage_times = [], []
        for repeat in range(args.repeat):
            # Alternate order to reduce systematic cache/frequency bias.
            routes = ("resident", "storage") if repeat % 2 == 0 else ("storage", "resident")
            for route in routes:
                start = time.perf_counter()
                if route == "resident":
                    schedule.execute({"x": values})
                else:
                    report = execute_storage()
                elapsed = (time.perf_counter() - start) * 1000
                (resident_times if route == "resident" else storage_times).append(elapsed)
        print(
            f"{budget:12d} {statistics.median(resident_times):12.3f} "
            f"{statistics.median(storage_times):12.3f} {report['decode_bytes']:12d} "
            f"{report['decodes']:10d} {physical_chunks:8d} "
            f"{report['decodes'] / physical_chunks:8.2f} {report['decoded_bytes']:14d}"
        )


if __name__ == "__main__":
    main()
