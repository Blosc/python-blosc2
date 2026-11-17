"""Compare direct prefilter reductions against the streaming reader route.

Run with conda Python, e.g.:
    python bench/ndarray/prefilter_reductions.py --size 32e6 --repeats 5

The reader route is produced by temporarily disabling prefilter input
eligibility, which is what the adapter consults before claiming a reduction.
Both routes must agree bit for bit except for mean/variance/std, whose different
summation/centered-moment algorithms use dtype-appropriate tolerances.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

import blosc2
import blosc2.native_graph as native_graph


def benchmark(array, ops, repeats, expression):
    ext = blosc2.blosc2_ext
    eligible = ext.menudet_prefilter_inputs_eligible
    moments = getattr(ext, "menudet_moments_available", lambda: False)
    for op in ops:
        statistical = op in {"var", "std"}
        samples = {mode: [] for mode in ("prefilter", "reader")}
        reports = {}
        expected = None
        for iteration in range(repeats + 1):
            # Rotate order to reduce systematic cache/thermal ordering bias.
            modes = list(samples)
            shift = iteration % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                try:
                    if statistical:
                        # Compare one-pass moments with the established two-pass
                        # implementation, keeping its accelerated sums enabled.
                        ext.menudet_moments_available = (lambda: False) if mode == "reader" else moments
                    else:
                        ext.menudet_prefilter_inputs_eligible = (
                            (lambda *args, **kwargs: False) if mode == "reader" else eligible
                        )
                    expr = blosc2.lazyexpr(expression, {"a": array})
                    start = time.perf_counter()
                    result = getattr(expr, op)()
                    elapsed = time.perf_counter() - start
                finally:
                    ext.menudet_prefilter_inputs_eligible = eligible
                    ext.menudet_moments_available = moments
                if expected is None:
                    expected = result.copy()
                elif statistical or op == "mean":
                    np.testing.assert_allclose(
                        result,
                        expected,
                        rtol=2e-5 if array.dtype == np.float32 else 1e-11,
                        # Means close to zero need an absolute tolerance; the
                        # existing reader and fixed-leaf sums have different trees.
                        atol=np.finfo(array.dtype).eps if op == "mean" else 0,
                    )
                elif result.tobytes() != expected.tobytes():
                    raise SystemExit(f"{op}: {mode} route changed result bits")
                if iteration:
                    samples[mode].append(elapsed)
                reports[mode] = getattr(expr, "_native_execution_report", {"route": "chunked-fallback"})
                if (
                    mode == "reader"
                    and reports[mode]["route"].startswith("parallel-prefilter")
                    and not statistical
                ):
                    raise SystemExit(f"{op}: baseline unexpectedly used a prefilter")
        timings = {mode: statistics.median(v) for mode, v in samples.items()}
        baseline = timings["reader"]
        record = {
            "op": op,
            "expression": expression,
            "dtype": array.dtype.name,
            "size": array.size,
            "chunks": array.chunks,
            "blocks": array.blocks,
            "threads": blosc2.nthreads,
            "ratio": array.cratio,
            "seconds": timings,
            "speedup": baseline / timings["prefilter"],
            "routes": {name: report["route"] for name, report in reports.items()},
            "prefilter_selected": reports["prefilter"]["route"].startswith("parallel-prefilter"),
            "worker_threads": {name: report.get("threads") for name, report in reports.items()},
        }
        print(
            f"{op:6} {array.dtype.name:7} threads={blosc2.nthreads:2} expr={expression:9} "
            f"{array.shape[0]:>12,} items  ratio={array.cratio:5.1f}  "
            f"reader={baseline * 1000:9.3f} ms  prefilter={timings['prefilter'] * 1000:9.3f} ms  "
            f"speedup={baseline / timings['prefilter']:5.2f}x"
        )
        for mode, report in reports.items():
            print(
                f"     {mode:9} route={report['route']:32} "
                f"extra={report.get('input_materialization_bytes', 0):>12,d} B"
            )
        sys.stdout.flush()
        yield record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, default=16e6, help="Array items")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 4, blosc2.nthreads])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float32", "float64"], default=["float32", "float64"]
    )
    parser.add_argument("--expressions", nargs="+", default=["a", "a + a"])
    parser.add_argument("--codecs", nargs="+", default=["BLOSCLZ", "ZSTD"])
    parser.add_argument(
        "--ops",
        nargs="+",
        choices=["min", "max", "any", "all", "argmin", "argmax", "mean", "var", "std"],
        default=["min", "max", "any", "all", "argmin", "argmax"],
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save per-run JSON outside the checkout (temporary storage or CI artifacts)",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.size < 262144 or min(args.threads) < 1:
        parser.error("--repeats/--threads must be positive; --size must be at least 262144")
    if not blosc2.blosc2_ext.menudet_shared_reduction_rules_available():
        parser.error(
            "Requires miniexpr shared-extrema primitives (use the qualified native source override)"
        )
    # Admit smaller arrays than the production crossover for quick local runs.
    native_graph.MINIMUM_COMPRESSED_GRAPH_ITEMS = 1
    records = []
    previous = blosc2.nthreads
    try:
        for name in args.codecs:
            codec = blosc2.Codec[name]
            for dtype in args.dtypes:
                print(f"\n{name}: random normal ({dtype})", flush=True)
                values = np.random.default_rng(42).standard_normal(int(args.size)).astype(dtype)
                array = blosc2.asarray(values, chunks=(262144,), blocks=(8192,), cparams={"codec": codec})
                del values
                for threads in dict.fromkeys(args.threads):
                    blosc2.set_nthreads(threads)
                    # Legacy single-worker reductions use the array's decoder
                    # policy rather than the global setting. Keep both matched.
                    array.schunk.dparams = blosc2.DParams(nthreads=threads)
                    for expression in args.expressions:
                        for record in benchmark(array, args.ops, args.repeats, expression):
                            records.append({"codec": name, **record})
                del array
    finally:
        blosc2.set_nthreads(previous)
    if args.output:
        args.output.write_text(
            json.dumps(
                {
                    "environment": {
                        "platform": platform.platform(),
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                        "blosc2": blosc2.__version__,
                    },
                    "repeats": args.repeats,
                    "results": records,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
