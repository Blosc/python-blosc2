"""Compare direct prefilter reductions against the streaming reader route.

Run with conda Python, e.g.:
    python bench/ndarray/prefilter_reductions.py --size 32e6 --repeats 5

The reader route is produced by temporarily disabling prefilter input
eligibility, which is what the adapter consults before claiming a reduction.
Both routes must agree bit for bit, so a mismatch aborts the run.
"""

import argparse
import statistics
import sys
import time

import numpy as np

import blosc2
import blosc2.native_graph as native_graph


def benchmark(array, ops, repeats):
    ext = blosc2.blosc2_ext
    eligible = ext.menudet_prefilter_inputs_eligible
    for op in ops:
        samples = {mode: [] for mode in ("prefilter", "reader")}
        reports = {}
        expected = None
        for iteration in range(repeats + 1):
            # Rotate order to reduce systematic cache/thermal ordering bias.
            modes = list(samples)
            shift = iteration % len(modes)
            for mode in modes[shift:] + modes[:shift]:
                try:
                    ext.menudet_prefilter_inputs_eligible = (
                        (lambda *args, **kwargs: False) if mode == "reader" else eligible
                    )
                    expr = blosc2.lazyexpr("a", {"a": array})
                    start = time.perf_counter()
                    result = getattr(expr, op)()
                    elapsed = time.perf_counter() - start
                finally:
                    ext.menudet_prefilter_inputs_eligible = eligible
                if expected is None:
                    expected = result.tobytes()
                elif result.tobytes() != expected:
                    raise SystemExit(f"{op}: {mode} route changed result bits")
                if iteration:
                    samples[mode].append(elapsed)
                reports[mode] = expr._native_execution_report
        timings = {mode: statistics.median(v) for mode, v in samples.items()}
        baseline = timings["reader"]
        print(
            f"{op:4} {array.shape[0]:>12,} items  ratio={array.cratio:5.1f}  "
            f"reader={baseline * 1000:9.3f} ms  prefilter={timings['prefilter'] * 1000:9.3f} ms  "
            f"speedup={baseline / timings['prefilter']:5.2f}x"
        )
        for mode, report in reports.items():
            print(
                f"     {mode:9} route={report['route']:32} "
                f"extra={report['input_materialization_bytes']:>12,d} B"
            )
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, default=16e6, help="Array items")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--codecs", nargs="+", default=["BLOSCLZ", "ZSTD"])
    parser.add_argument("--ops", nargs="+", default=["min", "max", "any", "all"])
    args = parser.parse_args()
    if args.repeats < 1 or args.size <= 0:
        parser.error("--repeats and --size must be positive")
    # Admit smaller arrays than the production crossover for quick local runs.
    native_graph.MINIMUM_COMPRESSED_GRAPH_ITEMS = 1
    blosc2.set_nthreads(args.threads)
    print(f"threads={blosc2.nthreads}")
    for name in args.codecs:
        codec = blosc2.Codec[name]
        print(f"\n{name}: random normal", flush=True)
        values = np.random.default_rng(42).standard_normal(int(args.size))
        array = blosc2.asarray(values, cparams={"codec": codec})
        del values
        benchmark(array, args.ops, args.repeats)
        del array


if __name__ == "__main__":
    main()
