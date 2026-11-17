"""Compare automatic identity-map bypass with the mapped prefilter baseline.

Run with conda Python, e.g.:
    python bench/ndarray/sum_identity.py --size 1e9 --repeats 5
"""

import argparse
import statistics
import time

import numpy as np

import blosc2
from blosc2.graph_plan import NativeGraph


def benchmark(array, repeats):
    plan = NativeGraph.from_expression("sum(a)", {"a": array.dtype}, jit=False)
    kernel = plan._handle.reduction_artifact()
    result = np.empty((), dtype=array.dtype)
    timings = {mode: [] for mode in ("mapped", "identity")}
    expected = None
    reports = {}
    for iteration in range(repeats + 1):
        # Rotate order to reduce systematic cache/thermal ordering bias.
        modes = list(timings)
        shift = iteration % len(modes)
        for mode in modes[shift:] + modes[:shift]:
            start = time.perf_counter()
            value, report = kernel.evaluate_prefilter(
                {"a": array},
                _reduction_op="sum",
                _reduction_out=result,
                _pairwise_sum=True,
                _skip_identity=mode == "identity",
            )
            elapsed = time.perf_counter() - start
            if expected is None:
                expected = value.tobytes()
            assert value.tobytes() == expected, f"{mode} changed sum bits"
            if iteration:
                timings[mode].append(elapsed)
            reports[mode] = report
    baseline = statistics.median(timings["mapped"])
    print(f"shape={array.shape} chunks={array.chunks} blocks={array.blocks} ratio={array.cratio:.1f}")
    print(f"{'mode':12} {'median ms':>12} {'speedup':>9} {'extra input bytes':>17}")
    for mode, samples in timings.items():
        median = statistics.median(samples)
        print(
            f"{mode:12} {median * 1000:12.3f} {baseline / median:8.2f}x "
            f"{reports[mode]['input_buffer_bytes']:17,d}"
        )
    print(f"sum={result[()]!r}; all paths bit-identical", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=float, default=1e9, help="Ramp size")
    parser.add_argument("--random-size", type=int, default=10_000_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--codecs", nargs="+", default=["BLOSCLZ", "ZSTD"])
    args = parser.parse_args()
    if args.repeats < 1 or args.size <= 0 or args.random_size < 0:
        parser.error("--repeats and --size must be positive; --random-size must be nonnegative")
    blosc2.set_nthreads(args.threads)
    print(f"threads={blosc2.nthreads}")
    for name in args.codecs:
        codec = blosc2.Codec[name]
        print(f"\n{name}: arange({args.size:g})", flush=True)
        array = blosc2.arange(args.size, cparams={"codec": codec})
        benchmark(array, args.repeats)
        del array
        if args.random_size:
            print(f"\n{name}: random normal ({args.random_size:,})", flush=True)
            values = np.random.default_rng(42).standard_normal(args.random_size)
            array = blosc2.asarray(values, cparams={"codec": codec})
            del values
            benchmark(array, args.repeats)
            del array
            print(f"\n{name}: quantized random ({args.random_size:,})", flush=True)
            values = np.random.default_rng(42).integers(0, 1000, size=args.random_size).astype("float64")
            array = blosc2.asarray(values, cparams={"codec": codec})
            del values
            benchmark(array, args.repeats)
            del array


if __name__ == "__main__":
    main()
