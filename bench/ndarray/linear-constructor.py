#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Compare public linspace and authored Menudet using the same storage prefilter.

NumPy is a decoded-output reference, not a compression-throughput competitor.
The authored route uses a benchmark-only adapter to the native materializer used
by blosc2.linspace; general 2-D Menudet .compute() does not select that path yet.
Preparation and first execution are separate from warmed execution measurements.
"""

import argparse
from statistics import median
from time import perf_counter

import numpy as np

import blosc2

dtype = np.float64
start, stop = 1, 2
cparams = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1)


def make_ramp(start, stop, nitems):
    first = np.float64(start)
    last = np.float64(stop)
    step = np.float64((last - first) / (nitems - 1) if nitems > 1 else 0)

    @blosc2.menudet(ndim=2)
    def kernel_ramp() -> np.float64:
        # _flat_idx is global, including when materializing individual chunks.
        # Match the built-in float64 constructor's arithmetic exactly.
        return first + _flat_idx * step  # noqa: F821

    return kernel_ramp


def materialize_ramp(kernel, shape, storage):
    """Benchmark-only adapter matching the built-in constructor's materializer."""
    output = blosc2.empty(shape, dtype=kernel.output_dtype, **storage)
    if not blosc2.blosc2_ext.menudet_prefilter_inputs_eligible({}, output=output):
        raise NotImplementedError("This benchmark requires native N-D generator prefilters")
    result, report = kernel._handle.evaluate_prefilter({}, out=output)
    return result, report


def check_result(result, expected):
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    if isinstance(result, blosc2.NDArray):
        # Bound validation scratch rather than decoding another complete array.
        for row in range(0, result.shape[0], result.chunks[0]):
            item = slice(row, row + result.chunks[0])
            np.testing.assert_array_equal(result[item], expected[item])
    else:
        np.testing.assert_array_equal(result, expected)


def measure(compute, expected, warmups, samples):
    t0 = perf_counter()
    result = compute()
    first = perf_counter() - t0
    check_result(result, expected)
    del result
    for _ in range(warmups):
        result = compute()
        del result
    times = []
    for _ in range(samples):
        t0 = perf_counter()
        result = compute()
        times.append(perf_counter() - t0)
        # Retain output until its timer stops; destruction is outside timing.
        del result
    result = compute()
    check_result(result, expected)
    del result
    return first, median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=12_000, help="Side length of the square array")
    parser.add_argument("--warmups", type=int, default=3, help="Untimed warmup calls after first execution")
    parser.add_argument("--samples", type=int, default=5, help="Timed calls; report their median")
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    args = parser.parse_args()
    if min(args.size, args.samples, args.threads) < 1 or args.warmups < 0:
        parser.error("size, samples and threads must be positive; warmups must be nonnegative")
    blosc2.set_nthreads(args.threads)
    shape = (args.size, args.size)
    nitems = int(np.prod(shape))
    compression = blosc2.CParams(codec=cparams.codec, clevel=cparams.clevel, nthreads=args.threads)
    chunks, blocks = blosc2.compute_chunks_blocks(shape, dtype=dtype, cparams=compression)
    storage = {"chunks": chunks, "blocks": blocks, "cparams": compression}
    expected = np.linspace(start, stop, nitems, dtype=dtype).reshape(shape)
    print(f"Package: {blosc2.__file__}")
    print(f"Extension: {blosc2.blosc2_ext.__file__}")
    print(f"{shape}, {nitems:,} float64 elements; {args.threads} threads")
    print(f"chunks={chunks}, blocks={blocks}; {compression.codec.name}, clevel={compression.clevel}")
    print("Warm times include fresh output allocation and compression, but not output destruction.")
    print("NumPy returns decoded output; Blosc2 rows return identically configured compressed NDArrays.")
    print("Public linspace includes preparation in first/warm calls; authored preparation is separate.")
    print("First calls are not cache-isolated cold-compilation measurements.")
    print(f"{'Method':<24} {'Prepare ms':>11} {'First ms':>11} {'Warm ms':>11}")
    print("-" * 62)

    def show(name, compute, preparation=None):
        first, warm = measure(compute, expected, args.warmups, args.samples)
        prep = "n/a" if preparation is None else f"{preparation * 1000:.3f}"
        print(f"{name:<24} {prep:>11} {first * 1000:>11.3f} {warm * 1000:>11.3f}", flush=True)

    show("NumPy (decoded)", lambda: np.linspace(start, stop, nitems, dtype=dtype).reshape(shape))
    for backend in ("tcc", "cc"):
        with blosc2.jit_options(jit=True, jit_backend=backend):
            show(
                f"blosc2.linspace/{backend}",
                lambda: blosc2.linspace(start, stop, nitems, dtype=dtype, shape=shape, **storage),
            )
            t0 = perf_counter()
            kernel = make_ramp(start, stop, nitems).specialize()
            preparation = perf_counter() - t0
            if not kernel.has_jit:
                raise RuntimeError(f"Authored ramp did not activate the requested {backend} JIT")

            def compute(kernel=kernel):
                result, report = materialize_ramp(kernel, shape, storage)
                compute.report = report
                return result

            show(f"Menudet prefilter/{backend}", compute, preparation)
            print(f"  Native report: {compute.report}", flush=True)


if __name__ == "__main__":
    main()
