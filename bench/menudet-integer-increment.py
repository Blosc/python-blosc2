"""Warm NumPy/Menudet/legacy DSL timings for x + 1 on resident arrays.

Example: python bench/menudet-integer-increment.py --dtype int64 --compiler cc
Compilation and correctness checks are outside timings; every read allocates a
fresh NumPy result. Blosc2's thread count does not change NumPy's threading.
"""

import argparse
import gc
import os
from statistics import median
from time import perf_counter

import numpy as np

import blosc2


@blosc2.menudet
def portable(x):
    return x + 1


@blosc2.dsl_kernel
def legacy(x):
    return x + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=100_000_000)
    parser.add_argument("--dtype", choices=("int8", "int64", "float64"), default="int8")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--compiler", choices=("tcc", "cc"))
    args = parser.parse_args()
    if args.items < 1 or args.repeat < 1 or min(args.threads) < 1:
        parser.error("items, repeat and thread counts must be positive")
    if args.compiler:
        os.environ["ME_DSL_JIT_COMPILER"] = args.compiler
    print(f"Package: {blosc2.__file__}\nExtension: {blosc2.blosc2_ext.__file__}")
    x = np.arange(args.items, dtype=args.dtype)
    expected = x + 1
    print(f"dtype={x.dtype}, items={x.size}, Menudet JIT={portable.specialize(x).has_jit}")
    previous = blosc2.nthreads
    try:
        print(f"{'Threads':>7} {'NumPy ms':>12} {'Menudet ms':>12} {'Legacy ms':>12}")
        for threads in args.threads:
            blosc2.set_nthreads(threads)
            modern = portable(x)
            old = blosc2.lazyudf(legacy, (x,))
            engines = [lambda: x + 1, lambda expr=modern: expr[:], lambda expr=old: expr[:]]
            for call in engines:
                result = call()
                assert result.dtype == x.dtype
                np.testing.assert_array_equal(result, expected)
                del result
            samples = [[] for _ in engines]
            for repeat in range(args.repeat):
                for offset in range(len(engines)):
                    index = (repeat + offset) % len(engines)
                    gc.collect()
                    start = perf_counter()
                    result = engines[index]()
                    samples[index].append(perf_counter() - start)
                    del result
            timings = " ".join(f"{median(values) * 1000:12.2f}" for values in samples)
            print(f"{threads:7d} {timings}", flush=True)
    finally:
        blosc2.set_nthreads(previous)


if __name__ == "__main__":
    main()
