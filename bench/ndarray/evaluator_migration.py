"""Compare the temporary legacy/typed routes on identical compressed inputs.

Run with the blosc2 conda environment. This is a timing probe, not a peak-memory
or WASM qualification benchmark. Modes are interleaved after warmup.
"""

import argparse
import json
import os
import statistics
import time

import numpy as np

import blosc2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="+", default=[32, 256, 256])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 1 or any(size < 1 for size in args.shape):
        parser.error("shape, rounds and threads must be positive")
    blosc2.set_nthreads(args.threads)
    rng = np.random.default_rng(1)
    values = rng.random(tuple(args.shape), dtype=np.float32)
    a, b = blosc2.asarray(values), blosc2.asarray(values + 1)
    methods = {
        "getitem": lambda expr: expr[:],
        "compute": lambda expr: expr.compute(),
        "sum": lambda expr: expr.sum(),
        "axis0": lambda expr: expr.sum(axis=0),
    }
    elapsed = {mode: {method: [] for method in methods} for mode in ("legacy", "typed")}
    previous = os.environ.get("BLOSC2_EXPRESSION_EVALUATOR")
    try:
        expressions = {}
        for mode in elapsed:
            os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = mode
            expressions[mode] = a + b * 2
            for operation in methods.values():
                operation(expressions[mode])
        for round_ in range(args.rounds):
            for mode in ("legacy", "typed") if round_ % 2 == 0 else ("typed", "legacy"):
                os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = mode
                for method, operation in methods.items():
                    start = time.perf_counter()
                    operation(expressions[mode])
                    elapsed[mode][method].append(1000 * (time.perf_counter() - start))
    finally:
        if previous is None:
            os.environ.pop("BLOSC2_EXPRESSION_EVALUATOR", None)
        else:
            os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = previous
    print(
        json.dumps(
            {
                "shape": a.shape,
                "chunks": a.chunks,
                "blocks": a.blocks,
                "threads": blosc2.nthreads,
                "rounds": args.rounds,
                "numpy": np.__version__,
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "median_ms": {
                    mode: {method: statistics.median(samples) for method, samples in timings.items()}
                    for mode, timings in elapsed.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
