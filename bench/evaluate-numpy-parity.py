"""Compare NumExpr/evaluate with resident arrays and warmed, explicit NumPy out.

Both public APIs are called on every sample. Correctness, output identity and
output casting qualify each case before timing; no decompression or allocation
of a full result is intentionally timed. Route probes run outside timing.
"""

import argparse
import importlib
import json
import os
import random
from math import prod
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numexpr as ne
import numpy as np

import blosc2

RECIPES = {
    "linear": "x + y * 2 - 1",
    "polynomial": "x * x + y * y + x * y",
    "division": "x / (y + 1)",
    "power": "x ** y",
    "sqrt": "sqrt(x * x + y * y)",
    "trig": "sin(x) + cos(y)",
    "log-exp": "log1p(x) + expm1(y)",
    "hypot": "hypot(x, y)",
    "fmod": "fmod(y, x + 0.1)",
    "round": "floor(x * 3) + ceil(y)",
    "predicate": "isfinite(x) & (x < y)",
    "where": "where(x > 0.5, x + y, x - y)",
    "copy": "copy(x)",
    "sum": "sum(x * y)",
    "complex": "complex(x, y)",
}


def make_inputs(size, dtype):
    if np.dtype(dtype).kind in "iu":
        return {"x": np.arange(size, dtype=dtype) % 97 + 1, "y": np.arange(size, dtype=dtype) % 31 + 1}
    return {"x": np.linspace(0.1, 0.9, size, dtype=dtype), "y": np.linspace(1.1, 1.9, size, dtype=dtype)}


def execution_route(operation):
    native = importlib.import_module("blosc2.native_graph")
    original_native, original_lower = native.compute_native_graph, native.lower_native_graph
    observed = set()

    def native_call(*args, **kwargs):
        observed.add("native")
        return original_native(*args, **kwargs)

    def lower_call(*args, **kwargs):
        result = original_lower(*args, **kwargs)
        observed.add("native")
        return result

    with (
        patch.object(native, "compute_native_graph", native_call),
        patch.object(native, "lower_native_graph", lower_call),
    ):
        operation()
    return "native" if "native" in observed else "numpy/eager"


def compare_case(expression, inputs, samples, warmups, rng, order="K"):
    options = {} if order == "K" else {"order": order}
    reference = ne.evaluate(expression, local_dict=inputs, **options)
    output = np.empty_like(reference, order=order)
    operations = {
        "numexpr": lambda: ne.evaluate(expression, local_dict=inputs, out=output, **options),
        "blosc2": lambda: blosc2.evaluate(expression, local_dict=inputs, out=output, **options),
    }
    for operation in operations.values():
        result = operation()
        if result is not output:
            raise AssertionError("The public API did not return the supplied destination")
        if reference.dtype.kind in "biu":
            np.testing.assert_array_equal(result, reference)
        else:
            np.testing.assert_allclose(
                result,
                reference,
                rtol=2e-6 if reference.dtype.itemsize <= 4 else 2e-12,
                atol=2e-7 if reference.dtype.itemsize <= 4 else 2e-12,
            )
    del reference
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    route = execution_route(operations["blosc2"])
    times = {name: [] for name in operations}
    for _ in range(samples):
        names = list(operations)
        rng.shuffle(names)
        for name in names:
            start = perf_counter()
            operations[name]()
            times[name].append((perf_counter() - start) * 1000)
    values = {name: median(measurements) for name, measurements in times.items()}
    return {**values, "ratio": values["blosc2"] / values["numexpr"], "route": route}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[100_000, 8_000_000])
    parser.add_argument(
        "--shape", type=int, nargs="+", help="one explicit multidimensional case; overrides items"
    )
    parser.add_argument("--input-order", choices=["C", "F"], default="C")
    parser.add_argument("--order", choices=["C", "F", "A", "K"], default="K")
    parser.add_argument("--dtype", choices=["float32", "float64", "int64"], default="float64")
    parser.add_argument("--cases", nargs="+", choices=tuple(RECIPES), default=list(RECIPES))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--jit", choices=["tcc", "cc"], default="cc")
    parser.add_argument("--backend", choices=["auto", "interpreter", "jit"], default="auto")
    parser.add_argument("--json", help="also save classifications, medians and runtime versions")
    args = parser.parse_args()
    if args.shape:
        if min(args.shape) <= 0:
            parser.error("shape extents must be positive")
        args.items = [prod(args.shape)]
    if min(args.items) <= 0 or args.samples <= 0 or args.warmups < 0 or args.threads <= 0:
        parser.error("items, samples and threads must be positive; warmups must be nonnegative")
    os.environ["ME_DSL_JIT_COMPILER"] = args.jit
    if args.backend != "auto":
        os.environ["ME_DSL_JIT"] = "0" if args.backend == "interpreter" else "1"
        blosc2.set_jit_options(jit=args.backend == "jit")
    blosc2.set_nthreads(args.threads)
    ne.set_num_threads(args.threads)
    print(
        f"NumPy {np.__version__} | NumExpr {ne.__version__} | Blosc2 {blosc2.__version__}"
        f" | {args.dtype} | threads {args.threads} | backend {args.backend} | compiler {args.jit}"
    )
    probe = blosc2.NativeGraph.from_expression("x + 1", {"x": args.dtype}, jit=True)
    print(
        f"Native JIT available/enabled: {probe.has_jit}; identity maps may use native copies without a kernel."
    )
    if args.backend == "jit" and not probe.has_jit:
        parser.error("JIT was requested but is unavailable")
    print("Resident inputs; shared warmed NumPy out; public API preparation/rebinding included.")
    print(f"Input order {args.input_order}; destination order {args.order}; shape {args.shape or 'flat'}")
    print(f"{'case':<14} {'items':>10} {'numexpr ms':>12} {'blosc2 ms':>12} {'BL/NE':>8}  route")
    records = []
    rng = random.Random(42)
    for size in args.items:
        inputs = make_inputs(size, args.dtype)
        if args.shape:
            inputs = {
                name: np.array(value.reshape(args.shape), order=args.input_order, copy=True)
                for name, value in inputs.items()
            }
        for name in args.cases:
            record = {"case": name, "expression": RECIPES[name], "items": size, "dtype": args.dtype}
            try:
                result = compare_case(
                    RECIPES[name], inputs, args.samples, args.warmups, rng, order=args.order
                )
                record.update(status="qualified", **result)
                print(
                    f"{name:<14} {size:>10} {result['numexpr']:>12.3f} {result['blosc2']:>12.3f}"
                    f" {result['ratio']:>8.2f}  {result['route']}",
                    flush=True,
                )
            except Exception as error:
                record.update(status="unqualified", error=f"{type(error).__name__}: {error}")
                print(f"{name:<14} {size:>10}  N/A: {record['error'].splitlines()[0]}", flush=True)
            records.append(record)
        del inputs
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(
                {
                    "versions": {
                        "numpy": np.__version__,
                        "numexpr": ne.__version__,
                        "blosc2": blosc2.__version__,
                    },
                    "configuration": vars(args),
                    "results": records,
                },
                stream,
                indent=2,
            )
            stream.write("\n")


if __name__ == "__main__":
    main()
