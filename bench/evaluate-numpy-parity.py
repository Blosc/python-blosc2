"""Compare NumPy, optional NumExpr and evaluate with warmed, explicit NumPy out.

Both public APIs are called on every sample. Correctness, output identity and
output casting qualify each engine before timing. NumPy expressions may allocate
intermediates before copying to out; Blosc2 reports its executed native route.
There is no decompression in this resident benchmark. Route probes run untimed.
"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
from math import prod
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np

import blosc2

try:
    import numexpr as ne
except ImportError:
    ne = None

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
    "minimum": "minimum(x, y)",
    "maximum": "maximum(x, y)",
    "extrema-chain": "minimum(maximum(x, y), x + 2)",
    "copy": "copy(x)",
    "sum": "sum(x * y)",
    "complex": "complex(x, y)",
}


def make_inputs(size, dtype):
    if np.dtype(dtype).kind in "iu":
        return {"x": np.arange(size, dtype=dtype) % 97 + 1, "y": np.arange(size, dtype=dtype) % 31 + 1}
    return {"x": np.linspace(0.1, 0.9, size, dtype=dtype), "y": np.linspace(1.1, 1.9, size, dtype=dtype)}


def execution_route(operation, *, return_report=False):
    native = importlib.import_module("blosc2.native_graph")
    original_native = native.compute_native_graph
    original_specialize = blosc2.NativeGraph.specialize
    observed = set()
    reports = []

    def record(report):
        reports.append(
            {
                key: report[key]
                for key in (
                    "temporary_bytes",
                    "gathered_bytes",
                    "scattered_bytes",
                    "intermediate_bytes",
                    "maximum_input_items",
                    "tasks",
                    "output_copy_bytes",
                    "input_materialization_bytes",
                    "jit_stages",
                    "interpreter_stages",
                )
                if key in report
            }
        )
        if report.get("jit_stages", 0):
            observed.add("native-jit")
        elif report.get("interpreter_stages", 0):
            observed.add("native-interpreter")
        elif report.get("backend"):
            observed.add("native-jit" if report["backend"] == "menudet-jit" else "native-interpreter")
        else:
            observed.add("native-copy")

    def native_call(*args, **kwargs):
        result = original_native(*args, **kwargs)
        if result is not None:
            record(getattr(args[0], "_native_execution_report", {}))
        return result

    class ScheduleProbe:
        def __init__(self, schedule):
            self.schedule = schedule

        def __getattr__(self, name):
            return getattr(self.schedule, name)

        def execute(self, *args, **kwargs):
            result = self.schedule.execute(*args, **kwargs)
            record(result[1])
            return result

        def execute_sources(self, *args, **kwargs):
            result = self.schedule.execute_sources(*args, **kwargs)
            record(result[1])
            return result

    def specialize(*args, **kwargs):
        return ScheduleProbe(original_specialize(*args, **kwargs))

    with (
        patch.object(native, "compute_native_graph", native_call),
        patch.object(blosc2.NativeGraph, "specialize", specialize),
    ):
        operation()
    route = "+".join(sorted(observed)) if observed else "numpy/eager"
    return (route, reports) if return_report else route


def numpy_evaluate(expression, inputs, out=None, order="K"):
    """Independent NumPy reference for the script's fixed recipe vocabulary."""

    if expression == "copy(x)" and out is not None:
        np.copyto(out, inputs["x"], casting="same_kind")
        return out

    def complex_parts(real, imag):
        real, imag = np.broadcast_arrays(real, imag)
        result = np.empty(real.shape, dtype="complex128") if out is None else out
        result.real, result.imag = real, imag
        return result

    names = {
        name: getattr(np, name)
        for name in (
            "sqrt",
            "sin",
            "cos",
            "log1p",
            "expm1",
            "hypot",
            "fmod",
            "floor",
            "ceil",
            "isfinite",
            "where",
            "minimum",
            "maximum",
            "ones_like",
            "sum",
        )
    }
    names.update(copy=lambda x: np.array(x, copy=True, order=order), complex=complex_parts)
    result = np.asarray(eval(expression, {"__builtins__": {}, **names}, inputs))
    if out is not None:
        if result is not out:
            np.copyto(out, result, casting="same_kind")
        return out
    return result


def assert_reference(actual, reference):
    if reference.dtype.kind in "biu":
        np.testing.assert_array_equal(actual, reference)
    else:
        np.testing.assert_allclose(
            actual,
            reference,
            rtol=2e-6 if reference.dtype.itemsize <= 4 else 2e-12,
            atol=2e-7 if reference.dtype.itemsize <= 4 else 2e-12,
        )


def compare_case(expression, inputs, samples, warmups, rng, order="K"):
    options = {} if order == "K" else {"order": order}
    reference = numpy_evaluate(expression, inputs, order=order)
    # Numerical values alone cannot qualify a timing with different result
    # promotion. NumPy defines the contract; a non-equivalent optional NumExpr
    # reference becomes N/A without excluding a qualified Blosc2/NumPy case.
    native_dtype = blosc2.evaluate(expression, local_dict=inputs).dtype
    if native_dtype != reference.dtype:
        raise TypeError(f"result dtype differs: Blosc2 {native_dtype}, NumPy {reference.dtype}")
    output = np.empty_like(reference, order=order)
    operations = {
        "numpy": lambda: numpy_evaluate(expression, inputs, out=output, order=order),
        "blosc2": lambda: blosc2.evaluate(expression, local_dict=inputs, out=output, **options),
    }
    reference_error = None
    if ne is not None:
        try:
            external = ne.evaluate(expression, local_dict=inputs, **options)
            if external.dtype != reference.dtype:
                raise TypeError(f"result dtype differs: NumExpr {external.dtype}, NumPy {reference.dtype}")
            assert_reference(external, reference)
            if ne.evaluate(expression, local_dict=inputs, out=output, **options) is not output:
                raise AssertionError("NumExpr did not return the supplied destination")
            operations["numexpr"] = lambda: ne.evaluate(expression, local_dict=inputs, out=output, **options)
        except Exception as error:
            reference_error = f"{type(error).__name__}: {error}"
    for operation in operations.values():
        result = operation()
        if result is not output:
            raise AssertionError("The public API did not return the supplied destination")
        assert_reference(result, reference)
    del reference
    for _ in range(warmups):
        for operation in operations.values():
            operation()
    route, reports = execution_route(operations["blosc2"], return_report=True)
    times = {name: [] for name in operations}
    for _ in range(samples):
        names = list(operations)
        rng.shuffle(names)
        for name in names:
            start = perf_counter()
            operations[name]()
            times[name].append((perf_counter() - start) * 1000)
    values = {name: median(measurements) for name, measurements in times.items()}
    return {
        **values,
        "numexpr": values.get("numexpr"),
        "numexpr_error": reference_error,
        "ratio": values["blosc2"] / values["numexpr"] if "numexpr" in values else None,
        "route": route,
        "execution_reports": reports,
        "output_bytes": output.nbytes,
        "spread_ms": {name: max(samples) - min(samples) for name, samples in times.items()},
    }


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
    parser.add_argument("--native-revision", help="explicit revision used to build the measured extension")
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
    if ne is not None:
        ne.set_num_threads(args.threads)
    print(
        f"NumPy {np.__version__} | NumExpr {ne.__version__ if ne is not None else 'unavailable'} | Blosc2 {blosc2.__version__}"
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
    print(
        f"{'case':<14} {'items':>10} {'numpy ms':>12} {'numexpr ms':>12} {'blosc2 ms':>12} {'BL/NE':>8}  route"
    )
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
                external = f"{result['numexpr']:.3f}" if result["numexpr"] is not None else "N/A"
                ratio = f"{result['ratio']:.2f}" if result["ratio"] is not None else "N/A"
                print(
                    f"{name:<14} {size:>10} {result['numpy']:>12.3f} {external:>12} {result['blosc2']:>12.3f}"
                    f" {ratio:>8}  {result['route']}",
                    flush=True,
                )
            except Exception as error:
                record.update(status="unqualified", error=f"{type(error).__name__}: {error}")
                print(
                    f"{name:<14} {size:>10} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>8}  unqualified",
                    flush=True,
                )
            records.append(record)
        del inputs
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(
                {
                    "versions": {
                        "numpy": np.__version__,
                        "numexpr": ne.__version__ if ne is not None else None,
                        "blosc2": blosc2.__version__,
                    },
                    "configuration": vars(args),
                    "platform": platform.platform(),
                    "python": sys.version,
                    "native_probe_has_jit": probe.has_jit,
                    "package": blosc2.__file__,
                    "extension_sha256": hashlib.sha256(
                        Path(blosc2.blosc2_ext.__file__).read_bytes()
                    ).hexdigest(),
                    "numpy_temporaries": "Chained recipes allocate intermediates; copy and complex write directly to out.",
                    "results": records,
                },
                stream,
                indent=2,
            )
            stream.write("\n")


if __name__ == "__main__":
    main()
