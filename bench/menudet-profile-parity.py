"""Application-layer paired timings; native numerical matrices live in miniexpr.

Run with the intended editable/isolated package, e.g. --items 1000000 10000000.
Results are JSONL, flushed per configuration. Unsupported/mismatching legacy
controls are retained and excluded from timing rankings. No timing assertions.
"""

import argparse
import gc
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

import blosc2
from blosc2.dsl_kernel import SourceKernel

RECIPES = [
    ("add-i8", "int8", "x + y", "int8"),
    ("add-i64", "int64", "x + y", "int64"),
    ("add-f64", "float64", "x + y", "float64"),
    ("literal-i8", "int8", "x + 1", "int8"),
    ("literal-i64", "int64", "x + 1", "int64"),
    ("deep-i64", "int64", "((x + y) * 2 - y) * 3 + x * 2 - y", "int64"),
    ("compare-i64", "int64", "x < y", "bool"),
    ("compare-f64", "float64", "x < y", "bool"),
    ("trig-f64", "float64", "sin(x) + cos(x) + 1", "float64"),
    ("minimum-f64", "float64", "minimum(x, y)", "float64"),
    ("predicate-f64", "float64", "isfinite(x)", "bool"),
    ("cast-i8-i64", "int8", "int64(x) + 1", "int64"),
]


def read_result(lazy, mode, selection, count):
    if mode == "numpy":
        return lazy[selection]
    return lazy.compute(chunks=(min(262144, count),), blocks=(min(32768, count),), cparams={"clevel": 0})


def main():  # noqa: C901 -- explicit serial benchmark matrix
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--items", type=int, nargs="+", default=[1_000_000, 10_000_000])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 12])
    parser.add_argument("--backends", nargs="+", choices=["tcc", "cc"], default=["tcc", "cc"])
    parser.add_argument("--layouts", nargs="+", default=["resident", "compressed", "strided", "broadcast", "partial", "step"])
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.repeat < 3 or any(n < 1 for n in args.items + args.threads):
        parser.error("positive sizes/threads and at least three repeats required")
    if args.output.exists():
        parser.error("choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "package": blosc2.__file__, "extension": blosc2.blosc2_ext.__file__,
        "platform": platform.platform(), "numpy": np.__version__,
        "arguments": vars(args) | {"output": str(args.output)},
    }
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    previous = blosc2.nthreads
    try:
        with args.output.open("w") as stream:
            for backend in args.backends:
                os.environ["ME_DSL_JIT_COMPILER"] = backend
                for name, dtype, expression, output in RECIPES:
                    if args.names and name not in args.names:
                        continue
                    source = f"def k(x, y):\n    return {expression}\n"
                    author = SourceKernel.from_source(source)
                    kernel = blosc2.MenudetKernel.from_json(author.export({"x": dtype, "y": dtype}, output), jit=True)
                    for count in args.items:
                        x = (np.arange(count) % 3 + 1).astype(dtype)
                        y = (np.arange(count) % 2 + 1).astype(dtype)
                        compressed = None
                        for layout in args.layouts:
                            selection = slice(0, count // 10) if layout == "partial" else slice(None, None, 4) if layout == "step" else slice(None)
                            if layout == "compressed":
                                if compressed is None:
                                    compressed = {
                                        key: blosc2.asarray(value, chunks=(min(262144, count),), blocks=(min(32768, count),),
                                                            cparams={"clevel": 1}, dparams={"nthreads": 1})
                                        for key, value in {"x": x, "y": y}.items()
                                    }
                                operands = compressed
                                resident = {"x": x, "y": y}
                            elif layout == "strided":
                                operands = resident = {"x": x[::-1], "y": y[::-1]}
                            elif layout == "broadcast":
                                operands = resident = {"x": x, "y": y[:1]}
                            else:
                                operands = resident = {"x": x, "y": y}
                            namespace = {"__builtins__": {}, "sin": np.sin, "cos": np.cos,
                                         "minimum": np.minimum, "isfinite": np.isfinite,
                                         "int64": lambda v: v.astype("int64"), **resident}
                            expected = eval(expression, namespace)[selection]
                            for threads in args.threads:
                                blosc2.set_nthreads(threads)
                                modern = kernel.lazy(operands)
                                old = blosc2.lazyudf(author, tuple(operands.values()), dtype=output, jit_backend=backend)
                                modes = ["numpy"]
                                if layout in ("resident", "compressed") and name in ("add-i8", "add-i64", "add-f64", "trig-f64"):
                                    modes.append("blosc2")
                                for mode in modes:
                                    row = {"name": name, "dtype": dtype, "output_dtype": output, "items": count,
                                           "threads": threads, "backend": backend, "layout": layout,
                                           "output": mode, "menudet_jit": kernel.has_jit}
                                    valid_engines = []
                                    row["status"] = "ok"
                                    for label, lazy in (("menudet", modern), ("legacy", old)):
                                        try:
                                            actual = read_result(lazy, mode, selection, count)
                                            values = actual[:] if isinstance(actual, blosc2.NDArray) else actual
                                            np.testing.assert_allclose(values, expected, rtol=2e-12, atol=2e-12)
                                            assert values.dtype == np.dtype(output)
                                            del actual, values
                                            valid_engines.append((label, lazy))
                                        except Exception as error:
                                            row["status"] = f"{label}_error"
                                            row["error"] = f"{type(error).__name__}: {error}"
                                    if valid_engines:
                                        times = {"menudet": [], "legacy": []}
                                        for repeat in range(args.repeat):
                                            engines = valid_engines.copy()
                                            if repeat % 2:
                                                engines.reverse()
                                            for label, lazy in engines:
                                                gc.collect()
                                                start = perf_counter()
                                                actual = read_result(lazy, mode, selection, count)
                                                times[label].append(perf_counter() - start)
                                                del actual
                                        row["samples"] = times
                                        for label, samples in times.items():
                                            if samples:
                                                row[f"{label}_s"] = median(samples)
                                        if all(times.values()):
                                            row["ratio"] = row["menudet_s"] / row["legacy_s"]
                                        row["source_report"] = getattr(modern, "_source_execution_report", None)
                                    stream.write(json.dumps(row) + "\n")
                                    stream.flush()
                                    print(row["name"], backend, count, layout, threads, mode,
                                          row["status"], round(row.get("ratio", 0), 2), flush=True)
                        del compressed, x, y
    finally:
        blosc2.set_nthreads(previous)


if __name__ == "__main__":
    main()
