"""Qualify loop materialization against an explicitly internal legacy control.

The legacy SourceKernel fixture reuses normalized author source solely to select
the old engine. It is not supported authoring or source-recipe persistence.
Kernel bodies, float64 intermediates and float32 output are never rewritten.
"""

import argparse
import cProfile
import json
import platform
import runpy
from pathlib import Path
from time import perf_counter

import numpy as np

import blosc2
from blosc2.dsl_kernel import SourceKernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--compiler", choices=("tcc", "cc"), default="tcc")
    parser.add_argument("--route", choices=("menudet", "legacy"), nargs="+", default=("menudet", "legacy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    blosc2.set_nthreads(args.threads)
    benchmark = runpy.run_path(str(Path(__file__).with_name("dsl-kernel-bench.py")))
    n = args.side
    params = blosc2.CParams(codec=blosc2.Codec.BLOSCLZ, clevel=1, nthreads=args.threads)
    x = blosc2.linspace(0, 1, n * n, shape=(n, n), dtype=np.float32, cparams=params)
    y = blosc2.linspace(1, 0, n * n, shape=(n, n), dtype=np.float32, cparams=params)
    storage = {"chunks": x.chunks, "blocks": x.blocks, "cparams": params}
    numpy_inputs = {"x": x[:], "y": y[:]}
    inputs = {"x": x, "y": y}
    cases = [(f"loop{steps}", benchmark["expr_for_steps"](steps)) for steps in (1, 2, 4)]
    cases += [
        ("loop4_heavy", benchmark["expr_for_steps_heavy"](4)),
        ("nested2", benchmark["expr_nested2"]()),
    ]
    metadata = {
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "shape": x.shape,
        "dtype": x.dtype.str,
        "chunks": x.chunks,
        "blocks": x.blocks,
        "cparams": repr(params),
        "threads": args.threads,
        "compiler_requested": args.compiler,
        "repeat": args.repeat,
        "protocol": "validate first output; best of warm allocation/compression samples",
    }
    rows = []
    with blosc2.jit_options(jit=True, jit_backend=args.compiler):
        for name, expression in cases:
            author = benchmark[f"kernel_{name}"]
            expected = eval(expression, {"__builtins__": {}, **vars(np)}, numpy_inputs)
            for route in args.route:
                start = perf_counter()
                has_jit = inferred = partitionable = None
                if route == "menudet":
                    kernel = author.specialize(**inputs)
                    lazy = kernel.lazy(inputs)
                    has_jit = kernel.has_jit
                    inferred = str(kernel.inferred_dtype)
                    partitionable = kernel.native_partitionable
                else:
                    # Deliberately internal old-engine fixture, not a Menudet migration shortcut.
                    fixture = SourceKernel.from_source(author._author.dsl_source)
                    lazy = blosc2.lazyudf(fixture, (x, y), np.float32, jit=True, jit_backend=args.compiler)
                    lazy._legacy_source_recipe = True
                preparation = perf_counter() - start
                start = perf_counter()
                result = lazy.compute(**storage)
                first = perf_counter() - start
                assert result.shape == x.shape
                assert result.dtype == x.dtype
                np.testing.assert_allclose(
                    result[:], expected, rtol=1e-5, atol=32 * np.finfo(np.float32).eps
                )
                if route == "legacy":
                    has_jit = x._dsl_jit_status(fixture.dsl_source, inputs)
                del result
                samples = []
                for _ in range(args.repeat):
                    start = perf_counter()
                    result = lazy.compute(**storage)
                    samples.append(perf_counter() - start)
                    del result
                report = getattr(lazy, "_source_execution_report", None)
                row = {
                    "case": name,
                    "route": route,
                    "preparation_s": preparation,
                    "first_s": first,
                    "samples_s": samples,
                    "best_s": min(samples),
                    "has_jit": has_jit,
                    "inferred_dtype": inferred,
                    "native_partitionable": partitionable,
                    "execution_report": report,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
                args.output.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n")
                if args.profile and route == "menudet" and name == "loop1":
                    profile = cProfile.Profile()
                    profile.runcall(lambda lazy=lazy: lazy.compute(**storage))
                    profile.dump_stats(str(args.profile))


if __name__ == "__main__":
    main()
