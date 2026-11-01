"""Internal legacy control and profiling for the Menudet predicate migration.

SourceKernel and _legacy_source_recipe deliberately select the old native engine;
they are NOT supported public authoring. All measured routes use identical data,
output grids and compression. No portable kernel is passed to lazyudf.
"""

import argparse
import cProfile
import json
import platform
import pstats
from pathlib import Path
from time import perf_counter

import numpy as np
from compute_expr_udf import KERNELS, execution_report, qualified_measure, reference

import blosc2
from blosc2.dsl_kernel import SourceKernel

sin, cos = np.sin, np.cos


@SourceKernel
def legacy1(x):
    return x < 0.5


@SourceKernel
def legacy2(x, y):
    return (x**2 + y**2) <= (2 * x * y + 1)


@SourceKernel
def legacy3(x, y, z):
    return (sin(x) ** 3 + cos(y) ** 2) >= (cos(x) * sin(y) + z)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=10000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--compiler", choices=("tcc", "cc"), default="tcc")
    parser.add_argument("--case", type=int, choices=(1, 2, 3), nargs="+", default=(1, 2, 3))
    parser.add_argument("--route", choices=("menudet", "legacy"), nargs="+", default=("menudet", "legacy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    blosc2.set_nthreads(args.threads)
    shape = (args.side, args.side)
    storage = {
        "chunks": (min(args.side, 1250), args.side),
        "blocks": (min(args.side, 2), args.side),
        "cparams": blosc2.CParams(codec=blosc2.Codec.LZ4, clevel=5, nthreads=args.threads),
    }
    arrays = tuple(
        blosc2.linspace(start, stop, args.side**2, shape=shape, dtype=np.float32, **storage)
        for start, stop in ((0, 1), (-1, 1), (0, 10))
    )
    metadata = {
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "blosc2": blosc2.__version__,
        "c_blosc2": blosc2.blosclib_version,
        "compiler": args.compiler,
        "threads": args.threads,
        "shape": shape,
        "chunks": arrays[0].chunks,
        "blocks": arrays[0].blocks,
        "cparams": repr(storage["cparams"]),
        "repeat": args.repeat,
    }
    rows = []
    with blosc2.jit_options(jit=True, jit_backend=args.compiler):
        for case in args.case:
            inputs = arrays[:case]
            expected = reference(case - 1, tuple(value[:] for value in inputs))
            for route in args.route:
                start = perf_counter()
                if route == "menudet":
                    bindings = dict(zip(("x", "y", "z")[:case], inputs, strict=True))
                    kernel = KERNELS[case - 1].specialize(**bindings)
                    assert kernel.has_jit
                    lazy = kernel.lazy(bindings)
                else:
                    author = (legacy1, legacy2, legacy3)[case - 1]
                    lazy = blosc2.lazyudf(author, inputs, np.bool_, jit=True, jit_backend=args.compiler)
                    lazy._legacy_source_recipe = True
                preparation = perf_counter() - start
                for mode in ("compute", "getitem"):
                    operation = (
                        (lambda lazy=lazy: lazy.compute(**storage))
                        if mode == "compute"
                        else (lambda lazy=lazy: lazy[:])
                    )
                    timing = qualified_measure(
                        operation, expected, materialized=mode == "compute", repeat=args.repeat
                    )
                    probe = None
                    if route == "legacy" and timing["qualified"]:
                        probe = inputs[0]._dsl_jit_status(
                            author.dsl_source, dict(zip(("x", "y", "z")[:case], inputs, strict=True))
                        )
                    row = dict(
                        case=case,
                        route=route,
                        mode=mode,
                        preparation_s=preparation,
                        execution_report=execution_report(lazy),
                        legacy_jit_probe=probe,
                        **timing,
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    args.output.write_text(
                        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n"
                    )
                    if args.profile and route == "menudet" and mode == "compute":
                        profile = cProfile.Profile()
                        profile.runcall(operation)
                        profile.dump_stats(str(args.profile))
                        pstats.Stats(profile).sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    main()
