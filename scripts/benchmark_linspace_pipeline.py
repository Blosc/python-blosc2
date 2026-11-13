"""Measure construction, two/three-input expressions and reads without changing defaults.

Run with conda Python -S and --package to qualify an isolated rebuilt package.
For example::

    conda run -n blosc2 python -S scripts/benchmark_linspace_pipeline.py \
        --package build_py314/linspace-candidate --output candidate.json

Times exclude validation and result destruction. Memory stays bounded by compressed
arrays and working chunks; no full-domain NumPy reference is allocated.
"""

import argparse
import json
import platform
import statistics
import sys
import sysconfig
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, help="Parent of the isolated blosc2 package")
    parser.add_argument("--side", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--layout", choices=["default", "large", "both"], default="default")
    parser.add_argument("--jit", choices=["inherit", "on", "off"], default="inherit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.side < 2 or args.repeats < 1:
        parser.error("--side must be at least 2 and --repeats must be positive")
    return args


def main():
    args = parse_args()
    if args.package:
        sys.path[:0] = [str(args.package.resolve()), sysconfig.get_paths()["purelib"]]
    import numpy as np

    import blosc2

    if args.package:
        assert Path(blosc2.__file__).is_relative_to(args.package.resolve())
        assert Path(blosc2.blosc2_ext.__file__).is_relative_to(args.package.resolve())
    blosc2.cparams_dflts.update(
        codec=blosc2.Codec.LZ4, filters=[blosc2.Filter.SHUFFLE, blosc2.Filter.BYTEDELTA]
    )
    if args.jit != "inherit":
        blosc2.set_jit_options(jit=args.jit == "on")
    shape = (args.side,) * 3
    size = args.side**3
    indices = np.unique(np.r_[0, size - 1, np.random.default_rng(2026).integers(0, size, 32)])
    coordinates = [tuple(map(int, np.unravel_index(i, shape))) for i in indices]
    reference = indices * (1 / (size - 1))
    layouts = ["default", "large"] if args.layout == "both" else [args.layout]
    report = {
        "platform": platform.platform(),
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "lz4_version": blosc2.clib_info(blosc2.Codec.LZ4)[1].decode(),
        "jit": args.jit,
        "shape": shape,
        "cases": {},
    }

    def measure(name, function, validate):
        walls, cpus = [], []
        for i in range(args.repeats + 1):
            cpu, start = time.process_time(), time.perf_counter()
            result = function()
            wall, cpu = time.perf_counter() - start, time.process_time() - cpu
            validate(result)
            if i:
                walls.append(wall)
                cpus.append(cpu)
            del result
        record = {"wall_seconds": walls, "cpu_seconds": cpus, "median_wall": statistics.median(walls)}
        print(f"{name:30} {record['median_wall'] * 1000:9.2f} ms", flush=True)
        return record

    def check_samples(result, expected):
        actual = np.array([result[c] for c in coordinates])
        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)

    for layout in layouts:
        storage = (
            {}
            if layout == "default"
            else {
                "chunks": (min(4, args.side), args.side, args.side),
                "blocks": (1, min(50, args.side), args.side),
            }
        )
        records = {}
        records["construct"] = measure(
            layout + ": construct",
            lambda storage=storage: blosc2.linspace(0, 1, size, shape=shape, **storage),
            lambda result: check_samples(result, reference),
        )

        def make_inputs(storage=storage):
            return tuple(blosc2.linspace(i, i + 1, size, shape=shape, **storage) for i in range(3))

        def check_inputs(arrays):
            for i, array in enumerate(arrays):
                check_samples(array, i + reference)

        records["construct_inputs"] = measure(layout + ": construct three", make_inputs, check_inputs)
        a = blosc2.linspace(0, 1, size, shape=shape, **storage)
        b = blosc2.linspace(1, 2, size, shape=shape, **storage)
        c = blosc2.linspace(2, 3, size, shape=shape, **storage)
        records["partitions"] = {"chunks": a.chunks, "blocks": a.blocks, "cratio": a.schunk.cratio}
        for name, expression, expected in (
            ("two_inputs", a + b, reference + (1 + reference)),
            ("three_inputs", (a + b) * c, (reference + (1 + reference)) * (2 + reference)),
        ):
            records[name] = measure(
                layout + ": " + name,
                lambda expr=expression, storage=storage: expr.compute(**storage),
                lambda result, expected=expected: check_samples(result, expected),
            )
        selection = (args.side // 2, slice(0, min(10, args.side)), slice(None))
        expected = np.ravel_multi_index((args.side // 2, 0, 0), shape) + np.arange(
            min(10, args.side) * args.side
        )
        expected = (expected * (1 / (size - 1))).reshape(min(10, args.side), args.side)
        records["read_slab"] = measure(
            layout + ": read_slab",
            lambda array=a, selection=selection: array[selection],
            lambda result, expected=expected: np.testing.assert_array_equal(result, expected),
        )
        records["lifecycle_wall"] = (
            records["construct_inputs"]["median_wall"]
            + records["two_inputs"]["median_wall"]
            + records["three_inputs"]["median_wall"]
            + records["read_slab"]["median_wall"]
        )
        report["cases"][layout] = records
        del a, b, c, expression
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
