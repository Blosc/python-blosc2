"""Compare Menudet with NumPy/numexpr, including compressed-input read costs."""

import argparse
import json
import os
import platform
from statistics import median
from time import perf_counter

import numexpr as ne
import numpy as np

import blosc2


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


def measure(operation, expected, samples):
    np.testing.assert_array_equal(operation(), expected)
    times = []
    for _ in range(samples):
        start = perf_counter()
        result = operation()
        times.append((perf_counter() - start) * 1000)
        del result
    return round(median(times), 4)


def numpy_read(inputs):
    x, y = (value[:] for value in inputs.values())
    return x + (y - 1)


def numexpr_read(inputs):
    resident = {name: value[:] for name, value in inputs.items()}
    return ne.evaluate("x + (y - 1)", local_dict=resident)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--items", type=int, nargs="+", default=[100_000, 1_000_000, 8_000_000])
    args = parser.parse_args()
    if args.samples <= 0 or any(size <= 0 for size in args.items):
        parser.error("samples and items must be positive")
    rows = []
    for size in args.items:
        x, y = np.linspace(0, 1, size), np.linspace(1, 2, size)
        expected = x + (y - 1)
        resident = {"x": x, "y": y}
        layout = {} if size <= 1_000_000 else {"chunks": (1_000_000,), "blocks": (15625,)}
        compressed = {name: blosc2.asarray(value, **layout) for name, value in resident.items()}
        lazy_numpy = adjusted(**resident)
        lazy_compressed = adjusted(**compressed)
        assert lazy_numpy.kernel.has_jit
        assert lazy_compressed.kernel.has_jit
        row = {
            "items": size,
            "chunks": compressed["x"].chunks,
            "blocks": compressed["x"].blocks,
            "numpy_ms": measure(lambda x=x, y=y: x + (y - 1), expected, args.samples),
            "menudet_numpy_ms": measure(lambda lazy=lazy_numpy: lazy[:], expected, args.samples),
            "numpy_compressed_ms": measure(lambda inputs=compressed: numpy_read(inputs), expected, args.samples),
            "menudet_compressed_ms": measure(lambda lazy=lazy_compressed: lazy[:], expected, args.samples),
            "menudet_compressed_route": lazy_compressed._source_execution_report,
        }
        for threads in (1, 12):
            ne.set_num_threads(threads)
            assert ne.get_num_threads() == threads, "Requested numexpr thread count was not applied"
            row[f"numexpr_{threads}t_ms"] = measure(
                lambda inputs=resident: ne.evaluate("x + (y - 1)", local_dict=inputs), expected, args.samples
            )
            row[f"numexpr_compressed_{threads}t_ms"] = measure(
                lambda inputs=compressed: numexpr_read(inputs), expected, args.samples
            )
        rows.append(row)
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "numpy": np.__version__,
                "numexpr": ne.__version__,
                "blosc2_threads": blosc2.nthreads,
                "samples": args.samples,
                "expression": "x + (y - 1)",
                "dtype": "float64",
                "compiler": os.environ.get("ME_DSL_JIT_COMPILER", "default"),
                "cc": os.environ.get("CC", "default"),
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "compressed_baseline": "public NDArray[:] materialization, included in timing",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
