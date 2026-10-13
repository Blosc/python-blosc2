"""Compare compute engines for ``x + (y - 1)`` over NumPy and Blosc2 operands.

A single human-readable table reports warm medians for the same expression
evaluated by four engines -- NumPy, numexpr, LazyExpr and a Menudet kernel -- on
resident NumPy operands and on compressed Blosc2 operands (both filled with
``numpy.linspace``). The fastest figure of every row is highlighted in green.

For Blosc2 operands, NumPy and numexpr must first materialize (decompress) the
inputs, and that cost is included in their timings. LazyExpr and the Menudet
kernel evaluate directly over the compressed chunks, so the table shows the
practical difference between the two input storages.

Usage::

    python bench/menudet-engines.py --items 100000 1000000 8000000
"""

import argparse
import random
import sys
from statistics import median
from time import perf_counter

import numexpr as ne
import numpy as np

import blosc2

EXPRESSION = "x + (y - 1)"
ENGINES = ("NumPy", "numexpr", "LazyExpr", "Menudet")
GREEN = "\033[1;32m"
RESET = "\033[0m"


def human_items(count):
    """Format an element count with a compact SI-style suffix (100K, 1M, 8M, 100M)."""
    for suffix, factor in (("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if count >= factor:
            value = f"{count / factor:.1f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return str(count)


@blosc2.menudet
def adjusted(x, y):
    return x + (y - 1)


def build_operands(size, kind):
    """Return the ``x``/``y`` operands and the expected result for one storage kind."""
    x = np.linspace(0, 1, size)
    y = np.linspace(1, 2, size)
    expected = x + (y - 1)
    if kind == "numpy":
        return {"x": x, "y": y}, expected
    layout = {} if size <= 1_000_000 else {"chunks": (1_000_000,), "blocks": (15625,)}
    inputs = {name: blosc2.asarray(value, **layout) for name, value in {"x": x, "y": y}.items()}
    return inputs, expected


def make_engines(inputs, kind):
    """Return an ordered mapping of engine name to a callable producing a result.

    Over Blosc2 operands the NumPy and numexpr callables include operand
    materialization, while LazyExpr and Menudet read the compressed inputs.
    """
    if kind == "numpy":

        def numpy_run():
            return inputs["x"] + (inputs["y"] - 1)

        def numexpr_run():
            return ne.evaluate(EXPRESSION, local_dict=inputs)

    else:

        def numpy_run():
            x = inputs["x"][:]
            y = inputs["y"][:]
            return x + (y - 1)

        def numexpr_run():
            resident = {name: value[:] for name, value in inputs.items()}
            return ne.evaluate(EXPRESSION, local_dict=resident)

    lazy = blosc2.lazyexpr(EXPRESSION, inputs)
    menudet = adjusted(**inputs)
    return {
        "NumPy": numpy_run,
        "numexpr": numexpr_run,
        "LazyExpr": lambda lazy=lazy: lazy[:],
        "Menudet": lambda menudet=menudet: menudet[:],
    }


def benchmark(engines, expected, samples, warmups, rng):
    """Correctness-check once, warm up, then time interleaved shuffled samples."""
    for operation in engines.values():
        result = operation()
        np.testing.assert_array_equal(result, expected)
        del result
    for _ in range(warmups):
        for operation in engines.values():
            result = operation()
            del result
    times = {name: [] for name in engines}
    for _ in range(samples):
        order = list(engines)
        rng.shuffle(order)
        for name in order:
            start = perf_counter()
            result = engines[name]()
            times[name].append((perf_counter() - start) * 1000)
            del result
    return {name: median(values) for name, values in times.items()}


def column_widths(items):
    """Fix column widths up front so rows can be streamed while staying aligned."""
    item_width = max(len("items"), *(len(human_items(size)) for size in items))
    operand_width = len("operands")
    engine_width = max(len(name) for name in ENGINES) + 1
    return item_width, operand_width, engine_width


def table_header(widths):
    item_width, operand_width, engine_width = widths
    cells = [
        "items".rjust(item_width),
        "operands".ljust(operand_width),
        *(name.rjust(engine_width) for name in ENGINES),
    ]
    return "  ".join(cells)


def table_separator(widths):
    item_width, operand_width, engine_width = widths
    per_column = [item_width, operand_width, *([engine_width] * len(ENGINES))]
    return "  ".join("-" * width for width in per_column)


def result_row(row, widths, use_color):
    """Format one (items, operands) row, highlighting its fastest engine."""
    item_width, operand_width, engine_width = widths
    times = row["times"]
    winner = min(times.values())
    cells = [human_items(row["items"]).rjust(item_width), row["operands"].ljust(operand_width)]
    for name in ENGINES:
        text = f"{times[name]:.3f}".rjust(engine_width)
        if use_color and times[name] == winner:
            text = f"{GREEN}{text}{RESET}"
        cells.append(text)
    return "  ".join(cells)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=int, nargs="+", default=[100_000, 1_000_000, 8_000_000, 100_000_000])
    parser.add_argument(
        "--samples", type=int, default=3, help="timed repetitions per row; the median is reported"
    )
    parser.add_argument(
        "--warmups", type=int, default=3, help="untimed repetitions run before timing each row"
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--operands", choices=("both", "numpy", "blosc2"), default="both")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args()
    if args.samples <= 0 or args.warmups < 0 or any(size <= 0 for size in args.items):
        parser.error("items and samples must be positive; warmups must be non-negative")
    if args.threads <= 0:
        parser.error("threads must be positive")

    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    blosc2.set_nthreads(args.threads)
    ne.set_num_threads(args.threads)
    kinds = ("numpy", "blosc2") if args.operands == "both" else (args.operands,)

    print(f"Expression: {EXPRESSION}")
    print("Warm medians in milliseconds (lower is better); fastest figure per row in green.")
    print(
        f"blosc2 {blosc2.__version__} | numpy {np.__version__} | numexpr {ne.__version__}"
        f" | dtype float64 | threads {args.threads} | samples {args.samples}"
    )
    print()
    widths = column_widths(args.items)
    print(table_header(widths))
    print(table_separator(widths), flush=True)

    rng = random.Random(42)
    previous = None
    for size in args.items:
        for kind in kinds:
            inputs, expected = build_operands(size, kind)
            engines = make_engines(inputs, kind)
            times = benchmark(engines, expected, args.samples, args.warmups, rng)
            del inputs, engines, expected
            if previous is not None and size != previous:
                print(table_separator(widths))
            row = {"items": size, "operands": kind, "times": times}
            print(result_row(row, widths, use_color), flush=True)
            previous = size

    print()
    print(
        "Blosc2 operands: NumPy and numexpr figures include operand decompression; "
        "LazyExpr and Menudet evaluate the compressed chunks directly."
    )


if __name__ == "__main__":
    main()
