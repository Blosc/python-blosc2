"""Compare compute engines for ``x + (y - 1)`` over NumPy and Blosc2 operands.

A single human-readable table reports warm medians for the same expression
evaluated by NumPy, numexpr, both LazyExpr output paths, blosc2.evaluate and a Menudet kernel -- on
resident NumPy operands and on compressed Blosc2 operands (both filled with
``numpy.linspace``). The fastest figure of every row is highlighted in green.

For Blosc2 operands, NumPy and numexpr must first materialize (decompress) the
inputs, and that cost is included in their timings. Blosc2-output rows exercise
the existing compute paths, whose input materialization and scheduling may differ
from the NumPy-output paths.
The evaluate column calls ``blosc2.evaluate`` afresh each time. Repeated
preparation uses an operand-free recipe cache; input rebinding and fresh NumPy
output allocation are timed, and no result buffer is reused. The evaluate column
always returns NumPy output for elementwise expressions, as do NumPy, numexpr
and LazyExpr-NumPy. LazyExpr always returns uncompressed Blosc2 output. Menudet
returns uncompressed Blosc2 output for Blosc2 inputs and NumPy output for NumPy
inputs. Blosc2 outputs use the input chunk/block layout for Blosc2 inputs.

With ``--reduc``, each engine returns a scalar. Menudet uses a native block
reduction, which requires resident operands (decompression is timed for Blosc2).
Numexpr's mean is implemented as sum/count. Reduction threading depends on the
engine and is not necessarily the configured thread count.
Numexpr any/all use max/min of exact floating 0/1 truth values.

Usage::

    python bench/menudet-engines.py --items 100000 1000000 8000000
    python bench/menudet-engines.py --reduc sum
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
ENGINES = ("NumPy", "numexpr", "LazyExpr-NumPy", "LazyExpr", "evaluate", "Menudet")
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


@blosc2.menudet(cardinality="block_scalar")
def adjusted_sum(x, y):
    return block_sum(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


@blosc2.menudet(cardinality="block_scalar")
def adjusted_mean(x, y):
    return block_mean(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


@blosc2.menudet(cardinality="block_scalar")
def adjusted_min(x, y):
    return block_min(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


@blosc2.menudet(cardinality="block_scalar")
def adjusted_max(x, y):
    return block_max(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


@blosc2.menudet(cardinality="block_scalar")
def adjusted_any(x, y):
    return block_any(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


@blosc2.menudet(cardinality="block_scalar")
def adjusted_all(x, y):
    return block_all(x + (y - 1))  # noqa: F821 -- native DSL intrinsic


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


def make_engines(inputs, kind, reduc=None):
    """Return an ordered mapping of engine name to a callable producing a result.

    LazyExpr returns uncompressed Blosc2 output; LazyExpr-NumPy, NumPy, numexpr
    and evaluate always return NumPy. Menudet preserves the input storage kind.
    """
    numpy_reduce = getattr(np, reduc) if reduc else None
    numexpr_expression = f"{('sum' if reduc == 'mean' else reduc)}({EXPRESSION})" if reduc else EXPRESSION
    if reduc in ("any", "all"):
        # Numexpr has no any/all intrinsic; reduce exact floating 0/1 values.
        # Float literals also avoid integer-extrema seed narrowing on NumPy 2.
        operation = "max" if reduc == "any" else "min"
        numexpr_expression = f"{operation}(where(({EXPRESSION}) != 0, 1.0, 0.0))"

    def finish_numpy(values):
        return numpy_reduce(values) if reduc else values

    def finish_numexpr(resident):
        result = ne.evaluate(numexpr_expression, local_dict=resident)
        if reduc == "mean":
            return result.item() / inputs["x"].size
        if reduc in ("any", "all"):
            return bool(result.item())
        return result.item() if reduc else result

    if kind == "numpy":

        def numpy_run():
            return finish_numpy(inputs["x"] + (inputs["y"] - 1))

        def numexpr_run():
            return finish_numexpr(inputs)

    else:

        def numpy_run():
            x = inputs["x"][:]
            y = inputs["y"][:]
            return finish_numpy(x + (y - 1))

        def numexpr_run():
            resident = {name: value[:] for name, value in inputs.items()}
            return finish_numexpr(resident)

    lazy = blosc2.lazyexpr(EXPRESSION, inputs)
    if reduc:
        author = {
            "sum": adjusted_sum,
            "mean": adjusted_mean,
            "min": adjusted_min,
            "max": adjusted_max,
            "any": adjusted_any,
            "all": adjusted_all,
        }[reduc]
        kernel = author.specialize(**inputs)

        def menudet_reduce():
            resident = inputs if kind == "numpy" else {name: value[:] for name, value in inputs.items()}
            return kernel.evaluate_block(resident).item()

        return {
            "NumPy": numpy_run,
            "numexpr": numexpr_run,
            "LazyExpr-NumPy": getattr(lazy, reduc),
            "LazyExpr": getattr(lazy, reduc),
            "evaluate": lambda: blosc2.evaluate(f"{reduc}({EXPRESSION})", local_dict=inputs),
            "Menudet": menudet_reduce,
        }
    menudet = adjusted(**inputs)
    output_options = {"cparams": blosc2.CParams(clevel=0)}
    if kind == "blosc2":
        first = inputs["x"]
        output_options.update(chunks=first.chunks, blocks=first.blocks)
    return {
        "NumPy": numpy_run,
        "numexpr": numexpr_run,
        "LazyExpr-NumPy": lambda: lazy[:],
        "LazyExpr": lambda: lazy.compute(**output_options),
        "evaluate": lambda: blosc2.evaluate(EXPRESSION, local_dict=inputs),
        "Menudet": (lambda: menudet.compute(**output_options)) if kind == "blosc2" else (lambda: menudet[:]),
    }


def benchmark(engines, expected, samples, warmups, rng):
    """Correctness-check once, warm up, then time interleaved shuffled samples."""
    for operation in engines.values():
        result = operation()
        if isinstance(result, blosc2.NDArray):
            result = result[:]
        if np.ndim(expected) == 0:
            if np.ndim(result) != 0:
                raise AssertionError("Reduction must return a scalar")
            # Native/parallel summation can group floating additions differently.
            np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
        else:
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
    parser.add_argument(
        "--reduc",
        choices=("sum", "mean", "min", "max", "any", "all"),
        help="reduce the expression to a scalar",
    )
    args = parser.parse_args()
    if args.samples <= 0 or args.warmups < 0 or any(size <= 0 for size in args.items):
        parser.error("items and samples must be positive; warmups must be non-negative")
    if args.threads <= 0:
        parser.error("threads must be positive")

    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    blosc2.set_nthreads(args.threads)
    ne.set_num_threads(args.threads)
    kinds = ("numpy", "blosc2") if args.operands == "both" else (args.operands,)

    expression = f"{args.reduc}({EXPRESSION})" if args.reduc else EXPRESSION
    print(f"Expression: {expression}")
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
            if args.reduc:
                expected = getattr(np, args.reduc)(expected)
            engines = make_engines(inputs, kind, args.reduc)
            times = benchmark(engines, expected, args.samples, args.warmups, rng)
            del inputs, engines, expected
            if previous is not None and size != previous:
                print(table_separator(widths))
            row = {"items": size, "operands": kind, "times": times}
            print(result_row(row, widths, use_color), flush=True)
            previous = size

    print()
    if args.reduc:
        print(
            "Blosc2 operands: NumPy, numexpr and Menudet include operand decompression; "
            "both LazyExpr columns and evaluate reduce directly."
        )
        print(
            "Menudet uses a native block reduction; numexpr mean uses sum/count. Reduction threading is engine-dependent."
        )
        if args.reduc in ("any", "all"):
            print("Numexpr any/all use 0/1 truth max/min; Menudet block truth reductions may short-circuit.")
    else:
        print(
            "Blosc2 operands: NumPy and numexpr figures include operand decompression; "
            "other engines evaluate directly over compressed inputs when eligible."
        )
        print(
            "NumPy, numexpr, LazyExpr-NumPy and evaluate return NumPy outputs. "
            "LazyExpr returns Blosc2 output (clevel=0); Menudet does so for Blosc2 inputs only."
        )
    print("evaluate caches eligible preparation; input rebinding and fresh output allocation are timed.")


if __name__ == "__main__":
    main()
