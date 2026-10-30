"""Compare compute engines for an expression over NumPy and Blosc2 operands.

A single human-readable table reports warm medians for the same expression
evaluated by NumPy, numexpr, both LazyExpr output paths, blosc2.evaluate and a Menudet kernel -- on
resident NumPy operands and on compressed Blosc2 operands (built with their
respective ``arange`` constructors). The fastest figure of every row is highlighted in green.

For Blosc2 operands, NumPy and numexpr must first materialize (decompress) the
inputs, and that cost is included in their timings. Blosc2-output rows exercise
the existing compute paths, whose input materialization and scheduling may differ
from the NumPy-output paths.
The evaluate column calls ``blosc2.evaluate`` afresh each time. Repeated
preparation uses an operand-free recipe cache; input rebinding and fresh NumPy
output allocation are timed, and no result buffer is reused. The evaluate column
always returns NumPy output for elementwise expressions, as do NumPy, numexpr
and LazyExpr-NumPy. LazyExpr always returns Blosc2 output. Menudet returns Blosc2
output for Blosc2 inputs and NumPy output for NumPy inputs. ``--codec`` and
``--clevel`` configure compressed inputs and outputs (defaults: zstd, 5).
Blosc2 outputs use the input chunk/block layout for Blosc2 inputs.
Engines with unsupported dtypes or different numerical results are reported as N/A.
Floating-point arrays use rtol=atol=4*eps of the reference's real dtype;
integer and boolean arrays must match exactly. Numerical failures report the
maximum absolute error and up to three mismatching indices and values.
``--warmups 0`` measures preparation-cold samples: expression and compiler caches
are isolated before every sample, and engine construction/compilation is timed.
Operand construction, imports and cache isolation are not timed; CPU/OS caches
are not flushed. Positive warmup counts retain prepared-engine warm measurements.
``--allocation-control`` adds a separate resident-input comparison of numexpr
and evaluate with fresh versus reused NumPy destinations. It does not change
the fresh-output policy of the main table.

With ``--reduc``, each engine returns a scalar. Menudet uses a native block
reduction, which requires resident operands (decompression is timed for Blosc2).
Numexpr's mean is implemented as sum/count. Reduction threading depends on the
engine and is not necessarily the configured thread count.
Numexpr any/all use max/min of exact floating 0/1 truth values.

Usage::

    python bench/menudet-engines.py --items 100000 1000000 8000000
    python bench/menudet-engines.py --maxitems 1000000
    python bench/menudet-engines.py --maxsize 8M
    python bench/menudet-engines.py --reduc sum
    python bench/menudet-engines.py --jit cc
    python bench/menudet-engines.py --dtype int16 --codec lz4 --clevel 3
    python bench/menudet-engines.py --dtype int64 --jit cc --allocation-control
    python bench/menudet-engines.py --expr "x * y + z"

``--expr`` defaults to ``x + (y - 1)`` and can use operands x, y and z,
constructed as arange(size), arange(1, size + 1) and arange(2, size + 2).
NumPy function names such as sin and sqrt are available without a prefix.

``--jit tcc|cc`` selects miniexpr's JIT compiler backend. With ``cc``, the
``CC`` environment variable selects the system compiler. Omitting ``--jit``
preserves the existing environment/default backend selection.
"""

import argparse
import ast
import importlib
import linecache
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from statistics import median
from time import perf_counter

import numexpr as ne
import numpy as np

import blosc2

EXPRESSION = "x + (y - 1)"
ENGINES = ("NumPy", "numexpr", "LazyExpr-NumPy", "LazyExpr", "evaluate", "Menudet")
GREEN = "\033[1;32m"
RESET = "\033[0m"
MENUDET_AUTHORS = {}
NUMPY_GLOBALS = {**vars(np), "np": np, "__builtins__": {}}


def parse_item_count(value):
    """Parse an integer item count with an optional decimal SI suffix (K, M, G)."""
    match = re.fullmatch(r"([0-9]+)([KMG]?)", value, re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("expected an integer item count, optionally suffixed with K, M or G")
    count, suffix = match.groups()
    return int(count) * {"": 1, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}[suffix.upper()]


def human_items(count):
    """Format an element count with a compact SI-style suffix (100K, 1M, 8M, 100M)."""
    for suffix, factor in (("G", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if count >= factor:
            value = f"{count / factor:.1f}".rstrip("0").rstrip(".")
            return f"{value}{suffix}"
    return str(count)


def operand_names(expression):
    """Return the used operands in a stable order, requiring at least one array."""
    tree = ast.parse(expression, mode="eval")
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    operands = tuple(name for name in ("x", "y", "z") if name in names)
    unknown = names - {"x", "y", "z", "np"} - set(vars(np))
    if unknown:
        raise ValueError(f"Unknown expression names: {', '.join(sorted(unknown))}")
    if not operands:
        raise ValueError("Expression must use at least one operand: x, y or z")
    return operands


def numpy_expression(expression, inputs):
    """Evaluate the CLI expression using NumPy's functions and resident arrays."""
    return eval(expression, NUMPY_GLOBALS, inputs)


def menudet_author(expression, reduc=None):
    """Build a cached Menudet author with inspectable source for the expression."""
    key = (expression, reduc)
    if key not in MENUDET_AUTHORS:
        names = ", ".join(operand_names(expression))
        value = f"block_{reduc}({expression})" if reduc else expression
        source = f"def adjusted({names}):\n    return {ast.unparse(ast.parse(value, mode='eval'))}\n"
        filename = f"<menudet-benchmark-{len(MENUDET_AUTHORS)}>"
        # Menudet inspects the function source; exec alone does not retain it.
        linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)
        namespace = {"np": np}
        exec(compile(source, filename, "exec"), namespace)
        MENUDET_AUTHORS[key] = blosc2.menudet(
            namespace["adjusted"], cardinality="block_scalar" if reduc else "elementwise"
        )
    return MENUDET_AUTHORS[key]


def build_operands(size, kind, dtype="float64", cparams=None, expression=EXPRESSION):
    """Return the used operands and NumPy reference result for one storage kind."""
    names = operand_names(expression)
    offsets = {"x": 0, "y": 1, "z": 2}
    resident = {name: np.arange(offsets[name], size + offsets[name], dtype=dtype) for name in names}
    expected = numpy_expression(expression, resident)
    if kind == "numpy":
        return resident, expected
    layout = {} if size <= 1_000_000 else {"chunks": (1_000_000,), "blocks": (15625,)}
    cparams = blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5) if cparams is None else cparams
    if size == 1:
        # The DSL-backed Blosc2 constructor rejects scalar-like domains.
        return {name: blosc2.asarray(value, cparams=cparams) for name, value in resident.items()}, expected
    inputs = {
        name: blosc2.arange(offsets[name], size + offsets[name], dtype=dtype, cparams=cparams, **layout)
        for name in names
    }
    return inputs, expected


def make_engines(inputs, kind, reduc=None, cparams=None, selected=None, expression=EXPRESSION):  # noqa: C901 -- explicit engine routes
    """Return an ordered mapping of engine name to a callable producing a result.

    LazyExpr returns compressed Blosc2 output; LazyExpr-NumPy, NumPy, numexpr
    and evaluate always return NumPy. Menudet preserves the input storage kind.
    """
    numpy_reduce = getattr(np, reduc) if reduc else None
    numpy_code = compile(expression, "<numpy-benchmark>", "eval")
    numexpr_expression = f"{('sum' if reduc == 'mean' else reduc)}({expression})" if reduc else expression
    if reduc in ("any", "all"):
        # Numexpr has no any/all intrinsic; reduce exact floating 0/1 values.
        # Float literals also avoid integer-extrema seed narrowing on NumPy 2.
        operation = "max" if reduc == "any" else "min"
        numexpr_expression = f"{operation}(where(({expression}) != 0, 1.0, 0.0))"

    def finish_numpy(values):
        return numpy_reduce(values) if reduc else values

    def finish_numexpr(resident):
        result = ne.evaluate(numexpr_expression, local_dict=resident)
        if reduc == "mean":
            return result.item() / next(iter(inputs.values())).size
        if reduc in ("any", "all"):
            return bool(result.item())
        return result.item() if reduc else result

    if kind == "numpy":

        def numpy_run():
            return finish_numpy(numpy_expression(numpy_code, inputs))

        def numexpr_run():
            return finish_numexpr(inputs)

    else:

        def numpy_run():
            resident = {name: value[:] for name, value in inputs.items()}
            return finish_numpy(numpy_expression(numpy_code, resident))

        def numexpr_run():
            resident = {name: value[:] for name, value in inputs.items()}
            return finish_numexpr(resident)

    if selected in ("NumPy", "numexpr", "evaluate"):
        return {
            selected: {
                "NumPy": numpy_run,
                "numexpr": numexpr_run,
                "evaluate": lambda: blosc2.evaluate(
                    f"{reduc}({expression})" if reduc else expression, local_dict=inputs
                ),
            }[selected]
        }
    lazy = blosc2.lazyexpr(expression, inputs) if selected in (None, "LazyExpr", "LazyExpr-NumPy") else None
    if reduc:
        kernel = (
            menudet_author(expression, reduc).specialize(**inputs) if selected in (None, "Menudet") else None
        )

        def menudet_reduce():
            resident = inputs if kind == "numpy" else {name: value[:] for name, value in inputs.items()}
            return kernel.evaluate_block(resident).item()

        return {
            "NumPy": numpy_run,
            "numexpr": numexpr_run,
            "LazyExpr-NumPy": getattr(lazy, reduc) if lazy is not None else None,
            "LazyExpr": getattr(lazy, reduc) if lazy is not None else None,
            "evaluate": lambda: blosc2.evaluate(f"{reduc}({expression})", local_dict=inputs),
            "Menudet": menudet_reduce,
        }
    menudet = menudet_author(expression)(**inputs) if selected in (None, "Menudet") else None
    output_options = {
        "cparams": blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5) if cparams is None else cparams
    }
    if kind == "blosc2":
        first = next(iter(inputs.values()))
        output_options.update(chunks=first.chunks, blocks=first.blocks)
    return {
        "NumPy": numpy_run,
        "numexpr": numexpr_run,
        "LazyExpr-NumPy": lambda: lazy[:],
        "LazyExpr": lambda: lazy.compute(**output_options),
        "evaluate": lambda: blosc2.evaluate(expression, local_dict=inputs),
        "Menudet": (lambda: menudet.compute(**output_options)) if kind == "blosc2" else (lambda: menudet[:]),
    }


def make_allocation_engines(inputs, expression=EXPRESSION):
    """Compare fresh/reused NumPy destinations without changing the main table.

    Reused cases share one buffer: benchmark calls are sequential and qualify
    each result immediately. No operands are decompressed in this control.
    """
    output = np.empty_like(numpy_expression(expression, inputs))
    return {
        "numexpr fresh": lambda: ne.evaluate(expression, local_dict=inputs),
        "numexpr reused": lambda: ne.evaluate(expression, local_dict=inputs, out=output),
        "evaluate fresh": lambda: blosc2.evaluate(expression, local_dict=inputs),
        "evaluate reused": lambda: blosc2.evaluate(expression, local_dict=inputs, out=output),
    }


@contextmanager
def cold_preparation():
    """Isolate expression/compiler caches, without flushing CPU or OS caches."""
    native = importlib.import_module("blosc2.native_graph")
    lazy = importlib.import_module("blosc2.lazyexpr")
    graph = importlib.import_module("blosc2.expression_graph")
    native._native_plan_cache.clear()
    for function in (
        native._native_first_syntax,
        native._native_text_expression,
        native.compile_plan,
        native.resident_map_kernel,
        lazy._evaluate_recipe_names,
        graph.parse_expression,
        graph.expression_syntax,
    ):
        function.cache_clear()
    lazy._evaluate_recipe_cache.clear()
    for author in MENUDET_AUTHORS.values():
        author._kernels.clear()
    for cache in (ne.necompiler._numexpr_cache, ne.necompiler._names_cache):
        getattr(cache, "c", {}).clear()
    keys = ("ME_DSL_JIT_CACHE_DIR", "ME_DSL_JIT_POS_CACHE")
    previous = {key: os.environ.get(key) for key in keys}
    with tempfile.TemporaryDirectory(prefix="menudet-cold-") as directory:
        os.environ.update(ME_DSL_JIT_CACHE_DIR=directory, ME_DSL_JIT_POS_CACHE="0")
        try:
            with blosc2.jit_options(cache_dir=directory):
                yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def assert_float_arrays_close(result, expected):
    """Check mixed tolerances in bounded batches, retaining useful diagnostics."""
    result, expected = np.asarray(result), np.asarray(expected)
    np.testing.assert_equal(result.shape, expected.shape)
    tolerance = 4 * np.finfo(expected.real.dtype).eps
    max_error = 0.0
    mismatch_count = 0
    examples = []
    # Avoid allocating several full-sized temporaries for the 100M-item rows.
    for start in range(0, expected.size, 65536):
        actual = result.flat[start : start + 65536]
        reference = expected.flat[start : start + 65536]
        with np.errstate(invalid="ignore", over="ignore"):
            close = np.isclose(actual, reference, rtol=tolerance, atol=tolerance, equal_nan=True)
            errors = np.abs(actual - reference)
        # Equal infinities/NaNs have no error; unmatched nonfinite values have
        # unbounded error rather than an uninformative NaN diagnostic.
        errors = np.where(np.isnan(errors), np.where(close, 0.0, np.inf), errors)
        max_error = max(max_error, float(np.max(errors, initial=0.0)))
        indices = np.flatnonzero(~close)
        mismatch_count += indices.size
        for index in indices[: max(0, 3 - len(examples))]:
            position = np.unravel_index(start + int(index), expected.shape)
            position = tuple(int(axis) for axis in position)
            examples.append(f"index={position}: actual={actual[index]!r}, expected={reference[index]!r}")
    if mismatch_count:
        raise AssertionError(
            f"different numerical results: {mismatch_count} mismatches; "
            f"max absolute error={max_error:.17g}; rtol=atol={tolerance:.17g}; "
            + "; ".join(examples)
        )


def benchmark(engines, expected, samples, warmups, rng, failures=None, cold_factory=None):
    """Correctness-check once, warm up, then time interleaved shuffled samples."""
    qualified = {}
    for name, operation in engines.items():
        try:
            result = operation()
            if isinstance(result, blosc2.NDArray):
                result = result[:]
            if np.ndim(expected) == 0:
                if np.ndim(result) != 0:
                    raise AssertionError("Reduction must return a scalar")
                # Native/parallel summation can group floating additions differently.
                np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)
            elif np.asarray(expected).dtype.kind in "fc":
                assert_float_arrays_close(result, expected)
            else:
                np.testing.assert_array_equal(result, expected)
            del result
        except Exception as error:
            if failures is None:
                raise
            message = str(error).strip().splitlines()
            failures[name] = (
                message[0] if message and message[0].startswith("different numerical results:")
                else "different numerical results"
                if isinstance(error, AssertionError)
                else f"{type(error).__name__}: {message[0] if message else 'unsupported execution'}"
            )
        else:
            qualified[name] = operation
    engines = qualified
    for _ in range(warmups):
        for operation in engines.values():
            result = operation()
            del result
    times = {name: [] for name in engines}
    for _ in range(samples):
        order = list(engines)
        rng.shuffle(order)
        for name in order:
            if cold_factory is not None:
                with cold_preparation():
                    start = perf_counter()
                    result = cold_factory(name)()
                    elapsed = (perf_counter() - start) * 1000
            else:
                start = perf_counter()
                result = engines[name]()
                elapsed = (perf_counter() - start) * 1000
            times[name].append(elapsed)
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
    winner = min(times.values(), default=None)
    cells = [human_items(row["items"]).rjust(item_width), row["operands"].ljust(operand_width)]
    for name in ENGINES:
        if name not in times:
            cells.append("N/A".rjust(engine_width))
            continue
        text = f"{times[name]:.3f}".rjust(engine_width)
        if use_color and times[name] == winner:
            text = f"{GREEN}{text}{RESET}"
        cells.append(text)
    return "  ".join(cells)


def main():  # noqa: C901 -- CLI validation and warm/cold table orchestration
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=int, nargs="+", default=[100_000, 1_000_000, 8_000_000, 100_000_000])
    parser.add_argument(
        "--maxitems",
        "--maxsize",
        type=parse_item_count,
        help="skip sizes exceeding this positive item count (K/M/G suffixes use powers of 1000)",
    )
    parser.add_argument(
        "--expr", default=EXPRESSION, help="expression using x, y and/or z (default: %(default)s)"
    )
    parser.add_argument(
        "--samples", type=int, default=3, help="timed repetitions per row; the median is reported"
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=3,
        help="untimed repetitions; 0 selects preparation-cold samples including compilation",
    )
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=(
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        ),
        help="operand dtype (default: float64)",
    )
    parser.add_argument(
        "--codec",
        type=str.lower,
        choices=tuple(codec.name.lower() for codec in blosc2.Codec),
        default="zstd",
        help="codec for Blosc2 inputs and outputs (default: zstd)",
    )
    parser.add_argument(
        "--clevel", type=int, choices=range(10), default=5, help="compression level (default: 5)"
    )
    parser.add_argument(
        "--jit",
        choices=("tcc", "cc"),
        metavar="BACKEND",
        help="miniexpr JIT compiler backend: tcc or cc (system compiler selected by CC)",
    )
    parser.add_argument("--operands", choices=("both", "numpy", "blosc2"), default="both")
    parser.add_argument(
        "--allocation-control",
        action="store_true",
        help="also compare fresh/reused NumPy outputs for numexpr and evaluate (resident maps only)",
    )
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    parser.add_argument(
        "--reduc",
        choices=("sum", "mean", "min", "max", "any", "all"),
        help="reduce the expression to a scalar",
    )
    args = parser.parse_args()
    try:
        operand_names(args.expr)
    except (SyntaxError, ValueError) as error:
        parser.error(f"invalid --expr: {error}")
    if args.samples <= 0 or args.warmups < 0 or any(size <= 0 for size in args.items):
        parser.error("items and samples must be positive; warmups must be non-negative")
    if args.threads <= 0:
        parser.error("threads must be positive")
    if args.allocation_control and (args.reduc or args.operands == "blosc2"):
        parser.error("--allocation-control requires elementwise NumPy operand rows")
    if args.allocation_control and args.warmups == 0:
        parser.error("--allocation-control requires warm measurements (--warmups > 0)")
    if args.maxitems is not None:
        if args.maxitems <= 0:
            parser.error("maxitems must be positive")
        # Apply the inclusive cap to default sizes and explicit --items alike.
        args.items = [size for size in args.items if size <= args.maxitems]
        if not args.items:
            print("No sizes at or below --maxitems; nothing to benchmark.")
            return

    # Select before LazyExpr preparation or Menudet dtype specialization. Both
    # native compilation paths consume this environment override.
    if args.jit is not None:
        os.environ["ME_DSL_JIT_COMPILER"] = args.jit
        # Enable JIT globally while leaving the Python backend override unset;
        # native-first routes consume ME_DSL_JIT_COMPILER directly.
        os.environ["BLOSC_ME_JIT"] = "true"

    use_color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    blosc2.set_nthreads(args.threads)
    ne.set_num_threads(args.threads)
    kinds = ("numpy", "blosc2") if args.operands == "both" else (args.operands,)
    cparams = blosc2.CParams(codec=blosc2.Codec[args.codec.upper()], clevel=args.clevel)

    expression = f"{args.reduc}({args.expr})" if args.reduc else args.expr
    print(f"Expression: {expression}")
    mode = "Preparation-cold" if args.warmups == 0 else "Warm"
    print(f"{mode} medians in milliseconds (lower is better); fastest figure per row in green.")
    if args.warmups == 0:
        print(
            "Cold samples include engine preparation/compilation; inputs are prebuilt; CPU/OS caches are not flushed."
        )
    print(
        f"blosc2 {blosc2.__version__} | numpy {np.__version__} | numexpr {ne.__version__}"
        f" | dtype {args.dtype} | threads {args.threads} | samples {args.samples}"
    )
    print(f"Blosc2 compression: codec {args.codec} | clevel {args.clevel}")
    print(f"Miniexpr JIT backend request: {os.environ.get('ME_DSL_JIT_COMPILER', 'runtime default')}")
    print()
    widths = column_widths(args.items)
    print(table_header(widths))
    print(table_separator(widths), flush=True)

    rng = random.Random(42)
    previous = None
    allocation_rows = []
    failure_rows = []
    for size in args.items:
        for kind in kinds:
            inputs, expected = build_operands(size, kind, args.dtype, cparams, expression=args.expr)
            if args.reduc:
                expected = getattr(np, args.reduc)(expected)
            engines = make_engines(inputs, kind, args.reduc, cparams=cparams, expression=args.expr)
            failures = {}

            def cold_factory(name, inputs=inputs, kind=kind):
                return make_engines(
                    inputs, kind, args.reduc, cparams=cparams, selected=name, expression=args.expr
                )[name]

            times = benchmark(
                engines,
                expected,
                args.samples,
                args.warmups,
                rng,
                failures=failures,
                cold_factory=cold_factory if args.warmups == 0 else None,
            )
            failure_rows.extend((size, kind, name, reason) for name, reason in failures.items())
            if args.allocation_control and kind == "numpy":
                controls = make_allocation_engines(inputs, expression=args.expr)
                control_times = benchmark(controls, expected, args.samples, args.warmups, rng, failures={})
                allocation_rows.append((size, tuple(controls), control_times))
                del controls
            del inputs, engines, expected
            if previous is not None and size != previous:
                print(table_separator(widths))
            row = {"items": size, "operands": kind, "times": times}
            print(result_row(row, widths, use_color), flush=True)
            previous = size

    print()
    if failure_rows:
        print("Qualification failures:")
        for size, kind, name, reason in failure_rows:
            print(f"  {human_items(size)} {kind} {name}: {reason}")
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
            f"LazyExpr returns Blosc2 output ({args.codec}, clevel={args.clevel}); "
            "Menudet does so for Blosc2 inputs only."
        )
    if args.warmups:
        print("evaluate caches eligible preparation; input rebinding and fresh output allocation are timed.")
    else:
        print(
            "Each cold sample rebuilds its engine with empty preparation/compiler caches and fresh output."
        )
    if allocation_rows:
        print("\nAllocation control: resident inputs, NumPy outputs; warm medians in ms.")
        print("Reused destinations are populated before timing; the main table above remains fresh-output.")
        for size, names, times in allocation_rows:
            figures = "; ".join(
                f"{name} {times[name]:.3f}" if name in times else f"{name} N/A" for name in names
            )
            print(f"{human_items(size)}: {figures}")


if __name__ == "__main__":
    main()
