"""Explicit typed-plan/adapter comparison; no automatic dispatch selectors.

Times are interleaved microsecond medians. Typed frontend lowering is measured
separately. Adapter preparation includes its native text frontend; that unlike
scope is labelled. This is not the cross-platform performance gate.
"""

import argparse
import statistics
import time

import numpy as np

import blosc2
from blosc2.graph_plan import NativeGraph
from blosc2.typed_exec import lower_typed_graph, prepare_typed_plan


def medians(first, second, repeat):
    first()
    second()
    times = [[], []]
    for run in range(repeat):
        for index in (run % 2, 1 - run % 2):
            function = (first, second)[index]
            start = time.perf_counter_ns()
            result = function()
            times[index].append((time.perf_counter_ns() - start) / 1000)
            del result
    return [statistics.median(samples) for samples in times]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, default=1000003)
    parser.add_argument("--tile", type=int, default=1024)
    parser.add_argument("--repeat", type=int, default=31)
    parser.add_argument("--map", action="store_true")
    args = parser.parse_args()
    if min(args.items, args.tile, args.repeat) <= 0:
        parser.error("items, tile and repeat must be positive")
    if not getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)():
        parser.error("requires the C-Blosc2 typed executor")
    expression = "x + y * 3" if args.map else "min(x + y * 3)"
    rng = np.random.default_rng(42)
    inputs = {name: rng.integers(-(2**60), 2**60, size=args.items) for name in ("x", "y")}
    metadata = {name: (value.dtype, value.shape) for name, value in inputs.items()}
    nodes, names, root, _ = lower_typed_graph(expression, inputs)
    arrays = tuple(inputs[name] for name in names)
    typed_metadata = tuple(metadata[name] for name in names)

    def prepare_typed():
        return prepare_typed_plan.__wrapped__(nodes, len(names), root)

    def prepare_adapter():
        return NativeGraph.from_expression(
            expression, dict.fromkeys(names, "int64"), jit=False, _executor="adapter"
        )

    typed = prepare_typed()
    adapter = prepare_adapter()
    native = typed.specialize(typed_metadata, args.tile)
    baseline = adapter.specialize(metadata, args.tile)
    actual, report = native.execute(arrays)
    expected, reference = baseline.execute(inputs)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()
    assert report["fp_flags"] == reference["fp_flags"]
    a, _ = medians(lambda: lower_typed_graph(expression, inputs), lambda: None, args.repeat)
    print(f"{'phase':<24} {'typed us':>12} {'adapter us':>12}")
    print(f"{'frontend':<24} {a:12.2f} {'in prepare':>12}")
    for label, first, second in (
        ("prepare/prep+frontend", prepare_typed, prepare_adapter),
        (
            "specialize",
            lambda: typed.specialize(typed_metadata, args.tile),
            lambda: adapter.specialize(metadata, args.tile),
        ),
        ("execute", lambda: native.execute(arrays), lambda: baseline.execute(inputs)),
    ):
        a, b = medians(first, second, args.repeat)
        print(f"{label:<24} {a:12.2f} {b:12.2f}")


if __name__ == "__main__":
    main()
