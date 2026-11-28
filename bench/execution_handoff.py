"""Release handoff gate; run separately with each isolated package.

Example (always use the blosc2 conda environment)::

    python -S bench/execution_handoff.py --package /path/to/isolated/package

Preparation and specialization are measured independently of warmed public
execution. Output is JSON lines for external comparison; keep raw runs outside
the repository. These are real-hardware measurements, not a QEMU gate.
"""

import argparse
import json
import statistics
import sys
import sysconfig
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--package", type=Path, required=True)
parser.add_argument("--items", type=int, default=16_000_019)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--repeat", type=int, default=15)
parser.add_argument(
    "--tile-items", type=int, default=1024, help="Ordinary default or explicit native tile size"
)
args = parser.parse_args()
sys.path[:0] = [str(args.package), sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]

import numpy as np  # noqa: E402

import blosc2  # noqa: E402
from blosc2.graph_plan import NativeGraph  # noqa: E402

assert Path(blosc2.__file__).is_relative_to(args.package)
assert Path(blosc2.blosc2_ext.__file__).is_relative_to(args.package)
blosc2.set_nthreads(args.workers)


def measure(operation):
    samples = []
    for _ in range(args.repeat):
        start = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples)


x = np.arange(args.items, dtype=np.float64)
bx = blosc2.asarray(x, chunks=(262144,), blocks=(4096,))
for compressed, operand in ((False, x), (True, bx)):
    for expression in ("x + 1", "min(x + 1)"):
        plan = NativeGraph.from_expression(expression, {"x": "float64"}, jit=True)
        preparation = measure(
            lambda expression=expression: NativeGraph.from_expression(expression, {"x": "float64"}, jit=True)
        )
        metadata = {"x": (operand.dtype, operand.shape)}
        specialization = measure(lambda plan=plan, metadata=metadata: plan.specialize(metadata, 4096))
        expr = blosc2.lazyexpr(expression, {"x": operand})

        def execute(expr=expr):
            return expr.compute(jit=True, _getitem=True, _native_tile_items=args.tile_items)

        actual = execute()
        np.testing.assert_array_equal(actual, x + 1 if expression == "x + 1" else 1.0)
        warmed = measure(execute)
        report = expr._native_execution_report
        print(
            json.dumps(
                {
                    "storage": "compressed" if compressed else "resident",
                    "expression": expression,
                    "items": args.items,
                    "requested_workers": args.workers,
                    "tasks": report.get("tasks", report.get("threads")),
                    "route": report.get("route"),
                    "tile_items": report.get("tile_items"),
                    "preparation_ms": preparation,
                    "specialization_ms": specialization,
                    "warmed_ms": warmed,
                }
            ),
            flush=True,
        )
