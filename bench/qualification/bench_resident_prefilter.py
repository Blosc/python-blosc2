"""Same-artifact, interleaved Python-scheduler versus resident-prefilter probe.

Geometry sweeps are diagnostic only. Preparation/context construction is reported
separately; the warm table never includes compilation or input generation.
"""

import argparse
import hashlib
import json
import random
import runpy
import statistics
import tempfile
import time
from pathlib import Path

import numexpr as ne
import numpy as np

import blosc2
from blosc2.native_graph import prepare_resident
from blosc2.resident_execution import evaluate_resident


def timed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, time.perf_counter_ns() - start


def compare(
    inputs, expected, schedule, workers, block_items, chunk_items, explicit, samples, seed, expression
):
    context, setup_ns = timed(
        lambda: schedule.kernel._handle.resident_prefilter(
            expected.shape, workers, block_items, chunk_items, graph=schedule.plan._handle
        )
    )
    outputs = {
        name: np.empty_like(expected) if explicit else None for name in ("python", "prefilter", "numexpr")
    }
    for out in outputs.values():
        if out is not None:
            out.fill(0)

    def python():
        return evaluate_resident(schedule.kernel, inputs, expected.shape, workers, out=outputs["python"])

    def prefilter():
        return context.evaluate(inputs, out=outputs["prefilter"])

    def numexpr():
        return ne.evaluate(expression, local_dict=inputs, out=outputs["numexpr"]), {"route": "numexpr"}

    operations = {"python": python, "prefilter": prefilter, "numexpr": numexpr}
    first = {}
    for name, operation in operations.items():
        (result, report), elapsed = timed(operation)
        np.testing.assert_array_equal(result, expected)
        if explicit:
            assert result is outputs[name]
        first[name] = {"ns": elapsed, "report": report}
        del result
        for _ in range(3):
            result, report = operation()
            del result
    times = {name: [] for name in operations}
    reports = {}
    rng = random.Random(seed)
    for _ in range(samples):
        order = list(operations)
        rng.shuffle(order)
        for name in order:
            value, elapsed = timed(operations[name])
            result, report = value
            times[name].append(elapsed)
            reports[name] = report
            del result, value
    for operation in operations.values():
        result, _ = operation()
        np.testing.assert_array_equal(result, expected)
        del result
    return {
        "block_items": block_items,
        "chunk_items": chunk_items,
        "explicit_output": explicit,
        "context_setup_ns": setup_ns,
        "first_execution": first,
        "samples_ns": times,
        "medians_ms": {name: statistics.median(values) / 1e6 for name, values in times.items()},
        "reports": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[1000000, 8000000])
    parser.add_argument("--block-items", type=int, nargs="+", default=[8192, 16384, 32768, 65536, 131072])
    parser.add_argument("--chunk-items", type=int, nargs="+", default=[0])
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--expr", default="(x**2 - 1)")
    parser.add_argument("--jit", choices=["auto", "off", "tcc", "cc"], default="auto")
    parser.add_argument("--out", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.threads < 1 or any(n < 1 for n in args.items):
        parser.error("Counts must be positive")
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1] / "menudet-engines.py"))
    blosc2.set_nthreads(args.threads)
    ne.set_num_threads(args.threads)
    rows = []
    result = {
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "extension_sha256": hashlib.sha256(Path(blosc2.blosc2_ext.__file__).read_bytes()).hexdigest(),
        "expression": args.expr,
        "requested_jit": args.jit,
        "threads": args.threads,
        "rows": rows,
        "limitations": [
            "Local geometry controls, not independent calibrated evidence.",
            "Python and prefilter arms execute the same artifact; NumExpr is contextual.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for size in args.items:
        inputs, expected = benchmark["build_operands"](size, "numpy", "float64", expression=args.expr)
        with tempfile.TemporaryDirectory(prefix="resident-prefilter-bench-") as cache:
            with blosc2.jit_options(
                jit=None if args.jit == "auto" else args.jit != "off",
                jit_backend=args.jit if args.jit in ("cc", "tcc") else None,
                cache_dir=cache,
            ):
                prepared, preparation = timed(
                    lambda inputs=inputs: prepare_resident(args.expr, inputs, optimization="aggressive")
                )
                if prepared is None:
                    raise RuntimeError("Resident admission failed")
                schedule = prepared[2]
                if args.jit in ("cc", "tcc") and not schedule.kernel.has_jit:
                    raise RuntimeError("Required JIT capability unavailable")
                for block in args.block_items:
                    for chunk in args.chunk_items:
                        row = compare(
                            inputs,
                            expected,
                            schedule,
                            args.threads,
                            block,
                            chunk,
                            args.out,
                            args.samples,
                            42,
                            args.expr,
                        )
                        row.update(size=size, preparation_ns=preparation, has_jit=schedule.kernel.has_jit)
                        rows.append(row)
                        med = row["medians_ms"]
                        print(
                            f"{size:>9} block={block:>6} chunk={chunk:>9} "
                            f"Python={med['python']:.3f} prefilter={med['prefilter']:.3f} "
                            f"NumExpr={med['numexpr']:.3f} ms",
                            flush=True,
                        )
                        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
