"""Interleaved, route-aware legacy/typed fallback qualification on a host runtime.

Timing excludes preparation and validation. Allocation probes run separately and
report tracemalloc-visible peak bytes, not total native scratch or process RSS.
Do not interpret this probe as complete native-memory or Pyodide qualification.
"""

import argparse
import contextlib
import importlib
import json
import os
import statistics
import tempfile
import time
import tracemalloc
from unittest.mock import patch

import numpy as np

import blosc2


@contextlib.contextmanager
def route_probe():
    """Observe numerical dispatch without changing admission or replay behavior."""
    lazy = importlib.import_module("blosc2.lazyexpr")
    calls = {"chunk_evaluations": 0, "numpy_evaluations": 0, "uniform_prefilters": 0, "typed_prefilters": 0}
    reports = []
    original_chunk = lazy.evaluate_chunk
    original_uniform = blosc2.NDArray._set_pref_expr
    original_typed = blosc2.NDArray._execute_menudet_prefilter
    original_numpy = lazy._numpy_eval_datetime_aware

    def chunk(*args, **kwargs):
        calls["chunk_evaluations"] += 1
        return original_chunk(*args, **kwargs)

    def uniform(*args, **kwargs):
        calls["uniform_prefilters"] += 1
        return original_uniform(*args, **kwargs)

    def numpy_eval(*args, **kwargs):
        calls["numpy_evaluations"] += 1
        return original_numpy(*args, **kwargs)

    def typed(*args, **kwargs):
        calls["typed_prefilters"] += 1
        result = original_typed(*args, **kwargs)
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
            reports.append(result[1])
        return result

    with (
        patch.object(lazy, "evaluate_chunk", chunk),
        patch.object(lazy, "_numpy_eval_datetime_aware", numpy_eval),
        patch.object(blosc2.NDArray, "_set_pref_expr", uniform),
        patch.object(blosc2.NDArray, "_execute_menudet_prefilter", typed),
    ):
        yield calls, reports


def validate(result, expected):
    actual = result[:] if isinstance(result, blosc2.NDArray) else np.asarray(result)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    if expected.dtype.kind in "fc":
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    else:
        np.testing.assert_array_equal(actual, expected)


def workloads(side, directory, requested_chunks=(31, 47), requested_blocks=(7, 11)):
    rng = np.random.default_rng(1)
    values = rng.random((side, side), dtype=np.float32)
    # Interior padding rejects flat native geometry. Use distinct stored handles
    # for the persisted case; all inputs are built before timing/allocation probes.
    chunks = tuple(min(size, side) for size in requested_chunks)
    blocks = tuple(min(size, chunk) for size, chunk in zip(requested_blocks, chunks, strict=True))
    a = blosc2.asarray(values, chunks=chunks, blocks=blocks)
    disk = blosc2.asarray(values, chunks=chunks, blocks=blocks, urlpath=f"{directory}/a.b2nd")
    row = np.arange(side, dtype=np.float32)[None, :]
    complex_values = values.astype(np.complex64) + 1j
    complex_a = blosc2.asarray(complex_values, chunks=chunks, blocks=blocks)
    half_values = values.astype(np.float16)
    half = blosc2.asarray(half_values, chunks=chunks, blocks=blocks)
    flat = blosc2.asarray(values.ravel(), chunks=(min(97, values.size),), blocks=(11,))
    mapped = values * 2 + 1
    return {
        "nonflat_compute": (lambda: a * 2 + 1, lambda e: e.compute(), mapped),
        "persisted_compute": (lambda: disk * 2 + 1, lambda e: e.compute(), mapped),
        "broadcast": (
            lambda: blosc2.lazyexpr("a + row", {"a": a, "row": row}),
            lambda e: e[:],
            values + row,
        ),
        "partial_slice": (lambda: a * 2 + 1, lambda e: e[1:-1:2, ::3], mapped[1:-1:2, ::3]),
        "strided_out": (
            lambda: a * 2 + 1,
            lambda e: e.compute(out=np.empty((side, side * 2), dtype=np.float32)[:, ::2]),
            mapped,
        ),
        "scalar_sum": (lambda: a * 2 + 1, lambda e: e.sum(), mapped.sum()),
        "axis_sum": (lambda: a * 2 + 1, lambda e: e.sum(axis=0), mapped.sum(axis=0)),
        "variance": (lambda: a * 2 + 1, lambda e: e.var(), mapped.var()),
        "selection": (lambda: flat[flat > 0.5], lambda e: e[:], values.ravel()[values.ravel() > 0.5]),
        "complex": (lambda: complex_a * complex_a + 1, lambda e: e[:], complex_values**2 + 1),
        "float16": (lambda: half + half, lambda e: e[:], half_values + half_values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--threads", type=int, default=blosc2.nthreads)
    parser.add_argument("--chunks", type=int, nargs=2, default=(31, 47))
    parser.add_argument("--blocks", type=int, nargs=2, default=(7, 11))
    parser.add_argument("--case", action="append", help="Select a workload; repeat for several")
    args = parser.parse_args()
    if args.side < 4 or args.rounds < 1 or args.threads < 1:
        parser.error("side must be >= 4; rounds and threads must be positive")
    if any(size < 1 for size in (*args.chunks, *args.blocks)):
        parser.error("chunks and blocks must be positive")
    blosc2.set_nthreads(args.threads)
    previous = os.environ.get("BLOSC2_EXPRESSION_EVALUATOR")
    rows = []
    try:
        with tempfile.TemporaryDirectory(prefix="blosc2-evaluator-") as directory:
            cases = workloads(args.side, directory, args.chunks, args.blocks)
            if args.case and set(args.case) - cases.keys():
                parser.error(f"Unknown cases: {sorted(set(args.case) - cases.keys())}")
            for name, (prepare, execute, expected) in cases.items():
                if args.case and name not in args.case:
                    continue
                expressions, timings, diagnostics = {}, {}, {}
                for mode in ("legacy", "typed"):
                    os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = mode
                    start = time.perf_counter()
                    expressions[mode] = prepare()
                    prepare_ms = 1000 * (time.perf_counter() - start)
                    with route_probe() as (calls, reports):
                        result = execute(expressions[mode])
                    validate(result, expected)
                    del result
                    diagnostics[mode] = {
                        "prepare_ms": prepare_ms,
                        "dispatch_calls": calls,
                        "native_report": getattr(expressions[mode], "_native_execution_report", None),
                        "prefilter_reports": reports,
                    }
                    timings[mode] = []
                for round_ in range(args.rounds):
                    order = ("legacy", "typed") if round_ % 2 == 0 else ("typed", "legacy")
                    for mode in order:
                        os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = mode
                        start = time.perf_counter()
                        result = execute(expressions[mode])
                        timings[mode].append(1000 * (time.perf_counter() - start))
                        del result
                for mode in timings:
                    os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = mode
                    tracemalloc.start()
                    try:
                        result = execute(expressions[mode])
                        _, peak = tracemalloc.get_traced_memory()
                    finally:
                        tracemalloc.stop()
                    validate(result, expected)
                    del result
                    diagnostics[mode].update(
                        median_ms=statistics.median(timings[mode]),
                        samples_ms=timings[mode],
                        tracked_peak_bytes=peak,
                    )
                rows.append({"case": name, "correct": True, "modes": diagnostics})
    finally:
        if previous is None:
            os.environ.pop("BLOSC2_EXPRESSION_EVALUATOR", None)
        else:
            os.environ["BLOSC2_EXPRESSION_EVALUATOR"] = previous
    print(
        json.dumps(
            {
                "numpy": np.__version__,
                "package": blosc2.__file__,
                "extension": blosc2.blosc2_ext.__file__,
                "threads": blosc2.nthreads,
                "side": args.side,
                "rounds": args.rounds,
                "requested_chunks": args.chunks,
                "requested_blocks": args.blocks,
                "memory_scope": "tracemalloc-visible allocations; excludes untracked native scratch; not RSS",
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
