"""Diagnostic probe for cold/warm resident execution; never a regression gate.

Reuses menudet-engines.py's inputs, selected-engine factories and cold contract.
Instrumented stage/worker traces are separate from uninstrumented observations.
"""

import argparse
import contextlib
import cProfile
import functools
import importlib
import io
import json
import math
import os
import platform
import pstats
import resource
import runpy
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numexpr as ne
import numpy as np

import blosc2

EXPRESSION = "(x**2 - 1)"
HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def configured_scheduler(kind):
    """Same-artifact old-scheduler control, never a production dispatch switch."""
    native = importlib.import_module("blosc2.native_graph")
    original = native._ConfiguredResidentSchedule.execute

    def python_execute(schedule, inputs, *, out=None, workers=1):
        resident = importlib.import_module("blosc2.resident_execution")
        shape = schedule.info()["shape"]
        if workers > 1:
            if not schedule.partitionable:
                raise ValueError("Graph rejected partitioned execution")
            result, report = resident.evaluate_resident(schedule.kernel, inputs, shape, workers, out=out)
        else:
            result, status = schedule.kernel._handle.evaluate_block(
                inputs, shape, return_status=True, _output=out
            )
            report = {"fp_flags": status["flags"], "fp_supported": status["supported"]}
        return result, {
            **report,
            "tasks": workers,
            "stages": 1,
            "jit_stages": int(schedule.kernel.has_jit),
            "interpreter_stages": int(not schedule.kernel.has_jit),
            "normalization_bytes": 0,
        }

    if kind == "python":
        native._ConfiguredResidentSchedule.execute = python_execute
    try:
        yield
    finally:
        native._ConfiguredResidentSchedule.execute = original


@contextlib.contextmanager
def stage_trace(events):
    """Inclusive spans; worker time sums overlap and are not main-thread wall."""
    native = importlib.import_module("blosc2.native_graph")
    resident = importlib.import_module("blosc2.resident_execution")
    patches = []

    def patch(owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
            finally:
                events.append({"stage": label, "ns": time.perf_counter_ns() - started})
            if name == "execute_resident":
                prepared = args[0]
                events[-1]["report"] = result[1]
                kernel = getattr(prepared[2], "kernel", None)
                events[-1]["configured"] = kernel is not None
                events[-1]["kernel_source"] = kernel.source if kernel is not None else None
                events[-1]["has_jit"] = kernel.has_jit if kernel is not None else prepared[1].has_jit
                events[-1]["semantics"] = json.loads(prepared[1].map_json()).get("semantics")
                events[-1]["kernel_policy"] = (
                    {key: kernel._info.get(key) for key in ("math_backend", "fp_mode", "execution_kind")}
                    if kernel is not None
                    else None
                )
            return result

        patches.append((owner, name, original))
        setattr(owner, name, wrapped)

    for name in ("prepare_resident", "compile_plan", "configured_map_kernel", "execute_resident"):
        patch(native, name, name)
    patch(np, "empty", "numpy.empty")
    original_dynamic = resident.run_dynamic

    def dynamic(jobs, workers, execute):
        events.append({"stage": "schedule", "jobs": len(jobs), "workers": workers})

        def tile(job):
            started, cpu_started = time.perf_counter_ns(), time.thread_time_ns()
            try:
                return execute(job)
            finally:
                events.append(
                    {
                        "stage": "worker-tile",
                        "thread": threading.get_ident(),
                        "ns": time.perf_counter_ns() - started,
                        "cpu_ns": time.thread_time_ns() - cpu_started,
                    }
                )

        return original_dynamic(jobs, workers, tile)

    resident.run_dynamic = dynamic
    try:
        yield
    finally:
        resident.run_dynamic = original_dynamic
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def observe(call, expected):
    before = resource.getrusage(resource.RUSAGE_SELF)
    start, cpu_start = time.perf_counter_ns(), time.process_time_ns()
    result = call()
    elapsed, cpu = time.perf_counter_ns() - start, time.process_time_ns() - cpu_start
    after = resource.getrusage(resource.RUSAGE_SELF)
    np.testing.assert_array_equal(result, expected)
    del result
    return {
        "ns": elapsed,
        "cpu_ns": cpu,
        "minor_faults": after.ru_minflt - before.ru_minflt,
        "major_faults": after.ru_majflt - before.ru_majflt,
    }


def summarize(samples):
    return {
        "median_ms": statistics.median(s["ns"] for s in samples) / 1e6,
        "median_cpu_ms": statistics.median(s["cpu_ns"] for s in samples) / 1e6,
        "median_minor_faults": statistics.median(s["minor_faults"] for s in samples),
        "samples": samples,
    }


def prepared_call(inputs, expected, threads, mode, expression):
    native = importlib.import_module("blosc2.native_graph")
    prepared = native.prepare_resident(expression, inputs, optimization="aggressive")
    if prepared is None:
        raise RuntimeError("Resident preparation rejected")
    out = np.empty_like(expected) if mode.endswith("reused") else None
    if out is not None:
        out.fill(0)
    if mode.startswith("prepared-coarse"):
        kernel = prepared[2].kernel
        size = expected.size
        tile = math.ceil(size / threads)
        resident = importlib.import_module("blosc2.resident_execution")

        def call():
            output = np.empty_like(expected) if out is None else out

            def execute(start):
                stop = min(start + tile, size)
                kernel._handle.evaluate_block(
                    {"x": inputs["x"][start:stop]},
                    (stop - start,),
                    _output=output[start:stop],
                    return_status=True,
                )

            resident.run_dynamic(range(0, size, tile), threads, execute)
            return output
    else:

        def call():
            return native.execute_resident(prepared, inputs, out=out)[0]

    return call


def run_case(benchmark, size, threads, mode, samples, expression=EXPRESSION):
    inputs, expected = benchmark["build_operands"](size, "numpy", "float64", expression=expression)
    if set(inputs) != {"x"}:
        raise ValueError("This diagnostic probe requires a single input named x")
    blosc2.set_nthreads(threads)
    ne.set_num_threads(threads)
    engine = "numexpr" if mode.startswith("numexpr") else "evaluate"

    def factory():
        return benchmark["make_engines"](inputs, "numpy", selected=engine, expression=expression)[engine]

    @contextlib.contextmanager
    def cold():
        with benchmark["cold_preparation"]():
            with blosc2.jit_options(cache_dir=None) if mode == "cold-env" else contextlib.nullcontext():
                yield

    records, events = [], []
    if mode in ("cold-configured", "cold-env", "numexpr-cold"):
        for _ in range(samples):
            with cold():
                records.append(observe(lambda: factory()(), expected))
        if engine == "evaluate":
            with cold(), stage_trace(events):
                observe(lambda: factory()(), expected)
            with cold():
                profiler = cProfile.Profile()
                profiler.runcall(lambda: factory()())
                stream = io.StringIO()
                pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
                profile = stream.getvalue()
        else:
            profile = None
    else:
        with tempfile.TemporaryDirectory(prefix="evaluate-profile-") as directory:
            configured = mode.startswith("warm-configured") or mode.startswith("prepared")
            with blosc2.jit_options(cache_dir=directory if configured else None):
                call = factory()
                if mode.startswith("prepared"):
                    call = prepared_call(inputs, expected, threads, mode, expression)

                for _ in range(3):
                    result = call()
                    del result
                records = [observe(call, expected) for _ in range(samples)]
                if engine == "evaluate":
                    with stage_trace(events):
                        observe(call, expected)
        profile = None
    return {
        "size": size,
        "threads": threads,
        "mode": mode,
        **summarize(records),
        "instrumented_events": events,
        "cprofile_main_thread": profile,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, nargs="+", default=[1000000, 8000000])
    parser.add_argument("--expr", default=EXPRESSION)
    parser.add_argument("--configured-scheduler", choices=["current", "python"], default="current")
    parser.add_argument("--threads", type=int, nargs="+", default=[12])
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "numexpr-cold",
            "cold-configured",
            "cold-env",
            "numexpr-warm",
            "warm-configured",
            "warm-auto",
            "prepared-fresh",
            "prepared-reused",
            "prepared-coarse-fresh",
            "prepared-coarse-reused",
        ],
        default=[
            "numexpr-cold",
            "cold-configured",
            "cold-env",
            "numexpr-warm",
            "warm-configured",
            "warm-auto",
            "prepared-fresh",
            "prepared-reused",
            "prepared-coarse-fresh",
            "prepared-coarse-reused",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or any(n < 1 for n in args.items + args.threads):
        parser.error("items, threads and samples must be positive")
    benchmark = runpy.run_path(str(HERE.parents[0] / "menudet-engines.py"))
    result = {
        "platform": platform.platform(),
        "python": os.sys.executable,
        "package": blosc2.__file__,
        "extension": blosc2.blosc2_ext.__file__,
        "expression": args.expr,
        "configured_scheduler": args.configured_scheduler,
        "requested_policy": {
            k: v.name if hasattr(v, "name") else v for k, v in blosc2.get_jit_options().items()
        },
        "rows": [],
        "limitations": [
            "Same-process observations, not independent statistical evidence.",
            "Instrumented stage/worker spans are separate diagnostic invocations.",
            "Coarse scheduling is a benchmark-only control, not a production change.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with configured_scheduler(args.configured_scheduler):
        for size in args.items:
            for threads in args.threads:
                for mode in args.modes:
                    row = run_case(benchmark, size, threads, mode, args.samples, args.expr)
                    result["rows"].append(row)
                    print(
                        f"{size:>8} {threads:>2} {mode:>24}: {row['median_ms']:.3f} ms; "
                        f"CPU {row['median_cpu_ms']:.3f} ms; faults {row['median_minor_faults']}",
                        flush=True,
                    )
                    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
