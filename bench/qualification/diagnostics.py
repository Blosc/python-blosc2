"""The sole adapter for private execution reports; never used inside warm timers."""

from contextlib import contextmanager


def normalize(raw, kernel=None, default_route=None):
    info = getattr(kernel, "_info", {})
    engine = raw.get("backend")
    if kernel is not None:
        engine = "menudet-jit" if kernel.has_jit else "menudet-interpreter"
    elif engine is None and "jit_stages" in raw:
        engine = "menudet-jit" if raw["jit_stages"] else "menudet-interpreter"
    return {
        "route": raw.get("route", default_route),
        "engine": engine,
        "effective_compiler": None,
        "effective_math_backend": info.get("math_backend"),
        "effective_fp_mode": info.get("fp_mode"),
        "effective_sleef_accuracy": info.get("sleef_accuracy"),
        "tasks": raw.get("tasks"),
        "workers": raw.get("threads"),
        "normalization_bytes": raw.get("normalization_bytes"),
        "gather_bytes": raw.get("gathered_bytes"),
        "report_scope": "whole-operation"
        if raw.get("route", default_route) in ("menudet-prefilter", "authored-flat")
        else "unavailable",
        "raw": raw,
    }


def owner_report(owner):
    if owner is None:
        return None
    raw = getattr(owner, "_source_execution_report", None) or getattr(
        owner, "_native_execution_report", None
    )
    kernel = getattr(owner, "kernel", None)
    if raw is not None:
        return normalize(raw, kernel)
    if hasattr(owner, "has_jit"):
        return normalize({}, owner, "authored-flat")
    return None


@contextmanager
def resident_reports():
    import blosc2
    import blosc2.native_graph as native

    original = native.execute_resident
    original_kernel = native.configured_map_kernel
    kernels = []
    original_compute = blosc2.LazyExpr._compute_expr
    reports = []

    def capture(prepared, inputs, *, out=None):
        result, raw = original(prepared, inputs, out=out)
        kernel = getattr(prepared[2], "kernel", None)
        reports.append(
            {
                **normalize(raw, kernel),
                "route": "resident.configured_map" if kernel is not None else "resident.native_graph",
                "engine": "menudet-jit" if raw.get("jit_stages") else "menudet-interpreter",
                # The runtime exposes JIT activation, but not the selected compiler.
                # Do not turn a requested compiler into an observed compiler fact.
                "effective_compiler": None,
                "compiler_directive_source": kernel.source if kernel is not None else None,
                "tasks": raw.get("tasks"),
                "normalization_bytes": raw.get("normalization_bytes"),
                "gather_bytes": None,
                "report_scope": "whole-operation",
                "partitionable": prepared[1].partitionable,
                "proof_authority": "native-graph",
                "raw": raw,
            }
        )
        return result, raw

    native.execute_resident = capture

    def capture_kernel(*args, **kwargs):
        kernel = original_kernel(*args, **kwargs)
        kernels.append(kernel)
        return kernel

    native.configured_map_kernel = capture_kernel

    def capture_compute(expr, *args, **kwargs):
        before = len(reports)
        result = original_compute(expr, *args, **kwargs)
        if len(reports) == before:
            raw = getattr(expr, "_native_execution_report", {})
            reports.append(
                normalize(raw, kernels[-1] if kernels else None, default_route="fallback.general_chunk")
            )
        return result

    blosc2.LazyExpr._compute_expr = capture_compute
    try:
        yield reports
    finally:
        native.execute_resident = original
        native.configured_map_kernel = original_kernel
        blosc2.LazyExpr._compute_expr = original_compute


def validate_contract(case, reports, size, threads):
    if len(reports) != 1:
        raise AssertionError("Expected one resident execution; fallback or missing report")
    report = reports[0]
    observed_math = report.get("effective_math_backend")
    if observed_math is not None and observed_math != case.math_backend:
        raise AssertionError("Effective math backend differs from requested numerical policy")
    if case.route == "fallback":
        if report["route"] and report["route"].startswith("resident."):
            raise AssertionError("Conservative fallback unexpectedly entered resident route")
        return report
    if case.route != "resident":
        if report["route"] != case.route:
            raise AssertionError(f"Expected {case.route}, observed {report['route']}")
        expected_engine = "menudet-jit" if case.jit else "menudet-interpreter"
        if report["engine"] != expected_engine:
            raise AssertionError("Required JIT/interpreter execution capability unavailable")
        if case.route == "menudet-prefilter" and not report["raw"].get("evaluated_blocks"):
            raise AssertionError("Prefilter did not evaluate physical storage blocks")
        if case.route == "menudet-prefilter" and threads > 1 and (report["workers"] or 0) < 2:
            raise AssertionError("Prefilter lost requested parallel workers")
        return report
    if case.backend is not None and report["route"] != "resident.configured_map":
        raise AssertionError("Explicit backend lost configured resident route")
    if report["engine"] != ("menudet-jit" if case.jit else "menudet-interpreter"):
        raise AssertionError("Required JIT capability unavailable or interpreter fallback")
    if report["normalization_bytes"] != 0:
        raise AssertionError("Resident map unexpectedly normalized inputs")
    # Require parallel coverage without hard-coding an incidental tile count.
    if size >= 262144 and threads > 1 and (report["tasks"] or 0) < 2:
        raise AssertionError("Expected parallel resident execution")
    return report
