"""Public evaluate dispatch/output contracts, not native numerical cross-products."""

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph


@pytest.fixture(autouse=True)
def execution_threads(native_graph_runtime):
    if blosc2.IS_WASM:
        pytest.skip("These automatic map routes require host JIT/thread/prefilter support")
    probe = NativeGraph.from_expression("x + 1", {"x": "float64"}, jit=True)
    if not probe.has_jit:
        pytest.skip("Automatic native map dispatch requires an available graph JIT")
    # Compressed-input NumPy maps need at least two workers. Do not inherit a
    # single-thread setting left by another test in the same xdist worker.
    previous = blosc2.set_nthreads(2)
    try:
        yield
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize("inputs", ["numpy", "blosc2", "mixed"])
@pytest.mark.parametrize("output", ["new", "numpy", "blosc2"])
@pytest.mark.parametrize("recipe", ["integer", "math"])
def test_evaluate_native_array_routes(inputs, output, recipe, native_graph_runtime, monkeypatch):
    import blosc2.native_graph as native

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_MAP_ITEMS", 1)
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    monkeypatch.setattr("blosc2.resident_execution.MINIMUM_WORKER_ITEMS", 64)
    dtype = "int64" if recipe == "integer" else "float64"
    x = np.arange(1027, dtype=dtype) % 3 + 1
    y = np.ones_like(x)
    expression = "((x + y) * 2 - y) * 3 + x * 2 - y" if recipe == "integer" else "sin(x) + cos(y) + 1"
    expected = ((x + y) * 2 - y) * 3 + x * 2 - y if recipe == "integer" else np.sin(x) + np.cos(y) + 1
    operands = {"x": x, "y": y}
    for key in operands:
        if inputs == "blosc2" or (inputs == "mixed" and key == "x"):
            operands[key] = blosc2.asarray(operands[key], chunks=(512,), blocks=(128,))
    destination = None if output == "new" else np.empty_like(x)
    if output == "blosc2":
        destination = blosc2.empty(x.shape, dtype=x.dtype, chunks=(512,), blocks=(128,))

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible evaluate expression fell back to legacy numerical execution")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate(expression, local_dict=operands, out=destination)
    if destination is not None:
        assert result is destination
    np.testing.assert_allclose(
        result[:] if isinstance(result, blosc2.NDArray) else result, expected, rtol=2e-12, atol=2e-12
    )
    # Current operands must be rebound, not retained as cached values.
    operands["x"][:] = 2
    result = blosc2.evaluate(expression, local_dict=operands, out=destination)
    expected = ((2 + y) * 2 - y) * 3 + 4 - y if recipe == "integer" else np.sin(2) + np.cos(y) + 1
    np.testing.assert_allclose(
        result[:] if isinstance(result, blosc2.NDArray) else result, expected, rtol=2e-12, atol=2e-12
    )


def test_evaluate_stored_multidimensional_broadcast(native_graph_runtime, monkeypatch):
    x = np.arange(51, dtype="int64").reshape(3, 17)
    y = np.arange(17, dtype="int64").reshape(1, 17)
    stored = blosc2.asarray(x, chunks=(2, 8), blocks=(1, 4))
    output = blosc2.empty(x.shape, dtype="int64", chunks=(2, 8), blocks=(1, 4))

    def forbidden(*args, **kwargs):
        pytest.fail("Stored native map fell back to legacy")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate("x + y * 2", local_dict={"x": stored, "y": y}, out=output)
    assert result is output
    np.testing.assert_array_equal(output[:], x + y * 2)


@pytest.mark.parametrize("expression", ["x / y", "x // y", "x % y", "x ** y", "x << y", "x >> y"])
def test_evaluate_native_integer_operators(expression, native_graph_runtime, monkeypatch):
    x = np.array([-9, -1, 0, 1, 9], dtype="int64")
    y = np.full_like(x, 2)

    def forbidden(*args, **kwargs):
        pytest.fail("Numeric operator did not reach native lowering")

    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    result = blosc2.evaluate(expression, local_dict={"x": x, "y": y})
    np.testing.assert_array_equal(result, eval(expression, {"x": x, "y": y}))


@pytest.mark.parametrize("backend", ["cc", "tcc"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("explicit_output", [False, True])
def test_evaluate_configured_resident(backend, parallel, explicit_output, monkeypatch):
    import blosc2.native_graph as native

    shape = (3, 200_001) if parallel else (3, 17)
    x = np.linspace(-1, 1, np.prod(shape), dtype="float32").reshape(shape)
    y = -x.copy()
    destination = np.empty(shape, dtype="bool") if explicit_output else None
    reports = []
    original = native.execute_resident

    def record(*args, **kwargs):
        result, report = original(*args, **kwargs)
        reports.append(report)
        return result, report

    def forbidden(*args, **kwargs):
        pytest.fail("Configured NumPy expression fell back to general chunk execution")

    monkeypatch.setattr(native, "execute_resident", record)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    with blosc2.jit_options(jit=True, jit_backend=backend, math_backend="system"):
        for _ in range(2):
            result = blosc2.evaluate(
                "x**2 + y**2 <= 2*x*y + 1", local_dict={"x": x, "y": y}, out=destination
            )
            assert type(result) is np.ndarray
            if explicit_output:
                assert result is destination
            np.testing.assert_array_equal(result, x**2 + y**2 <= 2 * x * y + 1)
            x[:] = 0.25  # Rebinding/current buffers, not cached numerical values.
    assert len(reports) == 2
    assert all(report["tasks"] == (2 if parallel else 1) for report in reports)
    assert all(report["normalization_bytes"] == 0 for report in reports)


def test_evaluate_configured_resident_cache_policy(monkeypatch):
    import blosc2.native_graph as native

    native.configured_map_kernel.cache_clear()
    original = blosc2.MenudetKernel.from_json
    policies = []

    def record(*args, **kwargs):
        policy = blosc2.jit_config.execution_options()
        policies.append((kwargs["jit"], policy["jit_backend"], policy["cflags"], policy["math_backend"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(blosc2.MenudetKernel, "from_json", record)
    x = np.arange(33, dtype="float64")
    requests = [
        (True, "cc", "-O1"),
        (True, "cc", "-O1"),
        (True, "cc", "-O2"),
        (True, "tcc", "-O2"),
        (False, "cc", "-O1"),
        (True, "cc", "-O1"),
    ]
    for jit, backend, flags in requests:
        with blosc2.jit_options(jit=jit, jit_backend=backend, cflags=flags, math_backend="system"):
            np.testing.assert_array_equal(blosc2.evaluate("x + 193", {"x": x}), x + 193)
    assert policies == [
        (True, "cc", "-O1", "system"),
        (True, "cc", "-O2", "system"),
        (True, "tcc", "-O2", "system"),
        (False, "cc", "-O1", "system"),
    ]


def test_evaluate_configured_resident_failure_never_replays(monkeypatch):
    import blosc2.native_graph as native

    def fail(*args, **kwargs):
        raise ArithmeticError("Configured resident execution failed")

    def forbidden(*args, **kwargs):
        pytest.fail("Failed configured resident evaluation was replayed")

    monkeypatch.setattr(native._ConfiguredResidentSchedule, "execute", fail)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    with blosc2.jit_options(jit=True, jit_backend="cc"):
        with pytest.raises(ArithmeticError, match="Configured resident execution failed"):
            blosc2.evaluate("x + 29", {"x": np.arange(33.0)})


def test_configured_resident_respects_graph_rejection(monkeypatch):
    import blosc2.native_graph as native

    def forbidden(*args, **kwargs):
        pytest.fail("Rejected parallel graph was reimported as a configured kernel")

    monkeypatch.setattr(NativeGraph, "partitionable", property(lambda self: False))
    monkeypatch.setattr(native, "configured_map_kernel", forbidden)
    with blosc2.jit_options(jit=True, jit_backend="cc"):
        assert native.prepare_resident("x + 1", {"x": np.ones(600_003)}) is None


def test_evaluate_configured_resident_weak_scalar_error():
    expected = OverflowError if int(np.__version__.split(".")[0]) >= 2 else blosc2.MenudetError
    with blosc2.jit_options(jit=True, jit_backend="cc"):
        with pytest.raises(expected):
            blosc2.evaluate("x + 256", {"x": np.ones(33, dtype="uint8")})


def test_configured_resident_fp_status(monkeypatch):
    import blosc2.native_graph as native

    reports = []
    original = native.execute_resident

    def record(*args, **kwargs):
        result, report = original(*args, **kwargs)
        reports.append(report)
        return result, report

    monkeypatch.setattr(native, "execute_resident", record)
    x = np.full(600_003, np.finfo("float64").max)
    for backend in (None, "cc"):
        with blosc2.jit_options(jit=True, jit_backend=backend, math_backend="system"):
            result = blosc2.evaluate("x * x", {"x": x})
            assert np.isposinf(result).all()
    assert len(reports) == 2
    assert reports[0]["fp_flags"] == reports[1]["fp_flags"]
