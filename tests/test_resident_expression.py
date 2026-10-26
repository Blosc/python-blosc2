"""The direct resident route retains rebinding, output and execution-error contracts."""

import importlib

import numpy as np
import pytest

import blosc2

lazy = importlib.import_module("blosc2.lazyexpr")
native = importlib.import_module("blosc2.native_graph")


@pytest.fixture(autouse=True)
def runtime(native_graph_runtime):
    lazy._evaluate_recipe_cache.clear()
    yield
    lazy._evaluate_recipe_cache.clear()


@pytest.mark.parametrize("size", [100_000, 1_000_000])
def test_resident_read_reuses_schedule_and_rebinds_arrays(monkeypatch, size):
    x = np.arange(size, dtype="int64")
    y = x + 1
    expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
    schedule = expr._resident_native[2]

    def forbidden(*args, **kwargs):
        pytest.fail("Resident read entered general preparation or storage dispatch")

    monkeypatch.setattr(native, "prepare_native_first", forbidden)
    monkeypatch.setattr(native, "compute_native_prefilter_if_eligible", forbidden)
    first = expr[:]
    expr.operands["o0"] = x + 10
    second = expr[:]
    assert expr._resident_native[2] is schedule
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, x + (y - 1))
    np.testing.assert_array_equal(second, x + 10 + (y - 1))
    report = expr._native_execution_report
    assert report["route"] == ("parallel-resident" if report["tasks"] > 1 else "native-resident")


@pytest.mark.parametrize("mode", ["full", "safe"])
def test_parallel_evaluate_reuses_schedule_without_lazy_dispatch(monkeypatch, mode):
    x = np.arange(1_000_000, dtype="int64")
    y = x + 1
    schedules = []
    original = native.prepare_resident

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        schedules.append(result[2])
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Parallel resident evaluate entered lazy preparation")

    monkeypatch.setattr(native, "prepare_resident", prepare)
    monkeypatch.setattr(lazy, "lazyexpr", forbidden)
    with blosc2.expression_evaluation(mode):
        first = blosc2.evaluate("x + (y - 1)", local_dict={"x": x, "y": y})
        second = blosc2.evaluate("x + (y - 1)", local_dict={"x": x + 10, "y": y})
    assert schedules[0] is schedules[1]
    assert not np.shares_memory(first, second)
    np.testing.assert_array_equal(first, x + (y - 1))
    np.testing.assert_array_equal(second, x + 10 + (y - 1))


def test_parallel_resident_uses_current_thread_count(monkeypatch):
    x = np.arange(1_000_000, dtype="int64")
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    schedule = expr._resident_native[2]
    for threads in (2, 4, 1):
        monkeypatch.setattr(blosc2, "nthreads", threads)
        np.testing.assert_array_equal(expr[:], x + 1)
        assert expr._resident_native[2] is schedule
        assert expr._native_execution_report["tasks"] == threads


def test_large_checked_map_retains_general_dispatch():
    x = np.arange(300_000, dtype="int64")
    y = x.astype("uint64")
    assert native.prepare_resident("x + y", {"x": x, "y": y}) is None
    actual = blosc2.evaluate("x + y", local_dict={"x": x, "y": y})
    assert actual.dtype == (x + y).dtype
    np.testing.assert_array_equal(actual, x + y)


def test_large_weak_scalar_overflow_does_not_write():
    x = np.zeros(300_000, dtype="int8")
    out = np.full_like(x, -1)
    with pytest.raises(OverflowError):
        blosc2.evaluate("x + 128", local_dict={"x": x}, out=out)
    np.testing.assert_array_equal(out, -1)


def test_resident_expression_mutation_invalidates_plan():
    x = np.arange(17.0)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    expr.expression = "o0 + 2"
    np.testing.assert_array_equal(expr[:], x + 2)
    expr.operands["o0"] = x.astype("float32")
    actual = expr[:]
    assert actual.dtype == np.dtype("float32")
    np.testing.assert_array_equal(actual, x + 2)


def test_resident_schedule_tracks_compilation_options():
    x = np.arange(17.0)
    with blosc2.jit_options(jit=False, cflags="-O1"):
        first = lazy._prepare_cached_evaluate("x + 1", {"x": x}, None)
    with blosc2.jit_options(jit=False, cflags="-O2"):
        second = lazy._prepare_cached_evaluate("x + 1", {"x": x}, None)
        np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}), x + 1)
    assert first[0] is not second[0]
    assert first[0][1] is not second[0][1]


@pytest.mark.parametrize("engine", ["evaluate", "lazyexpr"])
@pytest.mark.parametrize("size", [17, 1_000_000])
def test_resident_execution_failure_never_replays(engine, monkeypatch, size):
    x = np.arange(size, dtype="int64")
    original = native.prepare_resident
    executed = []

    class FailedSchedule:
        def execute(self, *args, **kwargs):
            executed.append(True)
            raise ValueError("resident execution failed")

        def info(self):
            return {"dtype": x.dtype, "shape": x.shape}

    def prepare(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        return result[0], result[1], FailedSchedule()

    def forbidden(*args, **kwargs):
        pytest.fail("Resident failure replayed through another execution route")

    monkeypatch.setattr(native, "prepare_resident", prepare)
    monkeypatch.setattr(native, "compute_native_graph", forbidden)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    operation = (
        (lambda: blosc2.evaluate("x + 1", local_dict={"x": x}))
        if engine == "evaluate"
        else (lambda: blosc2.lazyexpr("x + 1", {"x": x})[:])
    )
    with pytest.raises(ValueError, match="resident execution failed"):
        operation()
    assert executed == [True]


@pytest.mark.parametrize("kind", ["overlap", "strided", "cast", "fortran"])
@pytest.mark.parametrize("shape", [(4, 6), (1000, 1000)])
def test_resident_explicit_output_preserves_contract(kind, shape):
    x = np.arange(np.prod(shape), dtype="float64").reshape(shape)
    if kind == "overlap":
        out = x
    elif kind == "strided":
        out = np.empty((shape[0], shape[1] * 2))[:, ::2]
    elif kind == "cast":
        out = np.empty_like(x, dtype="float32")
    else:
        out = np.empty_like(x, order="F")
    expected = np.broadcast_to(x + 1, out.shape).copy()
    result = blosc2.evaluate("x + 1", local_dict={"x": x}, out=out)
    assert result is out
    np.testing.assert_array_equal(result, expected)


def test_resident_output_shape_failure_does_not_write():
    x = np.arange(6.0).reshape(1, 6)
    out = np.full((4, 6), -1.0)
    with pytest.raises(ValueError, match="output shape"):
        blosc2.evaluate("x + 1", local_dict={"x": x}, out=out)
    np.testing.assert_array_equal(out, -1)
