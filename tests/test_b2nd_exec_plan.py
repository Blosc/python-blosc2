"""Private typed boundary tests: explicit selection, never adapter fallback."""

import gc

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext
from blosc2.graph_plan import NativeGraph

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)(),
    reason="requires opt-in C-Blosc2 typed executor",
)


def descriptor(dtype="int64"):
    return [
        {"op": 0, "dtype": dtype, "slot": 0},
        {"op": 0, "dtype": dtype, "slot": 1},
        {"op": 1, "dtype": "int64", "scalar": np.int64(3).tobytes(), "weak": True},
        {"op": 4, "args": [1, 2]},
        {"op": 2, "args": [0, 3]},
        {"op": 6, "args": [4]},
    ]


def test_typed_prepare_copies_descriptor_and_schedule_owns_plan():
    nodes = descriptor()
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 5)
    nodes.clear()
    assert plan.info()["inferred_dtype"] == np.dtype("int64")
    schedule = plan.specialize([("int64", (17,)), ("int64", (17,))], 8, 64)
    del plan
    gc.collect()
    assert schedule.info() == {
        "items": 17,
        "tile_items": 8,
        "mapped_bytes": 64,
        "reducer_bytes": 8,
        "output_items": 1,
    }
    x = np.arange(17, dtype=np.int64)
    actual, _ = schedule.execute((x, -x))
    assert actual == -32


def test_typed_errors_and_metadata_only_specialization():
    nodes = descriptor()
    nodes[3]["args"] = [3, 2]
    with pytest.raises(ValueError, match="reference"):
        blosc2_ext._TypedExecPlan(nodes, 2, 5)
    plan = blosc2_ext._TypedExecPlan(descriptor(), 2, 5)
    with pytest.raises(ValueError, match="budget"):
        plan.specialize([("int64", (17,)), ("int64", (17,))], 8, 63)
    with pytest.raises(NotImplementedError, match="same-shape"):
        plan.specialize([("int64", (17,)), ("int64", (16,))])
    with pytest.raises(ValueError, match="rank"):
        plan.specialize([("int64", (1,) * 17), ("int64", (1,) * 17)])


def test_typed_multiple_inputs_map_min_and_no_old_executor(monkeypatch):
    inputs = {"x": np.arange(37, dtype=np.int64), "y": np.arange(37, dtype=np.int64)[::-1].copy()}
    expected = {}
    for expression in ("x + y * 3", "min(x + y * 3)"):
        expected[expression] = NativeGraph.from_expression(
            expression, {"x": "int64", "y": "int64"}, jit=False
        ).evaluate(inputs)

    def forbidden(*args, **kwargs):
        pytest.fail("typed execution reached the old graph executor")

    monkeypatch.setattr(NativeGraph, "from_expression", forbidden)
    monkeypatch.setattr(NativeGraph, "evaluate", forbidden)
    monkeypatch.setattr("blosc2.native_graph.compute_native_graph", forbidden)
    for expression, reference in expected.items():
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, inputs)
        with blosc2.compute_options(jit=False):
            actual = expr.compute(_native_tile_items=8) if expression.startswith("min") else expr[:]
        np.testing.assert_array_equal(actual, reference)
        report = expr._native_execution_report
        assert report["route"] == "c-blosc2-typed-resident"
        tile = 8 if expression.startswith("min") else 1024
        assert report["map_calls"] > 0
        assert report["temporary_bytes"] <= min(tile, 37) * 8
        inputs["x"][:] += 1
        with blosc2.compute_options(jit=False):
            changed = expr.compute(_native_tile_items=8) if expression.startswith("min") else expr[:]
        np.testing.assert_array_equal(changed, reference + 1)
        inputs["x"][:] -= 1


def test_typed_float_capture_and_fp_failure():
    nodes = descriptor("float32")
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 5)
    assert plan.info()["inferred_dtype"] == np.dtype("float32")
    schedule = plan.specialize([("float32", (5,)), ("float32", (5,))], 2)
    x = np.arange(5, dtype=np.float32)
    actual, report = schedule.execute((x, x))
    assert actual.dtype == np.dtype("float32")
    assert actual == 0
    assert report["map_calls"] == 3
    assert report["temporary_bytes"] == 8
    with pytest.raises(ValueError, match="floating"):
        schedule.execute((x, np.full(5, np.finfo("float32").max, dtype="float32")), 4)
    # A failed invocation must not retain scratch, flags, or values.
    actual, report = schedule.execute((x, x))
    assert actual == 0
    assert report["fp_flags"] == 0


def test_typed_identity_uses_declared_slot_and_no_map_kernel():
    nodes = [{"op": 0, "dtype": "int64", "slot": i} for i in range(2)]
    nodes.append({"op": 6, "args": [1]})
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 2)
    schedule = plan.specialize([("int64", (5,)), ("int64", (5,))], 2, 0)
    x = np.arange(5, dtype=np.int64)
    result, report = schedule.execute((x, -x))
    assert result == -4
    assert report["direct_tiles"] == 3
    assert report["map_calls"] == report["temporary_bytes"] == 0
    assert not report["fp_supported"]


def test_typed_failure_never_replays_adapter(monkeypatch):
    from blosc2 import typed_exec

    class FailedSchedule:
        def execute(self, inputs):
            raise ValueError("injected typed execution failure")

    class Prepared:
        def specialize(self, *args):
            return FailedSchedule()

    monkeypatch.setattr(typed_exec, "_prepare", lambda *args: Prepared())
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("min(x + 1)", {"x": np.arange(7, dtype=np.int64)})
    with pytest.raises(ValueError, match="injected typed execution"):
        expr.compute(jit=False)


def test_typed_shared_dag_and_unreachable_arithmetic():
    nodes = descriptor()[:-1]
    nodes[4]["args"] = [3, 3]
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 4)
    schedule = plan.specialize([("int64", (5,)), ("int64", (5,))], 2)
    x = np.arange(5, dtype=np.int64)
    result, report = schedule.execute((x, -x))
    np.testing.assert_array_equal(result, -6 * x)
    assert report["map_calls"] == 3

    nodes = [
        {"op": 0, "dtype": "float32", "slot": 0},
        {"op": 1, "dtype": "float32", "scalar": np.float32(np.finfo("float32").max).tobytes()},
        {"op": 4, "args": [0, 1]},  # Unreachable: must not raise overflow.
    ]
    plan = blosc2_ext._TypedExecPlan(nodes, 1, 0)
    schedule = plan.specialize([("float32", (5,))], 2, 0)
    values = np.full(5, 100, dtype="float32")
    result, report = schedule.execute((values,), 4)
    np.testing.assert_array_equal(result, values)
    assert report["map_calls"] == report["fp_flags"] == 0


def test_typed_admission_retains_environment_policy(monkeypatch):
    from blosc2.typed_exec import compute_typed_resident_if_eligible

    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("min(x)", {"x": np.arange(7, dtype=np.int64)})
    monkeypatch.setenv("ME_DSL_FP_MODE", "contract")
    assert compute_typed_resident_if_eligible(expr, (), 8, {}) is None


def test_typed_named_exact_capture():
    inputs = {
        "x": np.array([7, -3, 10], dtype=np.int64),
        "y": np.array([2, 1, 0], dtype=np.int64),
        "scale": np.int64(2**63 - 1),
    }
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("min(x + y * scale)", inputs)
    expected = expr.compute(jit=False, _require_native=True)
    actual = expr.compute(jit=False)
    assert actual.tobytes() == expected.tobytes()
    assert actual == 5
    assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"
