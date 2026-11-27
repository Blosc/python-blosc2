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
    geometry = schedule.info()
    assert geometry.pop("schedule_metadata_bytes") > 0
    assert geometry == {
        "items": 17,
        "tile_items": 8,
        "mapped_bytes": 64,
        "reducer_bytes": 8,
        "output_items": 1,
        "shape": (17,),
        "output_shape": (),
        "reduction_groups": 1,
        "reduction_lanes": 17,
        "pairwise_sum": False,
        "broadcast_gather_bytes": 0,
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
    with pytest.raises(ValueError, match="broadcast"):
        plan.specialize([("int64", (17,)), ("int64", (16,))])
    with pytest.raises(ValueError, match="rank"):
        plan.specialize([("int64", (1,) * 17), ("int64", (1,) * 17)])


def test_typed_backend_reports_and_concurrent_schedule():
    from concurrent.futures import ThreadPoolExecutor

    for execution in (0, 1):
        plan = blosc2_ext._TypedExecPlan(descriptor("float64"), 2, 5, execution=execution)
        prepared_jit = plan.info()["has_jit"]
        if execution == 0:
            assert not prepared_jit
        schedule = plan.specialize([("float64", (37,)), ("float64", (37,))], 8, 64)

        def evaluate(offset, schedule=schedule, prepared_jit=prepared_jit):
            x = np.arange(37, dtype=np.float64) + offset
            result, report = schedule.execute((x, -x), workers=2, max_worker_scratch=128)
            assert result == -2 * (36 + offset)
            assert report["map_calls"] == 5
            assert report["jit_calls"] == (5 if prepared_jit else 0)
            assert report["interpreter_calls"] == (0 if prepared_jit else 5)
            assert report["has_jit"] == prepared_jit
            assert report["fp_flags"] == 0
            return result

        with ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(evaluate, range(6))) == [-2 * (36 + offset) for offset in range(6)]

    identity = blosc2_ext._TypedExecPlan([{"op": 0, "dtype": "float64", "slot": 0}], 1, 0, execution=1)
    x = np.arange(17, dtype=np.float64)
    result, report = identity.specialize([("float64", x.shape)], 8, 0).execute((x,))
    np.testing.assert_array_equal(result, x)
    assert not report["has_jit"]
    assert report["jit_calls"] == report["interpreter_calls"] == 0
    assert report["direct_tiles"] == 3


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
        def execute(self, inputs, **kwargs):
            raise ValueError("injected typed execution failure")

    class Prepared:
        def info(self):
            return {
                "reduction": True,
                "identity": False,
                "partitionable": True,
                "inferred_dtype": np.dtype("int64"),
                "map_dtype": np.dtype("int64"),
            }

        def specialize(self, *args):
            return FailedSchedule()

    monkeypatch.setattr(typed_exec, "prepare_typed_plan", lambda *args: Prepared())
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


def test_typed_direct_map_needs_no_mapped_budget():
    nodes = descriptor()[:-1]
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 4)
    schedule = plan.specialize([("int64", (17003,)), ("int64", (17003,))], 16384, 0)
    assert schedule.info()["mapped_bytes"] == 0
    values = np.arange(17003, dtype=np.int64)
    result, report = schedule.execute((values, values))
    np.testing.assert_array_equal(result, 4 * values)
    assert report["temporary_bytes"] == 0


def test_typed_large_identity_tile_and_narrow_reduction():
    values = np.arange(17003, dtype=np.int16)
    for expression in ("min(x)", "min(x + 1)"):
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, {"x": values})
        expected = expr.compute(jit=False, _require_native=True)
        actual = expr.compute(jit=False, _native_tile_items=16384)
        assert actual.tobytes() == expected.tobytes()
        report = expr._native_execution_report
        assert report["route"] == "c-blosc2-typed-resident"
        assert report["temporary_bytes"] == (0 if expression == "min(x)" else 32768)


@pytest.mark.parametrize("shape", [(), (3, 5), (2, 3, 5), (2, 0, 3)])
def test_typed_contiguous_ranked_domains(shape):
    x = np.arange(np.prod(shape, dtype=int), dtype="int64").reshape(shape)
    metadata = {"x": (x.dtype, x.shape)}
    for expression in ("x * 3 + 1", "min(x * 3 + 1)"):
        if expression.startswith("min") and not x.size:
            continue  # Empty minimum retains the comparison error path.
        baseline = NativeGraph.from_expression(expression, {"x": "int64"}, jit=False)
        expected, _ = baseline.specialize(metadata, 4).execute({"x": x})
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, {"x": x})
        with blosc2.compute_options(jit=False):
            actual = expr.compute(_native_tile_items=4) if expression.startswith("min") else expr[()]
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"


def test_typed_output_identity_and_real_allocation_extents():
    nodes = descriptor()[:-1]
    plan = blosc2_ext._TypedExecPlan(nodes, 2, 4)
    schedule = plan.specialize([("int64", (5,)), ("int64", (5,))], 2, 0)
    values = np.arange(5, dtype=np.int64)
    output = np.empty_like(values)
    actual, report = schedule.execute((values, values), out=output)
    assert actual is output
    np.testing.assert_array_equal(output, 4 * values)
    assert report["temporary_bytes"] == 0
    with pytest.raises(NotImplementedError, match="Aliased"):
        schedule.execute((values, values), out=values)
    forged = np.lib.stride_tricks.as_strided(values[:1].copy(), shape=(5,), strides=(8,))
    assert forged.flags.c_contiguous
    output[:] = -7
    with pytest.raises(ValueError, match="extent"):
        schedule.execute((forged, values), out=output)
    np.testing.assert_array_equal(output, -7)
    with pytest.raises(ValueError, match="allocation extent"):
        schedule.execute((values, values), out=forged)


def test_typed_public_disjoint_output():
    values = np.arange(15, dtype="float32").reshape(3, 5)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x * 2 + 1", {"x": values})
    output = np.empty_like(values)
    actual = expr.compute(jit=False, out=output)
    assert actual is output
    np.testing.assert_array_equal(output, values * 2 + 1)
    assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"
    assert expr._native_execution_report["temporary_bytes"] == 0


@pytest.mark.parametrize("reducer", ["sum", "prod", "max"])
def test_typed_serial_reducers_and_accumulator_dtype(reducer):
    x = np.array([-2, 5, 7], dtype="int16")
    expression = f"{reducer}(x + 1)"
    expected = NativeGraph.from_expression(expression, {"x": "int16"}, jit=False).evaluate({"x": x})
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr(expression, {"x": x})
    actual = expr.compute(jit=False, _native_tile_items=2)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    report = expr._native_execution_report
    assert report["route"] == "c-blosc2-typed-resident"
    assert report["temporary_bytes"] == 4  # Map int16, even when the accumulator is int64.


def test_typed_sum_order_and_empty_identities():
    x = np.array([2**24, 1, -(2**24), 3], dtype="float32")
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("sum(x)", {"x": x})
    for tile in (1, 2, 128):
        result = expr.compute(jit=False, _native_tile_items=tile)
        assert result.dtype == np.dtype("float32")
        assert result == 3
        assert expr._native_execution_report["temporary_bytes"] == 0
    for reducer, identity in (("sum", 0), ("prod", 1)):
        with blosc2.expression_evaluation("safe"):
            empty = blosc2.lazyexpr(f"{reducer}(x)", {"x": np.empty((2, 0, 3), dtype="float32")})
        result = empty.compute(jit=False)
        assert result.dtype == np.dtype("float32")
        assert result == identity
        assert empty._native_execution_report["map_calls"] == 0


@pytest.mark.parametrize(
    "expression", ["x + .5", "x + 1.", "x * -0.0", "min(x + y * 3)", "sum(x)", "prod(x)", "max(x)"]
)
def test_named_typed_graph_matches_explicit_adapter(expression):
    inputs = {"x": np.arange(1, 7, dtype="float64").reshape(2, 3), "y": np.full((2, 3), 2, dtype="float64")}
    if "y" not in expression:
        inputs.pop("y")
    dtypes = dict.fromkeys(inputs, "float64")
    baseline = NativeGraph.from_expression(expression, dtypes, jit=False, _executor="adapter")
    typed = NativeGraph.from_expression(expression, dtypes, jit=False, _executor="c-blosc2")
    assert typed.input_dtypes == baseline.input_dtypes
    assert typed.inferred_dtype == baseline.inferred_dtype
    assert typed.reduction_info() == baseline.reduction_info()
    assert typed.to_json() == baseline.to_json()  # Explicit compatibility export only.
    with blosc2.compute_options(math_backend="system"):
        assert typed.to_json() == baseline.to_json()
    metadata = {name: (value.dtype, value.shape) for name, value in inputs.items()}
    schedule = typed.specialize(metadata, 2)
    assert schedule.info()["schedule_metadata_bytes"] > 0
    assert typed.info()["plan_metadata_bytes"] > 0
    result, report = schedule.execute(inputs)
    expected = baseline.evaluate(inputs, tile_items=2)
    assert schedule.stage_info(0) == baseline.specialize(metadata, 2).stage_info(0)
    assert result.shape == expected.shape == schedule.info()["shape"]
    assert result.tobytes() == expected.tobytes()
    assert report["route"] == "c-blosc2-typed-resident"


def test_named_typed_graph_has_no_adapter_prepare_or_borrow(monkeypatch):
    from blosc2 import graph_plan
    from blosc2.typed_exec import prepare_typed_plan

    def forbidden(*args, **kwargs):
        pytest.fail("typed metadata/execution reached the adapter")

    monkeypatch.setattr(graph_plan, "NativeGraphHandle", forbidden)
    plan = NativeGraph.from_expression("sum(x * 2)", {"x": "int16"}, jit=False, _executor="c-blosc2")
    assert plan._handle.resident_artifact() is None
    assert plan._handle.reduction_artifact() is None
    assert plan.info()["map_dtype"] == np.dtype("int16")
    schedule = plan.specialize({"x": ("int16", (3,))}, 2)
    prepare_typed_plan.cache_clear()
    del plan
    gc.collect()
    actual, _ = schedule.execute({"x": np.array([1, 2, 3], dtype="int16")})
    assert actual.dtype == np.dtype("int64")
    assert actual == 12
    with pytest.raises(NotImplementedError, match="Ordered reductions"):
        schedule.execute({"x": np.array([1, 2, 3], dtype="int16")}, workers=2)


@pytest.mark.parametrize(
    "expression",
    ["x.min()", "x ** 2", "x + 0x10", "x + 1_000", "sum(x, dtype=float64)", "x + 00.1", "x + 1.e3"],
)
def test_named_typed_graph_restricts_grammar_and_options(expression):
    with pytest.raises(NotImplementedError):
        NativeGraph.from_expression(expression, {"x": "float64"}, jit=False, _executor="c-blosc2")


def test_typed_frontend_scope_restores_after_parser_exception():
    with pytest.raises(RuntimeError, match="injected"):
        with blosc2_ext._TypedFPScope():
            raise RuntimeError("injected frontend failure")
    with blosc2_ext._TypedFPScope():
        plan = NativeGraph.from_expression("min(x)", {"x": "int64"}, jit=False, _executor="c-blosc2")
    assert plan.inferred_dtype == np.dtype("int64")


def test_named_typed_graph_rejects_unreachable_signature():
    for executor in ("adapter", "c-blosc2"):
        with pytest.raises(ValueError, match="unreachable"):
            NativeGraph.from_expression(
                "x + 1", {"x": "int64", "unused": "int64"}, jit=False, _executor=executor
            )


def test_named_typed_graph_fp_raise_parity():
    from blosc2.menudet_kernel import MenudetError

    x = np.array([np.finfo("float64").max, np.finfo("float64").max, 1.0])
    failures = []
    for executor in ("adapter", "c-blosc2"):
        graph = NativeGraph.from_expression("sum(x)", {"x": "float64"}, jit=False, _executor=executor)
        with pytest.raises(MenudetError) as caught:
            graph.evaluate({"x": x}, tile_items=1, raise_mask=4)
        failures.append((caught.value.status, caught.value.fp_status))
    assert failures[0] == failures[1] == ("evaluation_error", {"flags": 4, "supported": True})


def test_typed_parallel_jobs_preserve_values_and_bound_scratch():
    values = np.random.default_rng(7).integers(-(2**40), 2**40, size=200_003)
    for expression in ("min(x + 1)", "max(x)", "x * 2 + 1"):
        typed = NativeGraph.from_expression(expression, {"x": "int64"}, jit=False, _executor="c-blosc2")
        adapter = NativeGraph.from_expression(expression, {"x": "int64"}, jit=False, _executor="adapter")
        expected = adapter.evaluate({"x": values}, tile_items=1024)
        for workers in (1, 3, 12):
            result, report = typed.evaluate(
                {"x": values}, tile_items=1024, return_report=True, workers=workers
            )
            np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))
            assert report["tasks"] <= workers
            assert report["worker_scratch_bytes"] == report["tasks"] * report["temporary_bytes"]
            assert report["worker_scratch_bytes"] <= 65536
    # The aggregate budget caps the job count instead of growing scratch.
    result, report = typed.evaluate({"x": values}, tile_items=1024, return_report=True, workers=12)
    assert report["tasks"] * report["temporary_bytes"] <= 65536


def test_typed_ordered_reduction_rejects_partitioning():
    values = np.arange(4096, dtype="int64")
    for expression in ("sum(x)", "prod(x + 1)"):
        plan = NativeGraph.from_expression(expression, {"x": "int64"}, jit=False, _executor="c-blosc2")
        schedule = plan.specialize({"x": ("int64", values.shape)}, 64)
        with pytest.raises(NotImplementedError, match="Ordered reductions"):
            schedule.execute({"x": values}, workers=2)
        serial, _ = schedule.execute({"x": values}, workers=1)
        assert serial == NativeGraph.from_expression(
            expression, {"x": "int64"}, jit=False, _executor="adapter"
        ).evaluate({"x": values})


def test_typed_parallel_float_min_unions_flags():
    values = np.array([1.0, np.nan] * 4096, dtype="float64")
    typed = NativeGraph.from_expression("min(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    adapter = NativeGraph.from_expression("min(x)", {"x": "float64"}, jit=False, _executor="adapter")
    reference, reference_report = adapter.evaluate({"x": values}, tile_items=1024, return_report=True)
    for workers in (1, 4):
        result, report = typed.evaluate({"x": values}, tile_items=1024, return_report=True, workers=workers)
        assert (result == result) == (reference == reference)  # NaN propagation matches.
        assert report["fp_flags"] == reference_report["fp_flags"]


def test_typed_resident_rejects_mixed_storage_before_prepare(monkeypatch):
    from blosc2 import typed_exec

    values = np.arange(5, dtype="int64")
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("min(x + y)", {"x": values, "y": blosc2.asarray(values)})

    def forbidden(*args, **kwargs):
        pytest.fail("mixed storage entered resident preparation")

    monkeypatch.setattr(typed_exec, "prepare_typed_plan", forbidden)
    assert typed_exec.compute_typed_resident_if_eligible(expr, (), 2, {}) is None


@pytest.mark.parametrize(
    ("expression", "values"),
    [
        ("sum(x)", np.ones(200, dtype="int8")),
        ("sum(x)", np.ones(300, dtype="uint8")),
        ("sum(x)", np.ones(256, dtype="bool")),
        ("sum(x)", np.ones(40000, dtype="int16")),
        ("prod(x)", np.full(10, 2, dtype="int8")),
    ],
)
def test_typed_narrow_sum_prod_overflow_matches_adapter(expression, values):
    expected = NativeGraph.from_expression(
        expression, {"x": values.dtype.name}, jit=False, _executor="adapter"
    ).evaluate({"x": values}, tile_items=64)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr(expression, {"x": values})
    actual = expr.compute(jit=False, _native_tile_items=64)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"


@pytest.mark.parametrize("reducer", ["any", "all"])
@pytest.mark.parametrize("dtype", ["bool", "int8", "int64", "float32", "float64"])
def test_typed_truth_reductions_and_empty_domains(reducer, dtype):
    values = (np.arange(15) % 3).astype(dtype).reshape(3, 5)
    expression = f"{reducer}(x * 2)"
    typed = NativeGraph.from_expression(expression, {"x": dtype}, jit=False, _executor="c-blosc2")
    adapter = NativeGraph.from_expression(expression, {"x": dtype}, jit=False, _executor="adapter")
    assert typed.info()["partitionable"] is True
    for workers in (1, 3):
        actual, report = typed.evaluate({"x": values}, tile_items=4, return_report=True, workers=workers)
        expected = adapter.evaluate({"x": values}, tile_items=4)
        assert actual.dtype == expected.dtype == np.dtype("bool")
        assert actual.tobytes() == expected.tobytes()
        assert report["tasks"] <= workers
    empty = np.empty((2, 0, 3), dtype=dtype)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr(expression, {"x": empty})
    res = expr.compute(jit=False)
    assert res.dtype == np.dtype("bool")
    assert bool(res) is (reducer == "all")
    assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"


def test_typed_truth_non_short_circuit_fp_flags():
    for reducer, init_val in (("any", 1.0), ("all", 0.0)):
        x = np.full(16, init_val, dtype="float64")
        y = np.ones(16, dtype="float64")
        x[12] = np.inf
        y[12] = 0.0
        typed = NativeGraph.from_expression(
            f"{reducer}(x * y)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
        )
        adapter = NativeGraph.from_expression(
            f"{reducer}(x * y)", {"x": "float64", "y": "float64"}, jit=False, _executor="adapter"
        )
        expected, exp_report = adapter.evaluate({"x": x, "y": y}, tile_items=4, return_report=True)
        for workers in (1, 4):
            actual, report = typed.evaluate(
                {"x": x, "y": y}, tile_items=4, return_report=True, workers=workers
            )
            assert actual.tobytes() == expected.tobytes()
            assert report["fp_flags"] == exp_report["fp_flags"]
            assert report["fp_flags"] & 1


@pytest.mark.parametrize(
    ("shape_a", "shape_b"),
    [((), (2, 3)), ((3, 1), (1, 5)), ((2, 0, 3), (3,))],
)
def test_typed_resident_broadcast_maps_and_reductions(shape_a, shape_b):
    a = np.arange(np.prod(shape_a, dtype=int), dtype="int64").reshape(shape_a)
    b = (np.arange(np.prod(shape_b, dtype=int), dtype="int64") + 1).reshape(shape_b)
    for expression in ("a + b * 2", "sum(a + b * 2)", "any(a + b * 2)"):
        typed = NativeGraph.from_expression(
            expression, {"a": "int64", "b": "int64"}, jit=False, _executor="c-blosc2"
        )
        adapter = NativeGraph.from_expression(
            expression, {"a": "int64", "b": "int64"}, jit=False, _executor="adapter"
        )
        expected = adapter.evaluate({"a": a, "b": b}, tile_items=4)
        actual, report = typed.evaluate({"a": a, "b": b}, tile_items=4, return_report=True)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
        assert report["route"] == "c-blosc2-typed-resident"
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, {"a": a, "b": b})
        with blosc2.compute_options(jit=False):
            lazy_res = expr.compute(_native_tile_items=4) if expression != "a + b * 2" else expr[:]
        assert lazy_res.tobytes() == expected.tobytes()
        assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"


def test_typed_strided_unaligned_and_non_native_views():
    base_x = np.arange(24, dtype=">f8").reshape(4, 6)
    raw = bytearray(1 + 24 * 4)
    unaligned = np.frombuffer(raw, dtype="<i4", count=24, offset=1).reshape(4, 6)
    unaligned[:] = np.arange(24, dtype="<i4").reshape(4, 6)
    vx = base_x[::-1, ::2].T  # shape (3, 4), reversed, strided, transposed, big-endian!
    vy = unaligned[:, 1::2].T  # shape (3, 4), unaligned + strided + transposed!
    for expression in ("x + y", "min(x + y)", "sum(x)", "all(x + y)"):
        case_inputs = {"x": vx} if "y" not in expression else {"x": vx, "y": vy}
        case_dtypes = {name: arr.dtype.newbyteorder("=").name for name, arr in case_inputs.items()}
        adapter = NativeGraph.from_expression(expression, case_dtypes, jit=False, _executor="adapter")
        expected = adapter.evaluate(case_inputs, tile_items=4)
        with blosc2.expression_evaluation("safe"):
            expr = blosc2.lazyexpr(expression, case_inputs)
        with blosc2.compute_options(jit=False):
            actual = expr.compute(_native_tile_items=4) if "(" in expression else expr[:]
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()
        report = expr._native_execution_report
        assert report["route"] == "c-blosc2-typed-resident"
        assert report["gathered_bytes"] > 0

    # Forged negative-stride view must fail before output writes.
    buf = np.arange(4, dtype="int64")
    forged_neg = np.lib.stride_tricks.as_strided(buf[:1], shape=(4,), strides=(-8,))
    out = np.full(4, -11, dtype="int64")
    plan = blosc2_ext._TypedExecPlan([{"op": 0, "dtype": "int64", "slot": 0}], 1, 0)
    schedule = plan.specialize([("int64", (4,))], 4, 0)
    with pytest.raises(ValueError, match="extent"):
        schedule.execute((forged_neg,), out=out)
    np.testing.assert_array_equal(out, -11)


def test_typed_axes_keepdims_and_empty_reductions():
    x = np.arange(24, dtype="int32").reshape(2, 3, 4)
    cases = [
        ("sum(x, axis=1)", x),
        ("sum(x, axis=0, keepdims=True)", x),
        ("prod(x, axis=(0, 2), keepdims=True)", x),
        ("min(x, axis=-1)", x),
        ("sum(x, axis=())", x),
    ]
    for expression, arr in cases:
        typed = NativeGraph.from_expression(expression, {"x": "int32"}, jit=False, _executor="c-blosc2")
        adapter = NativeGraph.from_expression(expression, {"x": "int32"}, jit=False, _executor="adapter")
        assert typed.reduction_info() == adapter.reduction_info()
        ts = typed.specialize({"x": ("int32", arr.shape)}, tile_items=3)
        as_ = adapter.specialize({"x": ("int32", arr.shape)}, tile_items=3)
        assert ts.info()["shape"] == as_.info()["shape"]
        assert ts.info()["map_shape"] == as_.info()["map_shape"]
        actual, treport = ts.execute({"x": arr}, workers=1)
        expected = as_.execute({"x": arr})[0]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
        assert treport["route"] == "c-blosc2-typed-resident"

    empty = np.empty((0, 3, 2), dtype="int32")
    typed_e = NativeGraph.from_expression(
        "sum(x, axis=(0, 2))", {"x": "int32"}, jit=False, _executor="c-blosc2"
    )
    adapter_e = NativeGraph.from_expression(
        "sum(x, axis=(0, 2))", {"x": "int32"}, jit=False, _executor="adapter"
    )
    assert typed_e.evaluate({"x": empty}).tobytes() == adapter_e.evaluate({"x": empty}).tobytes()


def test_typed_initial_values_and_runtime_conversion():
    x = np.array([[1, 2], [3, 4]], dtype="int16")
    for expression in (
        "sum(x, initial=10)",
        "sum(x, axis=0, keepdims=True, initial=5)",
        "min(x, initial=-2.0)",
        "max(x, initial=100)",
    ):
        typed = NativeGraph.from_expression(expression, {"x": "int16"}, jit=False, _executor="c-blosc2")
        adapter = NativeGraph.from_expression(expression, {"x": "int16"}, jit=False, _executor="adapter")
        assert typed.reduction_info() == adapter.reduction_info()
        actual = typed.evaluate({"x": x}, tile_items=1)
        expected = adapter.evaluate({"x": x}, tile_items=1)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()

    empty = np.empty((0, 3), dtype="float64")
    typed_min = NativeGraph.from_expression(
        "min(x, axis=0, initial=7.5)", {"x": "float64"}, jit=False, _executor="c-blosc2"
    )
    adapter_min = NativeGraph.from_expression(
        "min(x, axis=0, initial=7.5)", {"x": "float64"}, jit=False, _executor="adapter"
    )
    assert typed_min.evaluate({"x": empty}).tobytes() == adapter_min.evaluate({"x": empty}).tobytes()

    bad = NativeGraph.from_expression("min(x, initial=-1)", {"x": "uint8"}, jit=False, _executor="c-blosc2")
    sched = bad.specialize({"x": ("uint8", (1,))})
    with pytest.raises(blosc2.MenudetError) as failure:
        sched.execute({"x": np.array([1], dtype="uint8")})
    assert failure.value.status == "evaluation_error"


def test_typed_pairwise_sum_opt_in_invariance():
    values = (np.arange(1, 301, dtype="float64") * 0.1) + 1e-7
    for expression in ("sum(x)", "sum(x, initial=0.25)"):
        plan = NativeGraph.from_expression(expression, {"x": "float64"}, jit=False, _executor="c-blosc2")
        adapter = NativeGraph.from_expression(expression, {"x": "float64"}, jit=False, _executor="adapter")
        with pytest.raises(NotImplementedError):
            adapter.specialize({"x": ("float64", values.shape)}, pairwise_sum=True)
        refs = []
        for tile in (1, 7, 128, 300):
            sched = plan.specialize({"x": ("float64", values.shape)}, tile_items=tile, pairwise_sum=True)
            assert sched.info()["pairwise_sum"] is True
            res, report = sched.execute({"x": values})
            assert report["route"] == "c-blosc2-typed-resident"
            refs.append(res)
        for res in refs[1:]:
            assert res.tobytes() == refs[0].tobytes()

    min_plan = NativeGraph.from_expression("min(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    with pytest.raises(NotImplementedError, match="Pairwise sum"):
        min_plan.specialize({"x": ("float64", (4,))}, pairwise_sum=True)


def test_typed_disjoint_strided_outputs_and_alias_staging():
    x = np.arange(12, dtype="int32").reshape(3, 4)
    y = np.arange(10, 22, dtype="int32").reshape(3, 4)

    # Fortran, transposed/strided and reversed disjoint outputs via typed facade and lazyexpr.
    out_f = np.empty((3, 4), dtype="int32", order="F")
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x + y * 2", {"x": x, "y": y})
    actual = expr.compute(jit=False, out=out_f)
    assert actual is out_f
    np.testing.assert_array_equal(out_f, x + y * 2)
    assert expr._native_execution_report["route"] == "c-blosc2-typed-resident"

    backing = np.full((6, 8), -55, dtype="int32")
    strided_out = backing[::-2, ::-2]  # shape (3, 4) reversed in both dims
    red_plan = NativeGraph.from_expression(
        "sum(x + 1, axis=1)", {"x": "int32"}, jit=False, _executor="c-blosc2"
    )
    red_sched = red_plan.specialize({"x": ("int32", x.shape)}, tile_items=2)
    red_back = np.full(6, -9, dtype="int64")
    red_view = red_back[::-2]
    res, _ = red_sched.execute({"x": x}, out=red_view)
    assert res is red_view
    np.testing.assert_array_equal(red_view, np.sum(x + 1, axis=1, dtype="int64"))
    assert list(red_back[::2]) == [-9, -9, -9]

    map_plan = NativeGraph.from_expression(
        "x + y * 2", {"x": "int32", "y": "int32"}, jit=False, _executor="c-blosc2"
    )
    map_sched = map_plan.specialize({"x": ("int32", x.shape), "y": ("int32", y.shape)}, tile_items=3)
    res_map, _ = map_sched.execute({"x": x, "y": y}, out=strided_out)
    assert res_map is strided_out
    np.testing.assert_array_equal(strided_out, x + y * 2)

    # Overlapping output strides rejected in typed schedule execute.
    zero_out = np.lib.stride_tricks.as_strided(backing[:1, :1], shape=(3, 4), strides=(0, 0))
    with pytest.raises(ValueError, match="output extent"):
        map_sched.execute({"x": x, "y": y}, out=zero_out)

    # Transactional alias staging: reverse in-place, over-budget, raise_mask unchanged.
    buf = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64")
    alias_view = buf[::-1]
    alias_plan = NativeGraph.from_expression("x + x", {"x": "float64"}, jit=False, _executor="c-blosc2")
    alias_sched = alias_plan.specialize({"x": ("float64", buf.shape)}, tile_items=2)
    with pytest.raises(NotImplementedError, match="Aliased"):
        alias_sched.execute({"x": buf}, out=alias_view)
    with pytest.raises(ValueError, match="staging budget"):
        alias_sched.execute_alias({"x": buf}, staging_budget=39, out=alias_view)
    np.testing.assert_array_equal(buf, [1.0, 2.0, 3.0, 4.0, 5.0])
    out_alias, report_alias = alias_sched.execute_alias({"x": buf}, staging_budget=40, out=alias_view)
    assert out_alias is alias_view
    assert report_alias["staging_bytes"] == 40
    np.testing.assert_array_equal(buf, [10.0, 8.0, 6.0, 4.0, 2.0])
    buf[0] = np.finfo("float64").max
    saved = buf.copy()
    with pytest.raises(blosc2.MenudetError) as exc:
        alias_sched.execute_alias({"x": buf}, staging_budget=40, raise_mask=4, out=alias_view)
    assert exc.value.status == "evaluation_error"
    np.testing.assert_array_equal(buf, saved)


def test_typed_partitionable_metadata():
    assert NativeGraph.from_expression(
        "min(x)", {"x": "int64"}, jit=False, _executor="c-blosc2"
    ).partitionable
    assert NativeGraph.from_expression(
        "any(x)", {"x": "int64"}, jit=False, _executor="c-blosc2"
    ).partitionable
    assert not NativeGraph.from_expression(
        "sum(x)", {"x": "int64"}, jit=False, _executor="c-blosc2"
    ).partitionable
    assert not NativeGraph.from_expression(
        "prod(x)", {"x": "int64"}, jit=False, _executor="c-blosc2"
    ).partitionable
