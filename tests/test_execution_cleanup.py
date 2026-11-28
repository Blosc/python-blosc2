"""Ordinary callers hand whole-domain ownership to the typed executor."""

import gc

import numpy as np
import pytest

import blosc2
import blosc2.native_graph as native
import blosc2.resident_execution as resident
import blosc2.storage_readers as readers
from blosc2.graph_plan import NativeGraph
from blosc2.typed_graph import TypedGraphSchedule


@pytest.fixture(autouse=True)
def typed_runtime(monkeypatch, request):
    if not getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)():
        pytest.skip("Requires the typed executor")
    previous = blosc2.set_nthreads(4)
    if request.node.get_closest_marker("allow_compat_fallback") is not None:
        yield
        blosc2.set_nthreads(previous)
        return

    def forbid(*args, **kwargs):
        raise AssertionError("Typed execution entered Python scheduling or a dummy prefilter")

    for name in (
        "_parallel_resident_graph",
        "_parallel_resident_reduction",
        "_parallel_compressed_map",
        "_parallel_compressed_reduction",
        "_execute_resident_kernel",
        "_prefilter_compressed_map",
    ):
        monkeypatch.setattr(native, name, forbid)
    monkeypatch.setattr(resident, "resident_pool", forbid)
    monkeypatch.setattr(resident, "run_dynamic", forbid)
    monkeypatch.setattr(readers, "evaluate_prefilter_numpy", forbid)
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("expression", ["x + 1", "min(x + 1)"])
def test_ordinary_resident_owner(expression, jit):
    x = np.arange(1_048_579, dtype=np.float64)
    expr = blosc2.lazyexpr(expression, {"x": x})
    actual = expr.compute(jit=jit, _getitem=True)
    np.testing.assert_array_equal(actual, x + 1 if expression == "x + 1" else 1.0)
    report = expr._native_execution_report
    assert report["tasks"] == (1 if blosc2.IS_WASM else 4)
    assert report.get("discarded_output_bytes", 0) == 0


@pytest.mark.parametrize("expression", ["x + 1", "min(x + 1)"])
def test_ordinary_compressed_owner(expression, monkeypatch):
    # Preserve public admission gates; lower their size constants for this routing
    # check, not the native budgets or effective worker-count assertions.
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 500_000)
    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_MAP_ITEMS", 500_000)
    blosc2.set_nthreads(8)
    x = np.arange(1_048_579, dtype=np.float64)
    bx = blosc2.asarray(x, chunks=(65536,), blocks=(4096,))
    expr = blosc2.lazyexpr(expression, {"x": bx})
    actual = expr.compute(jit=True, _getitem=True)
    np.testing.assert_array_equal(actual, x + 1 if expression == "x + 1" else 1.0)
    report = expr._native_execution_report
    assert report["route"] == "c-blosc2-typed-storage"
    assert report["tasks"] == (1 if blosc2.IS_WASM else 8)
    assert report["input_materialization_bytes"] == 0
    assert report["worker_scratch_bytes"] <= 64 * 1024 * 1024
    assert report.get("discarded_output_bytes", 0) == 0


def test_resident_output_and_no_failure_replay(monkeypatch):
    x = np.arange(524_291, dtype=np.float64)
    expr = blosc2.lazyexpr("x + 1", {"x": x})
    output = np.full_like(x, -99)
    assert expr.compute(jit=True, _getitem=True, out=output) is output
    np.testing.assert_array_equal(output, x + 1)
    failure = RuntimeError("native execution failed")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise failure

    monkeypatch.setattr(TypedGraphSchedule, "execute", fail)
    with pytest.raises(RuntimeError) as exc:
        expr.compute(jit=True, _getitem=True, out=output)
    assert exc.value is failure
    assert calls == [1]


def test_ordered_reduction_stays_in_c():
    # The old Python partial reductions changed the serial leaf order.
    x = np.tile(np.array([1e16, 1.0, -1e16, 1.0]), 131_073)
    plan = NativeGraph.from_expression("sum(x + 0)", {"x": "float64"}, jit=True)
    expected = plan.evaluate({"x": x})
    expr = blosc2.lazyexpr("sum(x + 0)", {"x": x})
    actual = expr.compute(jit=True)
    assert actual.tobytes() == expected.tobytes()
    assert expr._native_execution_report["tasks"] == 1


def test_configured_borrowed_kernel_retains_policy_and_owner():
    x = np.arange(524_291, dtype=np.float64)
    with blosc2.compute_options(jit_backend="cc", math_backend="system"):
        prepared = native.prepare_resident("x + 1", {"x": x}, jit=True)
    assert prepared is not None
    assert prepared[2].typed is not None
    gc.collect()
    actual, report = native.execute_resident(prepared, {"x": x})
    np.testing.assert_array_equal(actual, x + 1)
    assert report["jit_stages"] == 1
    assert report["tasks"] == (1 if blosc2.IS_WASM else 4)


def test_authored_kernel_uses_borrowed_traversal():
    @blosc2.menudet(jit=True, partitions=(4096,))
    def mapped(x):
        return x + 1

    x = np.arange(524_291, dtype=np.float64)
    lazy = mapped(x)
    np.testing.assert_array_equal(lazy[:], x + 1)
    assert lazy._source_execution_report["route"] == "c-blosc2-typed-storage"
    assert lazy._source_execution_report["tasks"] == (1 if blosc2.IS_WASM else 4)


def test_authored_partition_context_and_selection():
    @blosc2.menudet(jit=False, partitions=(8,))
    def grouped(x):
        return x + block_sum(x)  # noqa: F821 -- native Menudet intrinsic

    x = np.arange(35.0)
    lazy = grouped(x)
    expected = np.concatenate(
        [part + part.sum() for part in np.array_split(x[:32], 4)] + [x[32:] + x[32:].sum()]
    )
    np.testing.assert_array_equal(lazy[3:29], expected[3:29])
    assert lazy._source_execution_report["route"] == "c-blosc2-logical-partitions"


def test_mismatched_storage_aggregate_and_decode_budgets():
    x = np.arange(257.0)
    inputs = {
        "x": blosc2.asarray(x, chunks=(47,), blocks=(13,)),
        "y": blosc2.asarray(x + 1, chunks=(61,), blocks=(17,)),
    }
    plan = NativeGraph.from_expression("x + y", {"x": "float64", "y": "float64"}, jit=True)
    schedule = plan.specialize({name: (value.dtype, value.shape) for name, value in inputs.items()}, 16)
    _, baseline = schedule.execute_arrays(inputs, workers=8, max_worker_scratch=65536)
    tasks = 1 if blosc2.IS_WASM else 8
    assert baseline["tasks"] == tasks
    per_worker = baseline["worker_scratch_bytes"] // tasks
    assert per_worker > 0
    output = np.full_like(x, -99)
    for budget in (
        {"max_worker_scratch": 0},
        {"max_worker_scratch": per_worker - 1},
        {"max_decode_bytes": 1},
    ):
        with pytest.raises((ValueError, blosc2.MenudetError)):
            schedule.execute_arrays(inputs, workers=8, out=output, **budget)
        np.testing.assert_array_equal(output, -99)
    result, report = schedule.execute_arrays(
        inputs, workers=8, out=output, max_worker_scratch=3 * per_worker
    )
    assert result is output
    np.testing.assert_array_equal(result, x + x + 1)
    assert report["tasks"] == (1 if blosc2.IS_WASM else 3)
    assert report["worker_scratch_bytes"] <= 3 * per_worker


@pytest.mark.allow_compat_fallback
def test_storage_admission_falls_back_for_postfilter_bytedelta_and_generator():
    @blosc2.menudet(jit=False)
    def add_one(x):
        return x + 1.0

    @blosc2.menudet(ndim=1, jit=False)
    def gen_seq() -> np.int64:
        return _flat_idx + 10  # noqa: F821 -- native Menudet intrinsic

    x = np.arange(16, dtype=np.float64)
    delta_arr = blosc2.asarray(x, chunks=(8,), blocks=(4,), cparams={"filters": [blosc2.Filter.BYTEDELTA]})
    np.testing.assert_array_equal(add_one(delta_arr)[:], x + 1.0)
    np.testing.assert_array_equal(blosc2.lazyexpr("x + 1", {"x": delta_arr}).compute(), x + 1.0)

    post_arr = blosc2.asarray(x, chunks=(8,), blocks=(4,), dparams={"nthreads": 1})

    @post_arr.schunk.postfilter(np.float64)
    def post(input, output, offset):
        output[:] = input + 2.0

    np.testing.assert_array_equal(add_one(post_arr)[:], x + 3.0)
    np.testing.assert_array_equal(blosc2.lazyexpr("x + 1", {"x": post_arr}).compute(), x + 3.0)

    gen_lazy = gen_seq.specialize().lazy({}, shape=(9,), partitions=(4,))
    np.testing.assert_array_equal(gen_lazy[:], np.arange(9, dtype=np.int64) + 10)


@pytest.mark.parametrize(("chunk", "block"), [(5, 2), (4, 4)])
def test_authored_block_scalar_persisted_partitions_fp_and_empty(chunk, block, tmp_path):
    @blosc2.menudet(jit=False, cardinality="block_scalar")
    def partition_sum(x):
        return block_sum(x)  # noqa: F821

    @blosc2.menudet(jit=False, partitions=(3,), cardinality="block_scalar")
    def group_sum(x):
        return block_sum(x)  # noqa: F821

    @blosc2.menudet(jit=False, cardinality="block_scalar")
    def partition_fp(x):
        y = 1.0 / x + x * x
        return block_sum(y)  # noqa: F821

    # Persisted partition specification, strict summation order, slice selection and JSON roundtrip.
    values = np.array([1e16, 1, -1e16, 3, 4, 5, 6], dtype=np.float64)
    source = blosc2.asarray(values, chunks=(chunk,), blocks=(block,), urlpath=tmp_path / "input.b2nd")
    operation = partition_sum.specialize(source).lazy({"x": source}, partitions=(3,))
    path = tmp_path / "reduction.b2nd"
    operation.save(path)
    restored = blosc2.open(path, mode="r")
    assert restored.partitions == (3,)
    np.testing.assert_array_equal(restored[:], np.array([0.0, 12.0, 6.0]), strict=True)
    np.testing.assert_array_equal(restored[1:], np.array([12.0, 6.0]), strict=True)

    wide = group_sum.specialize(source).lazy({"x": source}, partitions=(10000,))
    np.testing.assert_array_equal(wide[:], [values.sum()])
    reloaded = blosc2.MenudetKernel.from_json(restored.kernel.to_json(), jit=False)
    np.testing.assert_array_equal(reloaded.lazy({"x": source}, partitions=(3,))[:], [0.0, 12.0, 6.0])

    # Multi-group FP status union (div-by-zero in group 0 + overflow in final group).
    fp_values = np.array([0.0, 2.0, 3.0, 4.0, 1e200], dtype=np.float64)
    fp_source = blosc2.asarray(fp_values, chunks=(3,), blocks=(2,))
    fp_lazy = partition_fp.specialize(fp_source).lazy({"x": fp_source}, partitions=(2,))
    fp_actual = fp_lazy[:]
    assert fp_lazy._source_execution_report["route"] == "c-blosc2-logical-partitions"
    assert fp_lazy._source_execution_report["fp_supported"] is True
    flags = fp_lazy._source_execution_report["fp_flags"]
    assert flags & 2
    assert flags & 4
    expected = [
        partition_fp.specialize(fp_source).evaluate_block({"x": fp_values[i : i + 2]})
        for i in range(0, len(fp_values), 2)
    ]
    np.testing.assert_array_equal(fp_actual, np.asarray(expected), strict=True)

    # Empty logical domain returns empty output with clean zero FP flags.
    empty_source = blosc2.asarray(np.empty(0, dtype=np.float64))
    empty_lazy = partition_sum.specialize(empty_source).lazy({"x": empty_source}, partitions=(3,))
    np.testing.assert_array_equal(empty_lazy[:], np.empty(0, dtype=np.float64), strict=True)
    assert empty_lazy._source_execution_report["fp_flags"] == 0
