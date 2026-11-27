"""Native compressed bindings share the typed resident schedule; no replay."""

import numpy as np
import pytest

import blosc2
from blosc2 import blosc2_ext
from blosc2.graph_plan import NativeGraph
from blosc2.menudet_kernel import MenudetError

pytestmark = pytest.mark.skipif(
    not getattr(blosc2_ext, "_b2nd_exec_available", lambda: False)(),
    reason="requires opt-in C-Blosc2 typed executor",
)


def storage_inputs():
    x = np.arange(35, dtype=np.float64).reshape(5, 7) + 10
    y = np.arange(35, dtype=np.float64).reshape(5, 7) + 100
    return {
        "x": blosc2.asarray(x, chunks=(3, 4), blocks=(2, 3), contiguous=False),
        "y": blosc2.asarray(y, chunks=(4, 3), blocks=(3, 2), contiguous=False),
    }


def test_storage_same_plan_different_layouts_and_no_python_reads(monkeypatch):
    inputs = storage_inputs()
    adapter = NativeGraph.from_expression(
        "min(x + y)", {"x": "float64", "y": "float64"}, jit=False, _executor="adapter"
    )
    expected = adapter.evaluate_sources(inputs, tile_items=8)
    typed = NativeGraph.from_expression(
        "min(x + y)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    with blosc2.compute_options(jit_backend="cc"):
        preferred = NativeGraph.from_expression(
            "min(x + y)", {"x": "float64", "y": "float64"}, jit=True, _executor="c-blosc2"
        )

    def forbidden(*args, **kwargs):
        pytest.fail("native compressed execution reached a Python reader or adapter")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr("blosc2.native_graph.evaluate_graph_sources", forbidden)
    actual, report = typed.evaluate_sources(inputs, tile_items=8, return_report=True)
    assert actual.tobytes() == expected.tobytes()
    assert report["route"] == "c-blosc2-typed-storage"
    assert report["tiles"] == report["map_calls"] == 5
    assert report["decode_bytes"] == 2 * 4 * 6 * 8
    assert report["gather_bytes"] == 2 * 8 * 8
    assert report["gathered_bytes"] == 2 * 35 * 8
    assert report["read_spans"] == 10
    assert report["decodes"] > 0
    assert report["decode_cache_hits"] + report["decodes"] == 70
    assert report["input_materialization_bytes"] == 0
    assert report["fp_flags"] == 0
    assert report["tasks"] == 1
    compiled, compiled_report = preferred.evaluate_sources(inputs, tile_items=8, return_report=True)
    assert compiled.tobytes() == expected.tobytes()
    assert compiled_report["has_jit"] == preferred.has_jit
    assert compiled_report["jit_calls"] == (5 if preferred.has_jit else 0)


def test_storage_preflight_budgets_and_invocation_reuse():
    from concurrent.futures import ThreadPoolExecutor

    inputs = storage_inputs()
    typed = NativeGraph.from_expression(
        "min(x + y)", {"x": "float64", "y": "float64"}, jit=False, _executor="c-blosc2"
    )
    schedule = typed.specialize({name: (a.dtype, a.shape) for name, a in inputs.items()}, 8)
    output = np.array(-77.0)
    for budget in ({"max_decode_bytes": 383}, {"max_gather_bytes": 127}):
        with pytest.raises(ValueError, match="budget"):
            schedule.execute_arrays(inputs, out=output, **budget)
        assert output == -77
    result, report = schedule.execute_arrays(inputs, out=output)
    assert result is output
    assert result == 110
    inputs["x"][0, 0] = -100.0
    changed, fresh_report = schedule.execute_arrays(inputs)
    assert changed == 0
    assert fresh_report["decodes"] == report["decodes"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(schedule.execute_arrays, [inputs] * 4))
    for result, concurrent_report in results:
        assert result == 0
        assert concurrent_report["decodes"] == report["decodes"]


def test_storage_fp_failure_does_not_publish_reduction():
    x = np.arange(35, dtype=np.float64) + 10
    x.view(np.uint64).flat[-1] = 0x7FF0000000000001
    inputs = {"x": blosc2.asarray(x, chunks=(9,), blocks=(4,), contiguous=False)}
    typed = NativeGraph.from_expression("min(x + x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    output = np.array(-77.0)
    with pytest.raises(MenudetError) as error:
        typed.evaluate_sources(inputs, tile_items=8, raise_mask=1, out=output)
    assert error.value.fp_status["flags"] & 1
    assert output == -77
    actual, report = typed.evaluate_sources(inputs, tile_items=8, return_report=True)
    assert np.isnan(actual)
    assert report["fp_flags"] & 1
    assert report["tiles"] == 5


def test_storage_identity_reduction_with_special_chunks():
    values = blosc2.zeros((17,), dtype="float64", chunks=(9,), blocks=(4,), contiguous=False)
    typed = NativeGraph.from_expression("min(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    actual, report = typed.evaluate_sources({"x": values}, tile_items=8, return_report=True)
    assert actual == 0
    assert report["map_calls"] == 0
    assert report["direct_tiles"] == 3
    assert report["temporary_bytes"] == 0
    assert report["gathered_bytes"] == 17 * 8


def test_storage_unsupported_bindings_do_not_replay(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported native storage binding replayed the adapter")

    monkeypatch.setattr("blosc2.native_graph.evaluate_graph_sources", forbidden)
    inputs = {"x": blosc2.asarray(np.arange(17, dtype="float64"), contiguous=True)}
    graph = NativeGraph.from_expression("min(x)", {"x": "float64"}, jit=False, _executor="c-blosc2")
    with pytest.raises(NotImplementedError, match="storage binding"):
        graph.evaluate_sources(inputs)
    non_native_b2 = blosc2.asarray(np.arange(17, dtype=">f8"), chunks=(9,), blocks=(4,), contiguous=False)
    f64_sched = graph.specialize({"x": ("float64", (17,))}, 8)
    with pytest.raises(NotImplementedError, match="storage binding"):
        f64_sched.execute_arrays({"x": non_native_b2})
    # A partial-axis reduction remains unadmitted even when its result has
    # one item (an unreduced unit dimension does not close the axis gate).
    plan = blosc2_ext._TypedExecPlan(
        [{"op": 0, "dtype": "float64", "slot": 0}, {"op": 7, "args": [0], "axes": [1]}], 1, 1
    )
    schedule = plan.specialize([("float64", (1, 7))], 4)
    values = blosc2.asarray(np.ones((1, 7)), chunks=(1, 5), blocks=(1, 3), contiguous=False)
    output = np.array([-77.0])
    with pytest.raises(NotImplementedError, match="whole-domain"):
        schedule.execute_arrays((values,), out=output)
    assert output[0] == -77


@pytest.mark.parametrize(
    ("expr", "dtype", "values", "expected"),
    [
        ("sum(x)", "int16", np.arange(-5, 30, dtype=np.int16).reshape(5, 7), np.int64(420)),
        ("min(x)", "float32", (np.arange(35, dtype=np.float32).reshape(5, 7) + 1) * 0.5, np.float32(0.5)),
        ("any(x)", "bool", np.array([False] * 34 + [True]).reshape(5, 7), np.bool_(True)),
    ],
)
def test_storage_numeric_dtypes(expr, dtype, values, expected):
    arr = blosc2.asarray(values, chunks=(3, 4), blocks=(2, 3), contiguous=False)
    graph = NativeGraph.from_expression(expr, {"x": dtype}, jit=False, _executor="c-blosc2")
    actual, report = graph.evaluate_sources({"x": arr}, tile_items=8, return_report=True)
    assert actual == expected
    assert actual.dtype == np.asarray(expected).dtype
    assert report["route"] == "c-blosc2-typed-storage"


def test_storage_mixed_resident_and_parallel_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("mixed/parallel storage execution reached a Python reader or adapter")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr("blosc2.native_graph.evaluate_graph_sources", forbidden)
    x_np = (np.arange(35, dtype=np.float32).reshape(5, 7) + 1) * 0.5
    y_np = np.arange(35, dtype=np.float32).reshape(5, 7) * 2.0
    x_b2 = blosc2.asarray(x_np, chunks=(3, 4), blocks=(2, 3), contiguous=False)

    min_graph = NativeGraph.from_expression(
        "min(x + y)", {"x": "float32", "y": "float32"}, jit=False, _executor="c-blosc2"
    )
    actual, report = min_graph.evaluate_sources(
        {"x": x_b2, "y": y_np}, tile_items=8, workers=3, return_report=True
    )
    assert actual == np.float32(0.5)
    assert report["tasks"] == 3
    assert report["decode_bytes"] == 3 * 24 * 4

    add_graph = NativeGraph.from_expression(
        "x + y", {"x": "float32", "y": "float32"}, jit=False, _executor="c-blosc2"
    )
    sum_out, add_report = add_graph.evaluate_sources(
        {"x": x_b2, "y": y_np}, tile_items=8, workers=3, return_report=True
    )
    np.testing.assert_array_equal(sum_out, x_np + y_np)
    assert add_report["tasks"] == 3

    sched = add_graph.specialize({"x": ("float32", (5, 7)), "y": ("float32", (5, 7))}, 8)
    capped_out, capped_report = sched.execute_arrays(
        {"x": x_b2, "y": y_np}, workers=3, max_decode_bytes=2 * 24 * 4
    )
    np.testing.assert_array_equal(capped_out, x_np + y_np)
    assert capped_report["tasks"] == 2
    assert capped_report["decode_bytes"] == 2 * 24 * 4

    with pytest.raises(ValueError, match="decode budget"):
        sched.execute_arrays({"x": x_b2, "y": y_np}, workers=3, max_decode_bytes=24 * 4 - 1)

    non_native_out, _ = sched.execute_arrays({"x": x_b2, "y": y_np.astype(">f4")}, workers=2)
    np.testing.assert_array_equal(non_native_out, x_np + y_np)

    sum_graph = NativeGraph.from_expression("sum(x)", {"x": "int16"}, jit=False, _executor="c-blosc2")
    x16 = blosc2.asarray(
        np.arange(-5, 30, dtype=np.int16).reshape(5, 7), chunks=(3, 4), blocks=(2, 3), contiguous=False
    )
    with pytest.raises(NotImplementedError, match="Ordered"):
        sum_graph.evaluate_sources({"x": x16}, tile_items=8, workers=2)
    sum_sched = sum_graph.specialize({"x": ("int16", (5, 7))}, 8)
    with pytest.raises(NotImplementedError, match="Ordered"):
        sum_sched.execute_arrays({"x": x16}, workers=2, max_decode_bytes=24 * 2)
