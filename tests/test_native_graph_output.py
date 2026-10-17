"""Bindings/dispatch for native descriptor outputs; numerical matrices live in miniexpr."""

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def output_runtime(native_graph_runtime):
    if not getattr(blosc2.blosc2_ext, "native_graph_output_available", lambda: False)():
        pytest.skip("Requires native destination descriptors")


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("layout", ["F", "reverse", "stepped"])
def test_native_graph_destination(jit, layout):
    x = np.arange(24.0, dtype="float64").reshape(4, 6)
    storage = np.full((8, 12), -999.0)
    if layout == "F":
        output = np.empty(x.shape, order="F")
    elif layout == "reverse":
        output = storage[6:2:-1, 10:4:-1]
    else:
        output = storage[::2, ::2]
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype}, jit=jit)
    schedule = plan.specialize({"x": (x.dtype, x.shape)}, tile_items=7)
    result, report = schedule.execute({"x": x}, out=output)
    assert result is output
    np.testing.assert_array_equal(result, x + 1)
    assert report["temporary_bytes"] <= 7 * 8
    if not jit:
        assert report["interpreter_stages"]
    if layout != "F":
        selected = np.zeros(storage.shape, dtype=bool)
        if layout == "reverse":
            selected[6:2:-1, 10:4:-1] = True
        else:
            selected[::2, ::2] = True
        np.testing.assert_array_equal(storage[~selected], -999)


def test_native_graph_strided_reduction_and_staged_final():
    x = np.arange(24.0).reshape(4, 6)
    for expression, expected in [("sum(x, axis=1)", x.sum(axis=1)), ("x - sum(x)", x - x.sum())]:
        plan = blosc2.NativeGraph.from_expression(expression, {"x": x.dtype})
        schedule = plan.specialize({"x": (x.dtype, x.shape)})
        storage = np.empty(expected.size * 2)
        output = storage[::2].reshape(expected.shape)
        result, _ = schedule.execute({"x": x}, out=output)
        assert result is output
        np.testing.assert_array_equal(result, expected)


def test_self_overlapping_output_rejected_before_write():
    x = np.ones((2, 3))
    storage = np.full(6, -999.0)
    output = np.lib.stride_tricks.as_strided(storage, shape=x.shape, strides=(8, 8))
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    schedule = plan.specialize({"x": (x.dtype, x.shape)})
    with pytest.raises(blosc2.PortableArtifactError):
        schedule.execute({"x": x}, out=output)
    np.testing.assert_array_equal(storage, -999)


@pytest.mark.parametrize("jit", [False, True])
def test_native_host_executor_partitions_strided_output(jit):
    x = np.arange(600_003.0, dtype="float64")
    plan = blosc2.NativeGraph.from_expression("x * 2 + 1", {"x": x.dtype}, jit=jit)
    assert plan.info()["partitionable"]
    schedule = plan.specialize({"x": (x.dtype, x.shape)}, tile_items=4096)
    storage = np.full(x.size * 2, -999.0)
    output = storage[::-2]
    result, report = schedule.execute({"x": x}, out=output, workers=4)
    assert result is output
    assert report["tasks"] == 4
    np.testing.assert_array_equal(result, x * 2 + 1)
    np.testing.assert_array_equal(storage[::2], -999)


def test_native_executor_joins_on_submission_failure(monkeypatch):
    import blosc2.resident_execution as resident

    pool = resident.resident_pool()
    futures = []

    class FailingPool:
        def submit(self, *args):
            if futures:
                raise ArithmeticError("submission failed")
            future = pool.submit(*args)
            futures.append(future)
            return future

    monkeypatch.setattr(resident, "resident_pool", FailingPool)
    x = np.arange(600_003.0)
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    schedule = plan.specialize({"x": (x.dtype, x.shape)})
    with pytest.raises(ArithmeticError, match="submission failed"):
        schedule.execute({"x": x}, workers=4)
    assert futures
    assert all(future.done() for future in futures)


def test_public_disjoint_views_of_same_allocation():
    storage = np.arange(48.0)
    x, output = storage[:24], storage[24:]
    expected = x + 1
    assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=output) is output
    np.testing.assert_array_equal(output, expected)


def test_buffer_backed_destination():
    x = np.arange(24.0)
    storage = bytearray(24 * 16)
    output = np.frombuffer(storage, dtype="float64")[::2]
    assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=output) is output
    np.testing.assert_array_equal(output, x + 1)


def test_public_native_sources_preserve_strided_output():
    x = np.arange(24.0).reshape(4, 6)
    stored = blosc2.asarray(x, chunks=(2, 3), blocks=(1, 3))
    output = np.empty(x.shape, order="F")
    plan = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    result, report = plan.evaluate_sources({"x": stored}, out=output, tile_items=7, return_report=True)
    assert result is output
    np.testing.assert_array_equal(result, x + 1)
    assert report["maximum_reader_items"] <= 7
    assert report["interpreter_stages"] == 1


@pytest.mark.parametrize("layout", ["C", "F", "reverse", "stepped"])
def test_native_identity_copy_without_interpreter_or_jit(layout):
    x = np.arange(24.0).reshape(4, 6)
    if layout == "F":
        x = np.asfortranarray(x)
        output = np.empty_like(x, order="F")
    elif layout == "reverse":
        x = x[::-1, ::-1]
        output = np.empty((4, 6))[::-1, ::-1]
    elif layout == "stepped":
        output = np.empty((8, 12))[::2, ::2]
    else:
        output = np.empty_like(x)
    plan = blosc2.NativeGraph.from_expression("x", {"x": x.dtype})
    result, report = plan.evaluate({"x": x}, out=output, return_report=True)
    assert result is output
    np.testing.assert_array_equal(result, x)
    assert report["evaluated_tiles"] == 0


@pytest.mark.parametrize("jit", [False, True])
def test_matching_fortran_maps_avoid_gather_and_scatter(jit):
    x = np.asfortranarray(np.arange(24.0).reshape(4, 6))
    y = np.asfortranarray(x + 2)
    output = np.empty(x.shape, order="F")
    plan = blosc2.NativeGraph.from_expression("x + y", {"x": x.dtype, "y": y.dtype}, jit=jit)
    result, report = plan.evaluate({"x": x, "y": y}, out=output, return_report=True)
    assert result is output
    np.testing.assert_array_equal(result, x + y)
    assert report["temporary_bytes"] == 0
    assert report["gathered_bytes"] == 0


def test_native_final_conversion_into_fortran_output():
    import json

    x = np.arange(24.0).reshape(4, 6)
    base = blosc2.NativeGraph.from_expression("x + 1", {"x": x.dtype})
    graph = json.loads(base.to_json())
    graph["output"]["dtype"] = "float32"
    plan = blosc2.NativeGraph(graph)
    output = np.empty(x.shape, dtype="float32", order="F")
    result = plan.evaluate({"x": x}, out=output)
    assert result is output
    np.testing.assert_array_equal(result, (x + 1).astype("float32"))


def test_native_staged_final_conversion_into_fortran_output():
    x = np.arange(24.0).reshape(4, 6)
    # Staged regions express conversion as an explicit cast node, not a final
    # output-dtype override; preserve the native format's established contract.
    plan = blosc2.NativeGraph.from_expression("float32(x - sum(x))", {"x": x.dtype})
    output = np.empty(x.shape, dtype="float32", order="F")
    result = plan.evaluate({"x": x}, out=output)
    assert result is output
    np.testing.assert_array_equal(result, (x - x.sum()).astype("float32"))
