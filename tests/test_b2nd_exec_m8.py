"""M8 cutover verification: default NativeGraph dispatch, fallback, and pilot retirement."""

import json

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph


def _exec_available():
    return bool(getattr(blosc2.blosc2_ext, "_b2nd_exec_available", lambda: False)())


def _adapter_available():
    return bool(getattr(blosc2.blosc2_ext, "native_graph_executor_available", lambda: True)())


def test_m8_default_executor_selects_c_blosc2(monkeypatch):
    if not _exec_available() or not _adapter_available():
        pytest.skip("Requires opt-in C-Blosc2 typed executor and adapter")

    class _ForbidAdapter:
        def __new__(cls, *args, **kwargs):
            raise AssertionError("adapter forbidden for admitted M8 path")

        @classmethod
        def from_expression(cls, *args, **kwargs):
            raise AssertionError("adapter forbidden for admitted M8 path")

    monkeypatch.setattr(blosc2.graph_plan, "NativeGraphHandle", _ForbidAdapter)

    x = np.arange(1, 13, dtype=np.float64).reshape(3, 4)
    y = np.arange(4, dtype=np.float64)

    expr_plan = NativeGraph.from_expression("x * 2.5 + y", {"x": "float64", "y": "float64"})
    assert expr_plan.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(expr_plan.evaluate({"x": x, "y": y}), x * 2.5 + y)
    assert json.loads(expr_plan.to_json())["format"] == "menudet-graph-1"
    assert json.loads(expr_plan.map_json())["schema_version"] == "0.1"

    roundtrip = NativeGraph.from_json(expr_plan.to_json())
    assert roundtrip.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(roundtrip.evaluate({"x": x, "y": y}), x * 2.5 + y)

    staged_plan = NativeGraph(NativeGraph.from_expression("x - sum(x, axis=0)", {"x": "float64"}).to_json())
    assert staged_plan.info()["executor"] == "c-blosc2"
    f_out = np.empty(x.shape, dtype=np.float64, order="F")
    result, report = staged_plan.evaluate({"x": x}, out=f_out, return_report=True)
    assert result is f_out
    np.testing.assert_allclose(result, x - np.sum(x, axis=0))
    assert report["stages"] == 2

    red_plan = NativeGraph.from_expression("sum(x, dtype=int8)", {"x": "int64"})
    assert red_plan.info()["executor"] == "c-blosc2"
    arr_i64 = np.array([120, 120, 120], dtype=np.int64)
    red_out = red_plan.evaluate({"x": arr_i64})
    assert red_out.dtype == np.dtype("int8")
    assert red_out == np.sum(arr_i64, dtype="int8")

    # Compressed source execution and lazyexpr compute without touching adapter.
    bx = blosc2.asarray(np.arange(1, 101, dtype=np.float64), chunks=(32,), blocks=(16,))
    stream_plan = NativeGraph.from_expression("sum(x * 2 + 1)", {"x": "float64"})
    assert stream_plan.info()["executor"] == "c-blosc2"
    np.testing.assert_allclose(
        stream_plan.evaluate_sources({"x": bx}, tile_items=17),
        np.sum(bx[:] * 2 + 1),
    )

    lazy_res = blosc2.evaluate(
        "x**3 + 1", local_dict={"x": np.arange(8, dtype=np.int64)}, optimization="aggressive"
    )
    np.testing.assert_array_equal(lazy_res, np.arange(8, dtype=np.int64) ** 3 + 1)


def test_m8_fallback_and_explicit_executor_modes():
    if _adapter_available():
        x = np.arange(8, dtype=np.float64)
        explicit_adapter = NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="adapter")
        assert explicit_adapter.info()["executor"] == "adapter"
        np.testing.assert_array_equal(explicit_adapter.evaluate({"x": x}), x + 1)

    if not _exec_available():
        if _adapter_available():
            plan = NativeGraph.from_expression("x + 1", {"x": "float64"})
            assert plan.info()["executor"] == "adapter"
            with pytest.raises(NotImplementedError):
                NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="c-blosc2")
        return

    if not _adapter_available():
        return

    # Unsupported typed executor policies/dtypes fall back under auto, reject under c-blosc2.
    with blosc2.compute_options(fp_accuracy=blosc2.FPAccuracy.HIGH):
        auto_plan = NativeGraph.from_expression("sin(x)", {"x": "float64"})
        assert auto_plan.info()["executor"] == "adapter"
        with pytest.raises(NotImplementedError):
            NativeGraph.from_expression("sin(x)", {"x": "float64"}, _executor="c-blosc2")

    with pytest.raises(ValueError, match="Unknown graph executor"):
        NativeGraph.from_expression("x + 1", {"x": "float64"}, _executor="invalid")


def test_m8_retires_single_case_pilots(monkeypatch):
    if not _exec_available():
        pytest.skip("Requires opt-in C-Blosc2 typed executor")

    def forbid_i64_pilot(*args, **kwargs):
        raise AssertionError("Retired _ResidentMinI64Plan pilot was entered")

    monkeypatch.setattr(blosc2.blosc2_ext, "_ResidentMinI64Plan", forbid_i64_pilot, raising=False)
    monkeypatch.setattr(blosc2.blosc2_ext, "_ResidentMapMinI64Plan", forbid_i64_pilot, raising=False)

    data = np.arange(1024, dtype=np.int64)[::-1].copy()
    assert blosc2.lazyexpr("min(x)", {"x": data}).compute() == 0

    bdata = blosc2.asarray(np.arange(1024, dtype=np.float64), chunks=(256,), blocks=(128,))

    @blosc2.menudet(cardinality="block_scalar", partitions=(256,))
    def my_sum(x):
        return block_sum(x)  # noqa: F821

    def forbid_kernel_pilot(*args, **kwargs):
        raise AssertionError("Retired _evaluate_b2nd_kernel_pilot was entered")

    monkeypatch.setattr(blosc2.NDArray, "_evaluate_b2nd_kernel_pilot", forbid_kernel_pilot, raising=False)
    lazy = my_sum(bdata)
    np.testing.assert_allclose(
        lazy[:], [np.sum(np.arange(i, i + 256, dtype=np.float64)) for i in range(0, 1024, 256)]
    )
