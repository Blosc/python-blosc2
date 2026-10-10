"""JIT specialization of immutable weak integers never relaxes range checks."""

import numpy as np
import pytest

import blosc2
from blosc2.native_graph import scalar_descriptor

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
def test_weak_integral_capture_boundaries(dtype):
    limits = np.iinfo(dtype)
    # Weak integer transport is signed int64, even next to uint64 arrays.
    values = (max(int(limits.min), -(2**63)), min(int(limits.max), 2**63 - 1), 1)
    x = np.array([0, 1, 2], dtype=dtype)
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": dtype}, jit=True)
    for value in values:
        expression = f"x + {value}"
        plan = blosc2.NativeGraph.from_expression(expression, {"x": dtype}, jit=True)
        reference = blosc2.NativeGraph.from_expression(expression, {"x": dtype}, jit=False)
        if oracle.has_jit:
            assert plan.has_jit
        for tile in (1, 64):
            result, report = plan.evaluate({"x": x}, tile_items=tile, return_report=True)
            expected = reference.evaluate({"x": x}, tile_items=tile)
            assert result.dtype == expected.dtype
            assert result.tobytes() == expected.tobytes()
            if oracle.has_jit:
                assert report["jit_stages"] == 1
        restored = blosc2.NativeGraph.from_json(plan.to_json(), jit=True)
        assert restored.evaluate({"x": x}).tobytes() == expected.tobytes()


@pytest.mark.parametrize(
    ("dtype", "expression"), [("int32", "x % 7"), ("int32", "x << 2"), ("int8", "int16(x) * 2")]
)
def test_graph_benchmark_integer_cases_jit(dtype, expression):
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": dtype}, jit=True)
    plan = blosc2.NativeGraph.from_expression(expression, {"x": dtype}, jit=True)
    reference = blosc2.NativeGraph.from_expression(expression, {"x": dtype}, jit=False)
    if oracle.has_jit:
        assert plan.has_jit
    x = np.array([-10, -3, 0, 5, 20], dtype=dtype)
    assert plan.evaluate({"x": x}).tobytes() == reference.evaluate({"x": x}).tobytes()


@pytest.mark.parametrize(("dtype", "value"), [("int8", 128), ("int8", -129), ("uint8", -1), ("uint8", 256)])
def test_out_of_range_weak_capture_keeps_lazy_fallback(dtype, value):
    plan = blosc2.NativeGraph.from_expression(f"where(x == 0, x, x + {value})", {"x": dtype}, jit=True)
    assert not plan.has_jit
    zero = np.zeros(3, dtype=dtype)
    np.testing.assert_array_equal(plan.evaluate({"x": zero}), zero)
    active = zero.copy()
    active[1] = 1
    with pytest.raises(blosc2.PortableArtifactError):
        plan.evaluate({"x": active})
    # No failure poisons the immutable plan or changes later branch participation.
    np.testing.assert_array_equal(plan.evaluate({"x": zero}), zero)


def test_bool_capture_is_checked_before_integer_specialization():
    record = {
        "format": "menudet-graph-1",
        "semantics": "menudet-numpy-1.1",
        "requires": ["numeric"],
        "nodes": [
            {"id": 0, "op": "input", "name": "x", "dtype": "int8"},
            {"id": 1, "op": "constant", **scalar_descriptor(True)},
            {"id": 2, "op": "add", "args": [0, 1]},
        ],
        "root": 2,
        "output": {"dtype": "auto", "casting": "unsafe"},
    }
    plan = blosc2.NativeGraph.from_json(record, jit=True)
    x = np.array([0, 1, 2], dtype="int8")
    np.testing.assert_array_equal(plan.evaluate({"x": x}), [1, 2, 3])


def test_capture_values_are_part_of_jit_cache_identity():
    source = blosc2.DSLKernel.from_source("def k(x, c):\n    return x % c\n")
    x = np.array([-10, -3, 0, 5, 20], dtype="int32")
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": "int32"}, jit=True)
    kernels = []
    for value in (7, 3, 11, 7):
        artifact = source.export({"x": "int32"}, "int32", constants={"c": value}, version="1.1")
        kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
        if oracle.has_jit:
            assert kernel.has_jit
        kernels.append((kernel, value))
    # Keep all kernels alive while compiling others and replay in reverse order.
    for kernel, value in reversed(kernels):
        np.testing.assert_array_equal(kernel.evaluate_block({"x": x}), x % value)
