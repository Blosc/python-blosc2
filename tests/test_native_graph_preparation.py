"""Native semantic preparation, independent of Python's numerical planner."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from blosc2.blosc2_ext import NativeGraphHandle

import blosc2
from blosc2.native_graph import lower_native_graph, scalar_descriptor

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


def graph(nodes, root=None, output="auto"):
    return json.dumps(
        {
            "format": "menudet-graph-1",
            "semantics": "menudet-numpy-1.1",
            "requires": ["numeric"],
            "nodes": [{"id": i, **node} for i, node in enumerate(nodes)],
            "root": len(nodes) - 1 if root is None else root,
            "output": {"dtype": output, "casting": "unsafe"},
        }
    ).encode()


def test_preparation_without_python_semantic_planning(monkeypatch):
    x = np.arange(12, dtype="float32").reshape(4, 3)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x * 2 + y", {"x": x, "y": np.ones((1, 3), dtype="float32")})

    def forbidden(*args, **kwargs):
        pytest.fail("Python semantic planning/source generation was called")

    from blosc2.dsl_kernel import DSLKernel

    monkeypatch.setattr(DSLKernel, "from_source", forbidden)
    monkeypatch.setattr(DSLKernel, "export", forbidden)
    monkeypatch.setattr(np, "broadcast_shapes", forbidden)
    monkeypatch.setattr(np, "result_type", forbidden)
    plan, arrays, reduction = lower_native_graph(expr)
    assert not reduction
    schedule = plan.specialize({name: (value.dtype, value.shape) for name, value in arrays.items()})
    assert schedule.info()["shape"] == (4, 3)
    result, _ = schedule.execute(arrays)
    np.testing.assert_array_equal(result, x * 2 + 1)


@pytest.mark.parametrize(
    "value", [np.float32(-0.0), np.float64(np.inf), np.int8(-128), np.uint64(2**64 - 1)]
)
def test_lossless_capture_and_roundtrip(value):
    encoded = graph([{"op": "constant", **scalar_descriptor(value)}])
    plan = NativeGraphHandle(encoded)
    imported = NativeGraphHandle(plan.to_json().encode())
    schedule = imported.specialize({})
    result, _ = schedule.execute({})
    assert result.dtype == value.dtype
    assert result.tobytes() == value.tobytes()


def test_weak_typed_and_zero_dimensional_strength():
    signatures = [{"op": "input", "name": "x", "dtype": "float32"}]
    weak = NativeGraphHandle(
        graph(
            signatures
            + [
                {"op": "constant", **scalar_descriptor(2.0)},
                {"op": "add", "args": [0, 1]},
            ]
        )
    )
    typed = NativeGraphHandle(
        graph(
            signatures
            + [
                {"op": "constant", **scalar_descriptor(np.float64(2))},
                {"op": "add", "args": [0, 1]},
            ]
        )
    )
    strong = NativeGraphHandle(
        graph(
            signatures
            + [
                {"op": "input", "name": "scalar", "dtype": "float64"},
                {"op": "add", "args": [0, 1]},
            ]
        )
    )
    assert weak.info()["inferred_dtype"] == np.dtype("float32")
    assert typed.info()["inferred_dtype"] == strong.info()["inferred_dtype"] == np.dtype("float64")
    schedule = strong.specialize({"x": ("float32", (3,)), "scalar": ("float64", ())})
    result, _ = schedule.execute({"x": np.ones(3, dtype="float32"), "scalar": np.array(4.0)})
    np.testing.assert_array_equal(result, [5, 5, 5])


def test_final_conversion_is_not_intermediate_precision():
    plan = NativeGraphHandle(
        graph(
            [
                {"op": "input", "name": "x", "dtype": "float32"},
                {"op": "constant", **scalar_descriptor(1.0)},
                {"op": "add", "args": [0, 1]},
                {"op": "sub", "args": [2, 0]},
            ],
            output="float64",
        )
    )
    assert plan.info()["inferred_dtype"] == np.dtype("float32")
    schedule = plan.specialize({"x": ("float32", (1,))})
    assert schedule.info()["dtype"] == np.dtype("float64")
    result, _ = schedule.execute({"x": np.array([2**24], dtype="float32")})
    assert result[0] == 0  # float32 add rounds before the final float64 cast.


def test_schedule_retains_plan_and_concurrent_invocations():
    plan = NativeGraphHandle(
        graph(
            [
                {"op": "input", "name": "x", "dtype": "int64"},
                {"op": "constant", **scalar_descriptor(2)},
                {"op": "mul", "args": [0, 1]},
            ]
        )
    )
    schedule = plan.specialize({"x": ("int64", (100,))})
    del plan

    def run(value):
        result, _ = schedule.execute({"x": np.full(100, value, dtype="int64")})
        np.testing.assert_array_equal(result, value * 2)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(16)))


def test_root_mask_initial_and_recovery():
    plan = NativeGraphHandle(
        graph(
            [
                {"op": "input", "name": "x", "dtype": "float64"},
                {"op": "function", "name": "sqrt", "args": [0]},
                {"op": "input", "name": "mask", "dtype": "bool"},
                {
                    "op": "sum",
                    "args": [1],
                    "axes": [-1],
                    "keepdims": True,
                    "dtype": "auto",
                    "initial": scalar_descriptor(2),
                    "where": 2,
                },
            ]
        )
    )
    schedule = plan.specialize({"x": ("float64", (2, 2)), "mask": ("bool", (2,))}, 1)
    x = np.array([[4.0, -1], [9, -1]])
    result, report = schedule.execute({"x": x, "mask": np.array([True, False])}, 1)
    np.testing.assert_array_equal(result, [[4], [5]])
    assert not report["fp_flags"]
    with pytest.raises(blosc2.PortableArtifactError, match="floating") as failure:
        schedule.execute({"x": x, "mask": np.array([True, True])}, 1)
    assert failure.value.fp_status["flags"] & 1
    result, report = schedule.execute({"x": x, "mask": np.array([True, False])}, 1)
    assert not report["fp_flags"]


def test_public_plan_and_root_initial_adapter():
    nodes = [{"op": "input", "name": "x", "dtype": "float32"}]
    plan = blosc2.NativeGraph.from_json(graph(nodes))
    assert plan.input_dtypes == {"x": np.dtype("float32")}
    values = np.arange(6, dtype="float32")
    np.testing.assert_array_equal(plan.evaluate({"x": values}), values)
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("sum(x)", {"x": values})
    assert expr.compute(_require_native=True, initial=2) == 17


def test_native_text_frontend_and_json_equivalence():
    text = blosc2.NativeGraph.from_expression("where(x != 0, y / x, y)", {"x": "float32", "y": "float32"})
    imported = blosc2.NativeGraph.from_json(text.to_json())
    values = {"x": np.array([0, 2, 4], dtype="float32"), "y": np.array(8, dtype="float32")}
    a, report = text.evaluate(values, return_report=True)
    b = imported.evaluate(values)
    assert a.dtype == b.dtype == np.dtype("float32")
    assert a.tobytes() == b.tobytes()
    np.testing.assert_array_equal(a, [8, 4, 2])
    assert not report["fp_flags"]


def test_explicit_conversion_budget_preflight():
    plan = blosc2.NativeGraph.from_json(
        graph([{"op": "input", "name": "x", "dtype": "float32"}], output="float64")
    )
    with pytest.raises(blosc2.PortableArtifactError, match="budget"):
        plan.specialize({"x": ("float32", (3,))}, intermediate_budget=4)
    schedule = plan.specialize({"x": ("float32", (3,))}, intermediate_budget=12)
    result, _ = schedule.execute({"x": np.ones(3, dtype="float32")})
    np.testing.assert_array_equal(result, [1, 1, 1])


def test_required_jit_fails_closed(monkeypatch):
    monkeypatch.setenv("CFLAGS", "-ffast-math")
    with pytest.raises(blosc2.PortableArtifactError, match="JIT"):
        blosc2.NativeGraph.from_expression("x + 1", {"x": "float32"}, require_jit=True)


@pytest.mark.parametrize("layout", ["c", "f", "reverse", "step", "unaligned", "swapped"])
def test_native_graph_layouts(layout):
    source = np.arange(24, dtype="float64").reshape(4, 6)
    if layout == "f":
        source = np.asfortranarray(source)
    elif layout == "reverse":
        source = source[::-1, ::-1]
    elif layout == "step":
        source = source[:, ::2]
    elif layout == "unaligned":
        storage = np.empty(source.nbytes + 1, dtype="uint8")
        unaligned = np.ndarray(source.shape, dtype=source.dtype, buffer=storage, offset=1)
        unaligned[:] = source
        source = unaligned
    elif layout == "swapped":
        source = source.astype(source.dtype.newbyteorder("S"))
    plan = blosc2.NativeGraph.from_expression("x * 2 + 1", {"x": "float64"})
    result, report = plan.evaluate({"x": source}, tile_items=5, return_report=True)
    np.testing.assert_array_equal(result, source * 2 + 1)
    assert report["normalization_bytes"] == 0


@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "any", "all"])
def test_reduction_shape_tile_and_roundtrip(op):
    source = np.arange(1, 13, dtype="int8").reshape(3, 4)
    plan = blosc2.NativeGraph.from_expression(f"{op}(x, axis=-1, keepdims=True)", {"x": "int8"})
    imported = blosc2.NativeGraph.from_json(plan.to_json())
    expected = getattr(np, op)(source, axis=-1, keepdims=True)
    for tile in [1, 3, 8]:
        actual = imported.evaluate({"x": source}, tile_items=tile)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tobytes() == expected.tobytes()


def test_empty_broadcast_and_axis_preflight():
    plan = blosc2.NativeGraph.from_expression("x + y", {"x": "float32", "y": "float32"})
    result = plan.evaluate({"x": np.empty((0, 3), dtype="float32"), "y": np.ones((1, 3), dtype="float32")})
    assert result.shape == (0, 3)
    for expression in ["sum(x, axis=[0, -2])", "sum(x, axis=2)"]:
        reduction = blosc2.NativeGraph.from_expression(expression, {"x": "float32"})
        with pytest.raises(blosc2.PortableArtifactError, match=r"axis|axes"):
            reduction.specialize({"x": ("float32", (2, 3))})


@pytest.mark.parametrize(("expression", "expected"), [("-2 ** 2", -4), ("2.0 ** -2", 0.25)])
def test_native_text_power_precedence(expression, expected):
    plan = blosc2.NativeGraph.from_expression(expression, {})
    assert plan.evaluate({}) == expected


def test_weak_initial_range_error_is_runtime_only():
    plan = blosc2.NativeGraph.from_expression("min(x, initial=-1)", {"x": "uint8"})
    schedule = plan.specialize({"x": ("uint8", (1,))})
    with pytest.raises(OverflowError):
        np.min(np.array([1], dtype="uint8"), initial=-1)
    with pytest.raises(blosc2.PortableArtifactError) as failure:
        schedule.execute({"x": np.array([1], dtype="uint8")})
    assert failure.value.status == "evaluation_error"
    assert failure.value.stage == 0


@pytest.mark.parametrize(
    "nodes",
    [
        [{"op": "input", "name": "x", "dtype": "float64"}, {"op": "add", "args": [0, 2]}],
        [{"op": "constant", **scalar_descriptor(1)}, {"op": "unknown", "args": [0]}],
        [{"op": "constant", **scalar_descriptor(1)}, {"op": "constant", **scalar_descriptor(2)}],
        [
            {"op": "input", "name": "x", "dtype": "float64"},
            {"op": "function", "name": "sqrt", "args": [0]},
            {"op": "add", "args": [1, 1]},
        ],
    ],
)
def test_invalid_graphs(nodes):
    with pytest.raises(blosc2.PortableArtifactError):
        NativeGraphHandle(graph(nodes))
