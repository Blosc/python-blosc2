"""Native semantic preparation, independent of Python's numerical planner."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from blosc2.blosc2_ext import NativeGraphHandle

import blosc2
from blosc2.native_graph import lower_native_graph, scalar_descriptor

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
@pytest.mark.parametrize("op", ["sum", "prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("tile", [1, 5, 64])
def test_direct_input_reduction_bypasses_map(dtype, op, tile):
    x = np.arange(1, 13, dtype=dtype).reshape(3, 4)
    plan = blosc2.NativeGraph.from_expression(f"{op}(x, axis=-1)", {"x": dtype}, jit=True)
    actual, report = plan.evaluate({"x": x}, tile_items=tile, return_report=True)
    np.testing.assert_array_equal(actual, getattr(np, op)(x, axis=-1))
    assert report["evaluated_tiles"] == 0
    assert report["temporary_bytes"] == 0
    assert report["jit_stages"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_direct_sum_order_status_initial_and_fallback(dtype):
    large = 2 ** (24 if dtype == "float32" else 53)
    plan = blosc2.NativeGraph.from_expression("sum(x, initial=2)", {"x": dtype})
    x = np.array([large, 1, -large, 3], dtype=dtype)
    # Serial accumulation includes initial before the first input, not after
    # a tile partial sum. Compute the expected rounding step by step.
    expected = np.dtype(dtype).type(2)
    for value in x:
        expected = np.dtype(dtype).type(expected + value)
    for tile in (1, 3, 64):
        actual, report = plan.evaluate({"x": x}, tile_items=tile, return_report=True)
        assert actual == expected
        assert report["evaluated_tiles"] == 0
    for values, flags in (([np.finfo(dtype).max, np.finfo(dtype).max], 4), ([np.inf, -np.inf], 1)):
        _, report = plan.evaluate({"x": np.array(values, dtype=dtype)}, return_report=True)
        assert report["fp_flags"] == flags
    reverse = x[::-1]
    actual, report = plan.evaluate({"x": reverse}, return_report=True)
    expected = np.dtype(dtype).type(2)
    for value in reverse:
        expected = np.dtype(dtype).type(expected + value)
    assert actual == expected
    assert report["evaluated_tiles"] > 0
    assert report["gathered_bytes"] > 0


def make_portable_sum(dtype, body="return block_sum(x)", cardinality="block_scalar", *, output_dtype=None):
    artifact = blosc2.DSLKernel.from_source(f"def k(x):\n    {body}\n").export(
        {"x": dtype}, output_dtype or dtype, version="1.1", cardinality=cardinality
    )
    return blosc2.PortableKernel.from_json(artifact, jit=False)


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float32", "float64"],
)
@pytest.mark.parametrize("op", ["prod", "min", "max", "any", "all"])
def test_typed_reductions_match_generic(dtype, op):
    x = np.array([1, 2, 0, 3, 1, 2], dtype=dtype)
    output_dtype = (
        "bool"
        if op in ("any", "all")
        else ("uint64" if dtype.startswith("uint") else "int64")
        if op == "prod" and dtype not in ("float32", "float64")
        else dtype
    )
    kernel = make_portable_sum(dtype, f"return block_{op}(x)", output_dtype=output_dtype)
    actual, status = kernel.evaluate_block({"x": x}, return_status=True)
    # A partial mask forces the original reducer, without altering input values.
    padded = np.append(x, np.array([0], dtype=dtype))
    mask = np.arange(padded.size) < x.size
    expected, reference = kernel.evaluate_block({"x": padded}, valid_mask=mask, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference
    np.testing.assert_array_equal(actual, getattr(np, op)(x))
    plan = blosc2.NativeGraph.from_expression(f"{op}(x)", {"x": dtype})
    strided = np.repeat(x, 2)[::2]
    for tile in (1, 4, 64):
        actual, report = plan.evaluate({"x": x}, tile_items=tile, return_report=True)
        expected, reference = plan.evaluate({"x": strided}, tile_items=tile, return_report=True)
        assert actual.tobytes() == expected.tobytes()
        assert report["fp_flags"] == reference["fp_flags"]
        assert report["evaluated_tiles"] == 0
        assert reference["evaluated_tiles"] > 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op", ["prod", "min", "max", "any", "all"])
@pytest.mark.parametrize("case", ["zero", "nan", "snan", "overflow", "underflow", "invalid", "empty"])
def test_typed_float_reduction_edge_parity(dtype, op, case):
    cases = {
        "zero": [-0.0, 0.0, -0.0],
        "nan": [1, np.nan, 2, -np.nan],
        "snan": [1, 2],
        "overflow": [np.finfo(dtype).max, 2],
        "underflow": [np.finfo(dtype).tiny, np.finfo(dtype).tiny],
        "invalid": [np.inf, 0],
        "empty": [],
    }
    x = np.array(cases[case], dtype=dtype)
    if case == "snan":
        x.view("uint32" if dtype == "float32" else "uint64")[1] = (
            0x7F800001 if dtype == "float32" else 0x7FF0000000000001
        )
    kernel = make_portable_sum(
        dtype, f"return block_{op}(x)", output_dtype="bool" if op in ("any", "all") else dtype
    )
    padded = np.append(x, np.array([0], dtype=dtype))
    mask = np.arange(padded.size) < x.size
    if not x.size and op in ("min", "max"):
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block({"x": x})
        return
    actual, status = kernel.evaluate_block({"x": x}, return_status=True)
    expected, reference = kernel.evaluate_block({"x": padded}, valid_mask=mask, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference
    expression = f"{op}(x, initial=1)" if op in ("min", "max") else f"{op}(x)"
    plan = blosc2.NativeGraph.from_expression(expression, {"x": dtype})
    for tile in (1, 3, 64):
        actual, report = plan.evaluate({"x": x}, tile_items=tile, return_report=True)
        expected, reference = plan.evaluate({"x": np.repeat(x, 2)[::2]}, tile_items=tile, return_report=True)
        assert actual.tobytes() == expected.tobytes()
        assert report["fp_flags"] == reference["fp_flags"]


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("int64", [2**63 - 1, 2, 0]),
        ("int64", [-(2**63), -1, 0]),
        ("uint64", [2**64 - 1, 2, 0]),
    ],
)
def test_typed_product_prefix_overflow(dtype, values):
    x = np.array(values, dtype=dtype)
    kernel = make_portable_sum(dtype, "return block_prod(x)")
    for data, mask in (
        (x, None),
        (np.append(x, np.array([1], dtype=dtype)), np.array([True, True, True, False])),
    ):
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block({"x": data}, valid_mask=mask)
    assert kernel.evaluate_block({"x": np.array([2, 3], dtype=dtype)}) == 6
    plan = blosc2.NativeGraph.from_expression("prod(x)", {"x": dtype})
    for tile in (1, 64):
        assert plan.evaluate({"x": x}, tile_items=tile) == 0


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"])
@pytest.mark.parametrize("op", ["min", "max"])
def test_typed_integer_extrema_bounds(dtype, op):
    limits = np.iinfo(dtype)
    x = np.array([limits.max, 0, limits.min, 1], dtype=dtype)
    kernel = make_portable_sum(dtype, f"return block_{op}(x)")
    assert kernel.evaluate_block({"x": x}) == getattr(np, op)(x)
    plan = blosc2.NativeGraph.from_expression(f"{op}(x, initial=1)", {"x": dtype})
    for tile in (1, 64):
        assert plan.evaluate({"x": x}, tile_items=tile) == getattr(np, op)(x, initial=1)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op", ["prod", "min", "max", "any", "all"])
def test_typed_reductions_initialized_locals_and_masks(dtype, op):
    output_dtype = "bool" if op in ("any", "all") else dtype
    x = np.array([-2, 0, 3, 4], dtype=dtype)
    local = make_portable_sum(dtype, f"y = x\n    return block_{op}(y)", output_dtype=output_dtype)
    assert local.evaluate_block({"x": x}) == getattr(np, op)(x)
    mask = np.array([False, False, True, True])
    assert local.evaluate_block({"x": x}, valid_mask=mask) == getattr(np, op)(x[mask])


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint64"])
def test_typed_product_initial_and_conversion_fallback(dtype):
    x = np.array([1, 2, 3, 4], dtype=dtype)
    plan = blosc2.NativeGraph.from_expression("prod(x, initial=3)", {"x": dtype})
    for tile in (1, 3, 64):
        assert plan.evaluate({"x": x}, tile_items=tile) == 72
    accumulator = ("float64" if dtype == "float32" else "float32") if dtype.startswith("float") else "int8"
    plan = blosc2.NativeGraph.from_expression(f"prod(x, dtype={accumulator}, initial=3)", {"x": dtype})
    x = np.array([12, 12, 12], dtype=dtype)
    result = plan.evaluate({"x": x})
    assert result.dtype == np.dtype(accumulator)
    assert result == np.prod(x, dtype=accumulator, initial=3)


@pytest.mark.parametrize(
    "dtype", ["bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]
)
@pytest.mark.parametrize("tile", [1, 5, 64])
def test_integer_bool_direct_sums(dtype, tile):
    output_dtype = "uint64" if dtype.startswith("uint") else "int64"
    values = [0, 1, 3, 7, 2, 0, 5] * 3
    if dtype.startswith("int"):
        values[2] = -3
        values[3] = -7
    x = np.array(values, dtype=dtype)
    direct = make_portable_sum(dtype, output_dtype=output_dtype)
    generic = make_portable_sum(dtype, "return block_sum(x + 0)", output_dtype=output_dtype)
    graph_plan = blosc2.NativeGraph.from_expression("sum(x)", {"x": dtype})
    expected = np.sum(x)
    for kernel in (direct, generic):
        result, status = kernel.evaluate_block({"x": x}, return_status=True)
        assert result.dtype == expected.dtype
        assert result == expected
        assert status["flags"] == 0
    result, report = graph_plan.evaluate({"x": x}, tile_items=tile, return_report=True)
    assert result.dtype == expected.dtype
    assert result == expected
    assert report["evaluated_tiles"] == report["temporary_bytes"] == 0
    mask = np.arange(x.size) % 2 == 0
    for kernel in (direct, generic):
        result = kernel.evaluate_block({"x": x}, valid_mask=mask)
        assert result == np.sum(x[mask])


@pytest.mark.parametrize(
    ("dtype", "values", "expected"),
    [
        ("int64", [2**63 - 1, 1], -(2**63)),
        ("int64", [-(2**63), -1], 2**63 - 1),
        ("uint64", [2**64 - 1, 1], 0),
    ],
)
def test_direct_integer_sum_preserves_route_overflow(dtype, values, expected):
    x = np.array(values, dtype=dtype)
    for body in ("return block_sum(x)", "return block_sum(x + 0)"):
        kernel = make_portable_sum(dtype, body)
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block({"x": x})
        # Failure must not poison subsequent use of the same immutable kernel.
        assert kernel.evaluate_block({"x": np.array([1, 2], dtype=dtype)}) == 3
    plan = blosc2.NativeGraph.from_expression("sum(x)", {"x": dtype})
    for tile in (1, 64):
        assert plan.evaluate({"x": x}, tile_items=tile) == expected


def test_direct_integer_sum_initial_and_nondefault_accumulator():
    x = np.array([True], dtype=bool)
    plan = blosc2.NativeGraph.from_expression("sum(x, initial=9223372036854775807)", {"x": "bool"})
    assert plan.evaluate({"x": x}) == -(2**63)
    x = np.array([120, 120, 120], dtype="int64")
    plan = blosc2.NativeGraph.from_expression("sum(x, dtype=int8)", {"x": "int64"})
    actual = plan.evaluate({"x": x})
    assert actual.dtype == np.dtype("int8")
    assert actual == np.sum(x, dtype="int8")


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("case", ["order", "overflow", "invalid", "nan", "zero", "empty"])
def test_dsl_direct_sum_matches_generic(dtype, case):
    large = 2 ** (24 if dtype == "float32" else 53)
    cases = {
        "order": [large, 1, -large, 3],
        "overflow": [np.finfo(dtype).max, np.finfo(dtype).max],
        "invalid": [np.inf, -np.inf],
        "nan": [np.nan, 1, 2],
        "zero": [-0.0, -0.0],
        "empty": [],
    }
    x = np.array(cases[case], dtype=dtype)
    direct = make_portable_sum(dtype)
    # Computed operands still execute the generic expression reduction loop.
    generic = make_portable_sum(dtype, "return block_sum(x * 1)")
    actual, status = direct.evaluate_block({"x": x}, return_status=True)
    expected, reference = generic.evaluate_block({"x": x}, return_status=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dsl_direct_sum_masks_and_initialized_locals(dtype):
    direct = make_portable_sum(dtype)
    x = np.array([np.inf, -np.inf, 2, 3], dtype=dtype)
    actual, status = direct.evaluate_block(
        {"x": x}, valid_mask=np.array([False, False, True, True]), return_status=True
    )
    assert actual == 5
    assert status["flags"] == 0
    local = make_portable_sum(dtype, "y = x\n    return block_sum(y)")
    actual = local.evaluate_block({"x": np.array([1, 2, 3], dtype=dtype)})
    assert actual == 6
    branch = make_portable_sum(dtype, "return where(x > 0, block_sum(x), 0)", "elementwise")
    actual = branch.evaluate_block({"x": np.array([-1, 2, 3, -4], dtype=dtype)})
    np.testing.assert_array_equal(actual, [0, 5, 5, 0])


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
