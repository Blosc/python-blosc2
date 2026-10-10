"""Single-return scalar execution preserves the general DSL interpreter contract."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


def scalar_kernel(body, inputs, output="float64", *, jit=False, ndim=0, cardinality="block_scalar"):
    arguments = ", ".join(inputs)
    artifact = blosc2.DSLKernel.from_source(f"def k({arguments}):\n    {body}\n").export(
        inputs, output, version="1.1", cardinality=cardinality, ndim=ndim
    )
    return blosc2.PortableKernel.from_json(artifact, jit=jit)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
@pytest.mark.parametrize(
    "expression",
    [
        "block_sum(x + y)",
        "block_sum(x) + block_sum(y)",
        "block_sum(x) + block_sum(x)",
        "block_sum(x) / 2",
        "block_mean(x)",
        "block_sum(x) > block_sum(y)",
    ],
)
def test_simple_scalar_expression_matches_statement_path(dtype, expression, jit):
    output = (
        "bool"
        if ">" in expression
        else "float64"
        if dtype == "int64" and ("/" in expression or expression.startswith("block_mean"))
        else dtype
    )
    signature = {"x": dtype, "y": dtype}
    direct = scalar_kernel(f"return {expression}", signature, output, jit=jit)
    general = scalar_kernel(f"s = {expression}\n    return s", signature, output, jit=jit)
    x = np.array([1, -2, 3, 4], dtype=dtype)
    y = np.array([2, 1, -1, 0], dtype=dtype)
    inputs = {"y": y, "x": x}  # Runtime binding order is not expression variable order.
    for mask in (
        None,
        np.ones(4, dtype=bool),
        np.array([False, True, True, False]),
        np.zeros(4, dtype=bool),
    ):
        actual, status = direct.evaluate_block(inputs, valid_mask=mask, return_status=True)
        expected, reference = general.evaluate_block(inputs, valid_mask=mask, return_status=True)
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()
        assert status == reference
    empty = {name: np.empty(0, dtype=dtype) for name in signature}
    actual, status = direct.evaluate_block(empty, return_status=True)
    expected, reference = general.evaluate_block(empty, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference


@pytest.mark.parametrize("output", ["float32", "float64", "int64"])
def test_simple_scalar_output_conversion(output):
    signature = {"x": "int64"}
    direct = scalar_kernel("return block_sum(x)", signature, output)
    general = scalar_kernel("s = block_sum(x)\n    return s", signature, output)
    inputs = {"x": np.array([1, 2, 3], dtype="int64")}
    actual = direct.evaluate_block(inputs)
    assert actual.dtype == np.dtype(output)
    assert actual.tobytes() == general.evaluate_block(inputs).tobytes()


def test_simple_scalar_errors_are_not_retried_or_cached():
    kernel = scalar_kernel("return block_sum(x)", {"x": "int64"}, "int64")
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": np.array([2**63 - 1, 1], dtype="int64")})
    assert kernel.evaluate_block({"x": np.array([1, 2], dtype="int64")}) == 3
    for inputs in (
        {},
        {"x": np.array([1], dtype="float64")},
        {"x": np.array([1], dtype="int64"), "y": np.array([1])},
    ):
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block(inputs)


def test_simple_scalar_fp_errors_and_inactive_lanes():
    kernel = scalar_kernel("return block_prod(x)", {"x": "float64"})
    x = np.array([np.inf, 0, 2], dtype="float64")
    actual, status = kernel.evaluate_block({"x": x}, return_status=True)
    assert np.isnan(actual)
    assert status["flags"] & 1
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": x}, fp_errors="raise")
    actual, status = kernel.evaluate_block(
        {"x": x}, valid_mask=np.array([False, False, True]), return_status=True, fp_errors="raise"
    )
    assert actual == 2
    assert status["flags"] == 0


def test_simple_scalar_context_and_elementwise_fallback():
    context = scalar_kernel("return block_sum(_flat_idx)", {}, "int64", ndim=1)
    assert context.evaluate_block({}, block_shape=(3,), logical_shape=(8,), block_origin=(2,)) == 9
    with pytest.raises((ValueError, blosc2.PortableArtifactError)):
        context.evaluate_block({}, block_shape=(3,))
    elementwise = scalar_kernel("return x * 2", {"x": "float64"}, cardinality="elementwise")
    x = np.array([1, 2, 3], dtype="float64")
    np.testing.assert_array_equal(elementwise.evaluate_block({"x": x}), x * 2)


def test_simple_scalar_statement_control_flow_fallback():
    kernel = scalar_kernel(
        "s = block_sum(x)\n    if s > 0:\n        return s + 1\n    return s - 1", {"x": "int64"}, "int64"
    )
    assert kernel.evaluate_block({"x": np.array([1, 2], dtype="int64")}) == 4
    assert kernel.evaluate_block({"x": np.array([-1, -2], dtype="int64")}) == -4
    uninitialized = scalar_kernel(
        "if block_all(x > 0):\n        s = 3\n    return s",
        {"x": "int64"},
        "int64",
        cardinality="elementwise",
    )
    np.testing.assert_array_equal(
        uninitialized.evaluate_block({"x": np.array([1, 2], dtype="int64")}), [3, 3]
    )
    with pytest.raises(blosc2.PortableArtifactError):
        uninitialized.evaluate_block({"x": np.array([-1, 2], dtype="int64")})


def test_simple_scalar_concurrent_reuse():
    kernel = scalar_kernel("return block_sum(x) + block_sum(x)", {"x": "float64"})

    def evaluate(value):
        return kernel.evaluate_block({"x": np.full(1024, value, dtype="float64")})

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, range(16)))
    np.testing.assert_array_equal(results, np.arange(16) * 2048)
