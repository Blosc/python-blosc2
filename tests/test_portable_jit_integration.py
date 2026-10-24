"""Small production-binding gate; exhaustive lowering parity belongs to miniexpr."""

import os

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


def make_kernel(source, signature, output, **kwargs):
    artifact = blosc2.DSLKernel.from_source(source).export(signature, output, version="1.0", **kwargs)
    kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
    if os.environ.get("MENUDET_REQUIRE_JIT") == "1":
        assert kernel.has_jit
    return kernel


def test_capture_binding_roundtrip_and_masked_error_recovery():
    source = "def k(x, c):\n    return x + int8(c)\n"
    for capture in (1.75, 7.75, 1.75):
        kernel = make_kernel(source, {"x": "int8"}, "int8", constants={"c": capture})
        restored = blosc2.PortableKernel.from_json(kernel.to_json(), jit=True)
        np.testing.assert_array_equal(
            restored.evaluate_block({"x": np.array([0, 1], dtype="int8")}), [int(capture), int(capture) + 1]
        )
    kernel = make_kernel("def k(x):\n    return int64(x)\n", {"x": "float64"}, "int64")
    x = np.array([2.75, np.nan])
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": x})
    values = kernel.evaluate_block({"x": x}, valid_mask=np.array([True, False]))
    assert values.dtype == np.dtype("int64")
    assert values[0] == 2
    np.testing.assert_array_equal(kernel.evaluate_block({"x": np.array([1.0, 2.0])}), [1, 2])


@pytest.mark.parametrize(
    ("body", "dtype", "output"),
    [
        ("return block_sum(x * 2)", "float32", "float32"),
        ("return block_sum(x) + block_prod(x)", "int32", "int64"),
        ("return block_mean(x * 2)", "int32", "float64"),
        ("s = block_sum(x)\n    for i in range(3):\n        s = s + i\n    return s", "int32", "int64"),
        (
            "s = block_mean(x)\n    j = 0\n    while j < 3:\n        j = j + 1\n        s = s + 1\n    return s",
            "float64",
            "float64",
        ),
        ("v = x * 2\n    return block_sum(v) + block_mean(v)", "int32", "float64"),
        (
            "s = block_sum(x)\n    if s > 0:\n        return block_mean(x)\n    return s",
            "float64",
            "float64",
        ),
    ],
)
def test_scalar_partition_dtype_and_status_binding(body, dtype, output):
    kernel = make_kernel(f"def k(x):\n    {body}\n", {"x": dtype}, output, cardinality="block_scalar")
    reference = blosc2.PortableKernel.from_json(kernel.to_json(), jit=False)
    x = np.arange(1, 7, dtype=dtype)
    for mask in (None, np.arange(6) % 2 == 0, np.zeros(6, dtype=bool)):
        actual, status = kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True)
        expected, expected_status = reference.evaluate_block({"x": x}, valid_mask=mask, return_status=True)
        assert actual.dtype == np.dtype(output)
        np.testing.assert_array_equal(actual, expected)
        assert status == expected_status
    np.testing.assert_array_equal(
        kernel.lazy({"x": x}, partitions=(4,))[:], reference.lazy({"x": x}, partitions=(4,))[:]
    )


@pytest.mark.parametrize(
    "body", ["return block_mean(float64(_flat_idx))", "v = float64(_flat_idx)\n    return block_mean(v)"]
)
def test_logical_context_binding(body):
    source = f"def k(x):\n    {body}\n"
    kernel = make_kernel(source, {"x": "float64"}, "float64", cardinality="block_scalar", ndim=1)
    result = kernel.evaluate_block(
        {"x": np.ones(3)},
        logical_shape=(8,),
        block_shape=(3,),
        block_origin=(2,),
        valid_mask=np.array([True, False, True]),
    )
    assert result == 3


def test_scalar_loop_cap_is_invocation_local(monkeypatch):
    source = (
        "def k(x):\n    s = block_mean(x)\n    j = 0\n"
        "    while j < 5:\n        j = j + 1\n        s = s + 1\n    return s\n"
    )
    kernel = make_kernel(source, {"x": "int64"}, "float64", cardinality="block_scalar")
    x = np.array([1, 3], dtype="int64")
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "2")
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": x})
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "0")
    assert kernel.evaluate_block({"x": x}) == 7


def test_system_compiler_binding_smoke(monkeypatch):
    if os.name == "nt" or blosc2.IS_WASM:
        pytest.skip("System-compiler portable JIT is unsupported on this platform")
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc")
    kernel = make_kernel("def k(x):\n    return x + 1\n", {"x": "float64"}, "float64")
    if not kernel.has_jit:
        pytest.skip("System compiler unavailable")
    values, status = kernel.evaluate_block({"x": np.array([1.0, 2.0])}, return_status=True)
    np.testing.assert_array_equal(values, [2, 3])
    assert status["flags"] == 0


def test_tcc_arithmetic_bridge_binding(monkeypatch):
    if blosc2.IS_WASM:
        pytest.skip("Host arithmetic callbacks are unavailable on WASM")
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "tcc")
    x = np.arange(513, dtype=np.float64)
    y = np.arange(513, dtype=np.float64) + 1
    kernel = make_kernel(
        "def k(x, y):\n    return x + (y - 1)\n",
        {"x": "float64", "y": "float64"},
        "float64",
    )
    if not kernel.has_jit:
        pytest.skip("Native TCC JIT unavailable")
    expected = x + (y - 1)
    actual, status = kernel.evaluate_block({"x": x, "y": y}, return_status=True)
    np.testing.assert_array_equal(actual, expected)
    assert status["flags"] == 0

    monkeypatch.setenv("ME_DSL_JIT_ARITH_BRIDGE", "0")
    scalar = blosc2.PortableKernel.from_json(kernel.to_json(), jit=True)
    np.testing.assert_array_equal(scalar.evaluate_block({"x": x, "y": y}), expected)
