"""Portable JIT parity regressions use the production artifact and graph APIs."""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


def portable_kernel(source, inputs, output, *, jit=True, constants=None, cardinality="elementwise"):
    artifact = blosc2.DSLKernel.from_source(source).export(
        inputs, output, constants=constants, version="1.1", cardinality=cardinality
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": next(iter(inputs.values()))}, jit=jit)
    if jit and os.environ.get("MENUDET_REQUIRE_JIT") == "1":
        assert oracle.has_jit, "Requested backend must compile the eligibility oracle"
    if oracle.has_jit:
        assert kernel.has_jit
    return kernel


@pytest.mark.parametrize("operator", ["==", "!=", "<", "<=", ">", ">="])
@pytest.mark.parametrize("mixed", [False, True])
def test_exact_integer_comparisons(operator, mixed):
    signature = {"x": "int64", "y": "uint64" if mixed else "int64"}
    source = f"def k(x, y):\n    return x {operator} y\n"
    kernel = portable_kernel(source, signature, "bool")
    reference = portable_kernel(source, signature, "bool", jit=False)
    x = np.array([-(2**63), -1, 0, 2**53, 2**63 - 1], dtype="int64")
    y = np.array(
        [0, 2**64 - 1, 0, 2**53 + 1, 2**64 - 1] if mixed else [-(2**63), 0, -1, 2**53 + 1, 2**63 - 1],
        dtype=signature["y"],
    )
    actual = kernel.evaluate_block({"x": x, "y": y})
    np.testing.assert_array_equal(actual, reference.evaluate_block({"x": x, "y": y}))
    if not mixed and operator == "<":
        assert actual[3]


def test_checked_weak_intermediate_and_mask_recovery():
    source = "def k(x, y):\n    if x == 0:\n        return x\n    return x + (y + 1)\n"
    kernel = portable_kernel(source, {"x": "int64"}, "int64", constants={"y": 2**63 - 1})
    zeros = np.zeros(2, dtype="int64")
    np.testing.assert_array_equal(kernel.evaluate_block({"x": zeros}), zeros)
    active = np.array([0, 1], dtype="int64")
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": active})
    kernel.evaluate_block({"x": active}, valid_mask=np.array([True, False]))
    np.testing.assert_array_equal(kernel.evaluate_block({"x": zeros}), zeros)


@pytest.mark.parametrize("max_iter", [0, 1, 16, 64])
def test_mandelbrot_portable_jit(max_iter):
    # Same per-pixel algorithm as bench/ndarray/jit-dsl-mandelbrot.py.
    source = """def k(cr, ci, max_iter):
    zr = 0.0
    zi = 0.0
    n = 0
    for _i in range(max_iter):
        if zr * zr + zi * zi > 4.0:
            break
        new_zr = zr * zr - zi * zi + cr
        zi = 2 * zr * zi + ci
        zr = new_zr
        n = n + 1
    return n
"""
    signature = {"cr": "float64", "ci": "float64"}
    constants = {"max_iter": max_iter}
    kernel = portable_kernel(source, signature, "int64", constants=constants)
    reference = portable_kernel(source, signature, "int64", constants=constants, jit=False)
    cr, ci = np.meshgrid(np.linspace(-2, 1, 31), np.linspace(-1.5, 1.5, 23))
    inputs = {"cr": cr.ravel(), "ci": ci.ravel()}
    actual = kernel.evaluate_block(inputs)
    expected = reference.evaluate_block(inputs)
    np.testing.assert_array_equal(actual, expected)
    zr = np.zeros_like(cr)
    zi = np.zeros_like(ci)
    n = np.zeros(cr.shape, dtype="int64")
    active = np.ones(cr.shape, dtype=bool)
    for _ in range(max_iter):
        active &= zr * zr + zi * zi <= 4.0
        new_zr = zr * zr - zi * zi + cr
        new_zi = 2 * zr * zi + ci
        zr = np.where(active, new_zr, zr)
        zi = np.where(active, new_zi, zi)
        n = np.where(active, n + 1, n)
    np.testing.assert_array_equal(actual.reshape(cr.shape), n)
    restored = blosc2.PortableKernel.from_json(kernel.to_json(), jit=True)
    np.testing.assert_array_equal(restored.evaluate_block(inputs), expected)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda _: kernel.evaluate_block(inputs), range(8)):
            np.testing.assert_array_equal(result, expected)


def test_while_limit_is_invocation_local(monkeypatch):
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    kernel = portable_kernel(
        "def k(x):\n    i = 0\n    while i < x:\n        i = i + 1\n    return i\n",
        {"x": "int64"},
        "int64",
    )
    values = np.array([0, 3], dtype="int64")
    np.testing.assert_array_equal(kernel.evaluate_block({"x": values}), values)
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "2")
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": values})
    kernel.evaluate_block({"x": values}, valid_mask=np.array([True, False]))
    monkeypatch.setenv("ME_DSL_WHILE_MAX_ITERS", "3")
    np.testing.assert_array_equal(kernel.evaluate_block({"x": values}), values)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("cast", ["int", "int8", "uint8", "int32", "uint32", "int64", "uint64"])
def test_checked_cast_boundaries(dtype, cast):
    output = "int64" if cast == "int" else cast
    source = f"def k(x, enabled):\n    if enabled:\n        return {cast}(x)\n    return 0\n"
    signature = {"x": dtype, "enabled": "bool"}
    kernel = portable_kernel(source, signature, output)
    reference = portable_kernel(source, signature, output, jit=False)
    limits = np.iinfo(output)
    edges = [0, -0.0, 1.75, -1.75, float(limits.min), float(limits.max), np.inf, -np.inf, np.nan]
    for value in edges:
        x = np.array([1, value], dtype=dtype)
        inputs = {"x": x, "enabled": np.ones(2, dtype=bool)}
        try:
            expected, expected_status = reference.evaluate_block(inputs, return_status=True)
        except blosc2.PortableArtifactError:
            with pytest.raises(blosc2.PortableArtifactError):
                kernel.evaluate_block(inputs)
        else:
            actual, status = kernel.evaluate_block(inputs, return_status=True)
            assert actual.tobytes() == expected.tobytes()
            assert status == expected_status
        kernel.evaluate_block(inputs, valid_mask=np.array([True, False]))
        inputs["enabled"][1] = False
        assert kernel.evaluate_block(inputs)[1] == 0


@pytest.mark.parametrize("dtype", ["int8", "int32", "int64", "uint64"])
@pytest.mark.parametrize(
    "function", ["abs", "sign", "square", "floor", "ceil", "trunc", "round", "real", "imag", "conj"]
)
def test_integer_builtins(dtype, function):
    source = f"def k(x):\n    return {function}(x)\n"
    kernel = portable_kernel(source, {"x": dtype}, dtype)
    reference = portable_kernel(source, {"x": dtype}, dtype, jit=False)
    limits = np.iinfo(dtype)
    inputs = {"x": np.array([limits.min, 0, 1, limits.max], dtype=dtype)}
    actual, status = kernel.evaluate_block(inputs, return_status=True)
    expected, reference_status = reference.evaluate_block(inputs, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference_status


@pytest.mark.parametrize("expression", ["fac(x)", "ncr(x, y)", "npr(x, y)", "pow(x, y)", "x ** y"])
def test_integer_combinatorial_math(expression):
    signature = {"x": "int32", "y": "int32"}
    source = f"def k(x, y):\n    return {expression}\n"
    kernel = portable_kernel(source, signature, "int32")
    reference = portable_kernel(source, signature, "int32", jit=False)
    inputs = {"x": np.array([0, 3, 5], dtype="int32"), "y": np.array([0, 1, 2], dtype="int32")}
    assert kernel.evaluate_block(inputs).tobytes() == reference.evaluate_block(inputs).tobytes()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("expression", ["ldexp(x, 2)", "fma(x, y, -1.0)"])
def test_multi_operand_math(dtype, expression):
    signature = {"x": dtype, "y": dtype}
    source = f"def k(x, y):\n    return {expression}\n"
    kernel = portable_kernel(source, signature, dtype)
    reference = portable_kernel(source, signature, dtype, jit=False)
    inputs = {
        "x": np.array([0, -0.0, 1 + 2**-23, np.inf, np.nan], dtype=dtype),
        "y": np.array([1, -1, 1 - 2**-23, 0, 1], dtype=dtype),
    }
    actual, status = kernel.evaluate_block(inputs, return_status=True)
    expected, reference_status = reference.evaluate_block(inputs, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference_status


@pytest.mark.parametrize("capture", [-130, -129, -1, 0, 126, 127, 2**63 - 1])
def test_computed_weak_capture_checks(capture):
    source = "def k(x, c):\n    if x == 0:\n        return x\n    return x + (c + 1)\n"
    constants = {"c": capture}
    kernel = portable_kernel(source, {"x": "int8"}, "int8", constants=constants)
    reference = portable_kernel(source, {"x": "int8"}, "int8", constants=constants, jit=False)
    inputs = {"x": np.array([0, 1], dtype="int8")}
    try:
        expected = reference.evaluate_block(inputs)
    except blosc2.PortableArtifactError:
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block(inputs)
    else:
        assert kernel.evaluate_block(inputs).tobytes() == expected.tobytes()
    kernel.evaluate_block(inputs, valid_mask=np.array([True, False]))
    np.testing.assert_array_equal(kernel.evaluate_block({"x": np.zeros(2, dtype="int8")}), [0, 0])


@pytest.mark.parametrize("capture", [1.75, -1.75, 127.0, 128.0, np.inf, np.nan])
def test_floating_weak_capture_conversion(capture):
    source = "def k(x, c):\n    if x == 0:\n        return x\n    return x + int8(c)\n"
    kernel = portable_kernel(source, {"x": "int8"}, "int8", constants={"c": capture})
    reference = portable_kernel(source, {"x": "int8"}, "int8", constants={"c": capture}, jit=False)
    inputs = {"x": np.array([0, 1], dtype="int8")}
    try:
        expected = reference.evaluate_block(inputs)
    except blosc2.PortableArtifactError:
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block(inputs)
    else:
        assert kernel.evaluate_block(inputs).tobytes() == expected.tobytes()
    kernel.evaluate_block(inputs, valid_mask=np.array([True, False]))


@pytest.mark.parametrize(
    "expression", ["where(x > 0, int(y), 0)", "x > 0 and int(y) > 0", "x <= 0 or int(y) > 0"]
)
def test_checked_expression_short_circuit(expression):
    output = "int64" if expression.startswith("where") else "bool"
    source = f"def k(x, y):\n    return {expression}\n"
    signature = {"x": "float64", "y": "float64"}
    kernel = portable_kernel(source, signature, output)
    reference = portable_kernel(source, signature, output, jit=False)
    inputs = {"x": np.array([-1, 1], dtype="float64"), "y": np.array([np.nan, 2], dtype="float64")}
    actual, status = kernel.evaluate_block(inputs, return_status=True)
    expected, reference_status = reference.evaluate_block(inputs, return_status=True)
    assert actual.tobytes() == expected.tobytes()
    assert status == reference_status


def test_computed_capture_cache_reuse():
    source = "def k(x, c):\n    return x + (c + 1)\n"
    kernels = [
        (portable_kernel(source, {"x": "int8"}, "int8", constants={"c": value}), value)
        for value in (1, 7, -3, 1)
    ]
    x = np.array([0, 1, 2], dtype="int8")
    for kernel, value in reversed(kernels):
        np.testing.assert_array_equal(kernel.evaluate_block({"x": x}), x + value + 1)
        restored = blosc2.PortableKernel.from_json(kernel.to_json(), jit=True)
        np.testing.assert_array_equal(restored.evaluate_block({"x": x}), x + value + 1)


@pytest.mark.parametrize("expression", ["c + d", "c - d", "c * d", "-c"])
@pytest.mark.parametrize(
    ("c", "d"),
    [
        (2**63 - 1, 1),
        (-(2**63), -1),
        (2**63 - 1, 2),
        (-(2**63), 0),
        (0, -(2**63)),
        (2**63 - 1, -1),
        (-(2**63), 1),
        (7, -3),
        (0, 0),
    ],
)
def test_inline_weak_arithmetic_boundaries(expression, c, d):
    source = f"def k(x, c, d):\n    if x == 0:\n        return x\n    return x + ({expression})\n"
    constants = {"c": c, "d": d}
    kernel = portable_kernel(source, {"x": "int64"}, "int64", constants=constants)
    reference = portable_kernel(source, {"x": "int64"}, "int64", constants=constants, jit=False)
    inputs = {"x": np.array([0, 1], dtype="int64")}
    try:
        expected = reference.evaluate_block(inputs)
    except blosc2.PortableArtifactError:
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block(inputs)
    else:
        assert kernel.evaluate_block(inputs).tobytes() == expected.tobytes()
    kernel.evaluate_block(inputs, valid_mask=np.array([True, False]))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("output", ["float32", "float64"])
@pytest.mark.parametrize(
    "expression",
    [
        "block_sum(x)",
        "block_prod(x)",
        "block_sum(x * 2)",
        "block_prod(x + y)",
        "block_sum(where(x >= 0, x, 0))",
        "block_sum(sqrt(x))",
    ],
)
def test_serial_floating_reduction_jit(dtype, output, expression):
    source = f"def k(x, y):\n    return {expression}\n"
    signature = {"x": dtype, "y": dtype}
    kernel = portable_kernel(source, signature, output, cardinality="block_scalar")
    reference = portable_kernel(source, signature, output, jit=False, cardinality="block_scalar")
    for values in ([1e16, 1, -1e16, 3], [np.inf, 1, -np.inf, 3], [1, np.nan, 2, 3], [-0.0, 1, 0, 3], []):
        x = np.array(values, dtype=dtype)
        inputs = {"x": x, "y": x}
        for mask in (None, np.arange(x.size) % 2 == 0, np.zeros(x.size, dtype=bool)):
            actual, status = kernel.evaluate_block(inputs, valid_mask=mask, return_status=True)
            expected, reference_status = reference.evaluate_block(
                inputs, valid_mask=mask, return_status=True
            )
            # NaN sign/payload propagation is backend-dependent; every other
            # result, including signed zero and serial rounding, remains exact.
            if not (np.isnan(actual).all() and np.isnan(expected).all()):
                assert actual.tobytes() == expected.tobytes()
            assert status == reference_status
    # The artifact's original partitions remain the reduction boundaries.
    x = np.array([1, 2, 3, 4, 5, 6], dtype=dtype)
    actual = kernel.lazy({"x": x, "y": x}, partitions=(4,))[:]
    expected = reference.lazy({"x": x, "y": x}, partitions=(4,))[:]
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("expression", ["block_sum(x)", "block_prod(x)"])
def test_checked_integer_reductions_keep_fallback(expression):
    source = f"def k(x):\n    return {expression}\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        {"x": "int64"}, "int64", version="1.1", cardinality="block_scalar"
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
    assert not kernel.has_jit
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": np.array([2**63 - 1, 2], dtype="int64")})
