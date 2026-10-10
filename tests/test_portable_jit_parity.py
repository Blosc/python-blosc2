"""Portable JIT parity regressions use the production artifact and graph APIs."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.usefixtures("native_graph_runtime")


def portable_kernel(source, inputs, output, *, jit=True, constants=None):
    artifact = blosc2.DSLKernel.from_source(source).export(
        inputs, output, constants=constants, version="1.1"
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": next(iter(inputs.values()))}, jit=jit)
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
