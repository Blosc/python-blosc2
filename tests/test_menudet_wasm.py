"""Public-API qualification of the bridge-free Menudet JIT on wasm32."""

from __future__ import annotations

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(not getattr(blosc2, "IS_WASM", False), reason="wasm32 only")


def make(expression, dtype, *, inputs=None, output=None, jit=True):
    inputs = inputs or {"x": dtype, "y": dtype}
    source = f"def k({', '.join(inputs)}):\n    return {expression}\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        inputs, output or dtype, version="1.0", casting="unsafe"
    )
    return blosc2.PortableKernel.from_json(artifact, jit=jit)


def require_helpers():
    if not getattr(blosc2, "_WASM_MINIEXPR_ENABLED", False):
        pytest.skip("wasm JIT helpers are unavailable in this runtime")


def assert_same(actual, expected):
    assert np.array_equal(actual, expected, equal_nan=True)
    non_nan = ~np.isnan(expected)
    assert actual[non_nan].tobytes() == expected[non_nan].tobytes()


def test_exact_integer_boundaries_and_control_flow():
    require_helpers()
    x = np.array([-(2**63), 2**63 - 1, 2**53 + 1, -1, 0, 2**62], dtype="int64")
    y = np.array([-1, 1, 7, 2**63 - 1, 0, 3], dtype="int64")
    for expression in ("x + y", "x - y", "x * y", "-x"):
        accelerated = make(expression, "int64")
        reference = make(expression, "int64", jit=False)
        assert accelerated.has_jit
        assert not reference.has_jit
        assert_same(
            accelerated.evaluate({"x": x, "y": y}),
            reference.evaluate({"x": x, "y": y}),
        )

    ux = np.array([2**64 - 1, 2**63, 2**53 + 1, 1], dtype="uint64")
    uy = np.array([1, 7, 9, 2**64 - 1], dtype="uint64")
    for expression in ("x + y", "x - y", "x * y"):
        accelerated = make(expression, "uint64")
        assert accelerated.has_jit
        assert_same(
            accelerated.evaluate({"x": ux, "y": uy}),
            make(expression, "uint64", jit=False).evaluate({"x": ux, "y": uy}),
        )

    xf = np.array([-3.0, -0.0, 0.0, 1.0, 2.5, np.nan], dtype="float64")
    yf = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype="float64")
    for expression in (
        "x * 1.25 + y * 0.75",
        "where(x > 0.0, x + y, x - y)",
        "where(x != 0.0, where(y > 0.0, x / y, -x), y)",
    ):
        accelerated = make(expression, "float64")
        assert accelerated.has_jit
        assert_same(
            accelerated.evaluate({"x": xf, "y": yf}),
            make(expression, "float64", jit=False).evaluate({"x": xf, "y": yf}),
        )

    loop = "def k(x, y):\n    v = x\n    for i in range(16):\n        v = v + y\n    return v\n"
    artifact = blosc2.DSLKernel.from_source(loop).export(
        {"x": "float64", "y": "float64"}, "float64", version="1.0", casting="unsafe"
    )
    accelerated = blosc2.PortableKernel.from_json(artifact, jit=True)
    assert accelerated.has_jit
    assert_same(
        accelerated.evaluate({"x": xf, "y": yf}),
        blosc2.PortableKernel.from_json(artifact, jit=False).evaluate({"x": xf, "y": yf}),
    )


def test_masks_tiles_and_reuse():
    require_helpers()
    x = np.array([1.0, 100.0, -2.0, 4.0, 0.0, np.nan], dtype="float64")
    y = np.array([3.0, 100.0, 1.0, -1.0, 0.0, 1.0], dtype="float64")
    accelerated = make("x * 2.0 + y", "float64")
    reference = make("x * 2.0 + y", "float64", jit=False)
    assert accelerated.has_jit

    mask = np.array([1, 0, 1, 1, 0, 1], dtype=np.uint8)
    actual = accelerated.evaluate_block({"x": x, "y": y}, block_shape=x.shape, valid_mask=mask)
    expected = reference.evaluate_block({"x": x, "y": y}, block_shape=x.shape, valid_mask=mask)
    active = mask.astype(bool)
    assert_same(actual[active], expected[active])

    for tile in (1, 3, 17, 1024):
        assert_same(
            accelerated.evaluate_array({"x": x, "y": y}, tile_items=tile),
            reference.evaluate_array({"x": x, "y": y}, tile_items=tile),
        )

    # Reusing the same artifact should keep returning an executable JIT kernel.
    source = "def k(x, y):\n    return x * y + 1.0\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        {"x": "float64", "y": "float64"}, "float64", version="1.0", casting="unsafe"
    )
    for _ in range(8):
        kernel = blosc2.PortableKernel.from_json(artifact, jit=True)
        assert kernel.has_jit
        assert_same(kernel.evaluate({"x": x, "y": y}), x * y + 1.0)


def test_unaligned_views_are_normalized():
    require_helpers()
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    y = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
    raw_x = np.zeros(1 + x.nbytes, dtype=np.uint8)
    raw_y = np.zeros(1 + y.nbytes, dtype=np.uint8)
    raw_x[1:] = x.view(np.uint8)
    raw_y[1:] = y.view(np.uint8)
    unaligned_x = np.frombuffer(raw_x[1:], dtype=np.float64)
    unaligned_y = np.frombuffer(raw_y[1:], dtype=np.float64)
    assert not unaligned_x.flags.aligned
    assert not unaligned_y.flags.aligned

    accelerated = make("x * 2.0 + y", "float64")
    reference = make("x * 2.0 + y", "float64", jit=False)
    assert accelerated.has_jit
    assert_same(
        accelerated.evaluate({"x": unaligned_x, "y": unaligned_y}),
        reference.evaluate({"x": unaligned_x, "y": unaligned_y}),
    )


@pytest.mark.parametrize(
    ("expression", "dtype", "output", "x", "y"),
    [
        ("x + sin(y)", "float64", "float64", [1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]),
        ("isnan(x)", "float64", "bool", [1.0, np.nan, np.inf, -0.0], [0.0] * 4),
        ("int(x)", "float64", "int64", [1.5, -2.5, 0.0, 4.0], [0.0] * 4),
        (
            "x < -1",
            "uint64",
            "bool",
            [0, 1, 2**64 - 1, 2**63],
            [1, 1, 1, 1],
        ),
    ],
)
def test_unsupported_bridges_fall_back(expression, dtype, output, x, y):
    require_helpers()
    accelerated = make(expression, dtype, output=output)
    reference = make(expression, dtype, output=output, jit=False)
    assert not accelerated.has_jit
    bindings = {"x": np.array(x, dtype=dtype), "y": np.array(y, dtype=dtype)}
    assert_same(accelerated.evaluate(bindings), reference.evaluate(bindings))


@blosc2.menudet(require_jit=True)
def _decorated_affine(x: np.float64, y: np.float64) -> np.float64:
    return x * 1.25 + y * 0.75


_CAPTURE_SCALE = np.float64(1.5)


@blosc2.menudet(require_jit=True)
def _captured_affine(x: np.float64, y: np.float64) -> np.float64:
    return x * _CAPTURE_SCALE + y


def test_scalar_capture_through_decorator():
    require_helpers()
    x = np.arange(8, dtype=np.float64)
    y = np.ones(8, dtype=np.float64)
    kernel = _captured_affine.specialize(x, y)
    assert kernel.has_jit
    assert_same(_captured_affine(x, y)[:], x * _CAPTURE_SCALE + y)


def test_decorator_requires_actual_jit():
    require_helpers()
    x = np.arange(8, dtype=np.float64)
    y = np.ones(8, dtype=np.float64)
    kernel = _decorated_affine.specialize(x, y)
    assert kernel.has_jit
    assert_same(_decorated_affine(x, y)[:], x * 1.25 + y * 0.75)


def test_nd_context_falls_back():
    require_helpers()
    source = "def k(x, y):\n    return x + _flat_idx + y\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        {"x": "float64", "y": "float64"}, "float64", version="1.0", casting="unsafe", ndim=2
    )
    accelerated = blosc2.PortableKernel.from_json(artifact, jit=True)
    reference = blosc2.PortableKernel.from_json(artifact, jit=False)
    assert not accelerated.has_jit
    x = np.arange(4, dtype=np.float64).reshape(2, 2)
    y = np.ones((2, 2), dtype=np.float64)
    actual = accelerated.evaluate_block(
        {"x": x, "y": y}, block_shape=(2, 2), logical_shape=(2, 2), block_origin=(0, 0)
    )
    expected = reference.evaluate_block(
        {"x": x, "y": y}, block_shape=(2, 2), logical_shape=(2, 2), block_origin=(0, 0)
    )
    assert_same(actual, expected)


def test_reduction_falls_back():
    require_helpers()
    source = "def k(x):\n    return block_sum(x)\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        {"x": "float64"}, "float64", version="1.0", casting="unsafe", cardinality="block_scalar"
    )
    accelerated = blosc2.PortableKernel.from_json(artifact, jit=True)
    reference = blosc2.PortableKernel.from_json(artifact, jit=False)
    assert not accelerated.has_jit
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype="float64")
    assert_same(accelerated.evaluate({"x": x}), reference.evaluate({"x": x}))
