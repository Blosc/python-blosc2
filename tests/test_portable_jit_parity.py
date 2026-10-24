"""Portable JIT parity regressions use the production artifact and graph APIs."""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

pytestmark = [pytest.mark.usefixtures("native_graph_runtime"), pytest.mark.native_semantics]


def portable_kernel(source, inputs, output, *, jit=True, constants=None, cardinality="elementwise"):
    artifact = blosc2.DSLKernel.from_source(source).export(
        inputs, output, constants=constants, version="1.0", cardinality=cardinality
    )
    kernel = blosc2.PortableKernel.from_json(artifact, jit=jit)
    # Probe backend availability with universally eligible arithmetic. Boolean
    # addition itself is not an eligible graph map, although Boolean reductions are.
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": "float64"}, jit=jit)
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
    # Pyodide's single-threaded runtime still qualifies values and round trips.
    if sys.platform != "emscripten":
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
def test_checked_integer_reductions_jit(expression):
    source = f"def k(x):\n    return {expression}\n"
    kernel = portable_kernel(source, {"x": "int64"}, "int64", cardinality="block_scalar")
    with pytest.raises(blosc2.PortableArtifactError):
        kernel.evaluate_block({"x": np.array([2**63 - 1, 2], dtype="int64")})


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("output", ["float32", "float64"])
@pytest.mark.parametrize(
    "body",
    [
        "return block_sum(x * 2) / 2",
        "return sqrt(block_sum(x))",
        "s = block_sum(x)\n    return s + 1",
        "a = 2.0\n    s = block_sum(x * a)\n    t = s / a\n    return t",
        "return block_mean(x)",
        "return block_mean(x * 2) / 2",
        "s = block_mean(x)\n    return s + 1",
        "s = block_prod(x + 1)\n    return s / 2",
    ],
)
def test_scalar_reduction_recipes(dtype, output, body):
    source = f"def k(x):\n    {body}\n"
    kernel = portable_kernel(source, {"x": dtype}, output, cardinality="block_scalar")
    reference = portable_kernel(source, {"x": dtype}, output, jit=False, cardinality="block_scalar")
    for values in (
        [1e16, 1, -1e16, 3],
        [np.inf, 1, -np.inf, 3],
        [1, np.nan, 2, 3],
        [-0.0, 1, 0, 3],
        [np.finfo(dtype).max, 2],
        [np.finfo(dtype).tiny, np.finfo(dtype).tiny],
        [],
    ):
        x = np.array(values, dtype=dtype)
        for mask in (None, np.arange(x.size) % 2 == 0, np.zeros(x.size, dtype=bool)):
            actual, status = kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True)
            expected, expected_status = reference.evaluate_block(
                {"x": x}, valid_mask=mask, return_status=True
            )
            assert actual.dtype == expected.dtype
            if not (np.isnan(actual).all() and np.isnan(expected).all()):
                assert actual.tobytes() == expected.tobytes()
            assert status == expected_status
    x = np.arange(1, 7, dtype=dtype)
    np.testing.assert_array_equal(
        kernel.lazy({"x": x}, partitions=(4,))[:], reference.lazy({"x": x}, partitions=(4,))[:]
    )


@pytest.mark.parametrize("first", [True, False])
def test_scalar_checked_operand_order_and_recovery(first):
    cast = "int64(c)"
    reduction = "block_sum(sqrt(x))"
    expression = f"{cast} + {reduction}" if first else f"{reduction} + {cast}"
    source = f"def k(x, c):\n    return {expression}\n"
    kernels = [
        portable_kernel(
            source,
            {"x": "float64"},
            "float64",
            jit=jit,
            constants={"c": 1e100},
            cardinality="block_scalar",
        )
        for jit in (True, False)
    ]
    errors = []
    for kernel in kernels:
        with pytest.raises(blosc2.PortableArtifactError) as failure:
            kernel.evaluate_block({"x": np.array([-1.0, 4.0])})
        errors.append(failure.value)
    assert errors[0].status == errors[1].status
    # Checked status is invocation-local; subsequent calls still report the error.
    with pytest.raises(blosc2.PortableArtifactError) as repeated:
        kernels[0].evaluate_block({"x": np.array([1.0, 4.0])})
    assert repeated.value.status == errors[0].status


@pytest.mark.parametrize("cast", ["int8", "uint8", "int64", "uint64"])
def test_scalar_final_checked_cast_and_recovery(cast):
    source = f"def k(x):\n    s = block_sum(x)\n    return {cast}(s)\n"
    kernel = portable_kernel(source, {"x": "float64"}, cast, cardinality="block_scalar")
    reference = portable_kernel(source, {"x": "float64"}, cast, jit=False, cardinality="block_scalar")
    for value in (np.nan, np.inf, 1e100):
        errors = []
        for k in (kernel, reference):
            with pytest.raises(blosc2.PortableArtifactError) as failure:
                k.evaluate_block({"x": np.array([value])})
            errors.append(failure.value)
        assert errors[0].status == errors[1].status
    assert kernel.evaluate_block({"x": np.array([1.75, 2.0])}) == 3


@pytest.mark.parametrize("select", [True, False])
def test_scalar_lazy_reduction_with_capture(select):
    source = "def k(x, select):\n    return where(select, block_mean(sqrt(x)), 7.0)\n"
    kernel = portable_kernel(
        source, {"x": "float64"}, "float64", constants={"select": select}, cardinality="block_scalar"
    )
    reference = portable_kernel(
        source,
        {"x": "float64"},
        "float64",
        jit=False,
        constants={"select": select},
        cardinality="block_scalar",
    )
    for x in (np.array([-1.0, 4.0]), np.array([], dtype="float64")):
        for mask in (None, np.zeros(x.size, dtype=bool)):
            actual, status = kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True)
            expected, expected_status = reference.evaluate_block(
                {"x": x}, valid_mask=mask, return_status=True
            )
            np.testing.assert_array_equal(actual, expected)
            assert status == expected_status
            if not select:
                assert actual == 7
                assert status["flags"] == 0


@pytest.mark.parametrize(
    "body",
    [
        "s = block_sum(x)\n    if s > 0:\n        v = x\n    else:\n        v = x * 2\n    return block_sum(v)",
        "v = x\n    for i in range(3):\n        v = v * 2\n    return block_sum(v)",
    ],
)
def test_scalar_recipes_outside_scope_keep_fallback(body):
    artifact = blosc2.DSLKernel.from_source(f"def k(x):\n    {body}\n").export(
        {"x": "float64"}, "float64", version="1.0", cardinality="block_scalar"
    )
    assert not blosc2.PortableKernel.from_json(artifact, jit=True).has_jit


def test_scalar_mean_logical_context():
    source = "def k(x):\n    s = block_mean(float64(_flat_idx))\n    return s + 1\n"
    artifact = blosc2.DSLKernel.from_source(source).export(
        {"x": "float64"}, "float64", version="1.0", cardinality="block_scalar", ndim=1
    )
    kernels = [blosc2.PortableKernel.from_json(artifact, jit=jit) for jit in (True, False)]
    oracle = blosc2.NativeGraph.from_expression("x + x", {"x": "float64"}, jit=True)
    assert kernels[0].has_jit == oracle.has_jit
    for mask in (None, np.array([True, False, True]), np.zeros(3, dtype=bool)):
        results = [
            kernel.evaluate_block(
                {"x": np.ones(3)},
                block_shape=(3,),
                logical_shape=(8,),
                block_origin=(2,),
                valid_mask=mask,
                return_status=True,
            )
            for kernel in kernels
        ]
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    "body",
    [
        "return block_sum(x) / block_sum(x * 2)",
        "a = block_sum(x)\n    b = block_prod(x)\n    return a + b",
        "return block_sum(x) + block_sum(x)",
        "s = block_sum(x)\n    if s > 0:\n        return block_mean(x)\n    return s",
        "s = block_sum(x)\n    if s > 0:\n        return block_max(x)\n    elif s < 0:\n        return block_min(x)\n    else:\n        return s",
        "s = block_sum(x)\n    if s < 0:\n        return s\n    else:\n        t = block_prod(x)\n    return t",
        "s = block_sum(x)\n    if s > 0:\n        t = block_prod(x)\n    else:\n        t = block_sum(x * 2)\n    return t",
        "return where(block_sum(x) > 0, block_mean(x), block_sum(x * 2))",
    ],
)
def test_multiple_reductions_and_scalar_flow(dtype, body):
    source = f"def k(x):\n    {body}\n"
    kernels = [
        portable_kernel(source, {"x": dtype}, dtype, jit=jit, cardinality="block_scalar")
        for jit in (True, False)
    ]
    for values in ([1, 2, 3], [-1, -2, -3], [0, 0, 0], [np.inf, -np.inf], [np.nan, 1], []):
        x = np.array(values, dtype=dtype)
        for mask in (None, np.arange(x.size) % 2 == 0, np.zeros(x.size, dtype=bool)):
            results, errors = [], []
            for kernel in kernels:
                try:
                    results.append(kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True))
                except blosc2.PortableArtifactError as error:
                    errors.append(error.status)
            if errors:
                assert len(errors) == 2
                assert errors[0] == errors[1]
                continue
            if not (np.isnan(results[0][0]).all() and np.isnan(results[1][0]).all()):
                assert results[0][0].tobytes() == results[1][0].tobytes()
            assert results[0][1] == results[1][1]


@pytest.mark.parametrize(
    "dtype", ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
)
@pytest.mark.parametrize("op", ["sum", "prod"])
@pytest.mark.parametrize("mapped", [False, True])
def test_integer_reduction_recipes(dtype, op, mapped):
    operand = "where(x != 0, x, x)" if mapped else "x"
    source = f"def k(x):\n    s = block_{op}({operand})\n    return s\n"
    output = "uint64" if dtype.startswith("uint") else "int64"
    kernels = [
        portable_kernel(source, {"x": dtype}, output, jit=jit, cardinality="block_scalar")
        for jit in (True, False)
    ]
    cases = [[0, 1, 1], [1, 2, 3], []]
    if dtype != "bool":
        limits = np.iinfo(dtype)
        cases += [[limits.max, 2], [limits.min, -1] if dtype.startswith("int") else [limits.max, 1]]
    for values in cases:
        x = np.array(values, dtype=dtype)
        for mask in (None, np.arange(x.size) % 2 == 0, np.zeros(x.size, dtype=bool)):
            results, errors = [], []
            for kernel in kernels:
                try:
                    results.append(kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True))
                except blosc2.PortableArtifactError as error:
                    errors.append(error.status)
            if errors:
                assert len(errors) == 2
                assert errors[0] == errors[1]
            else:
                assert results[0][0].tobytes() == results[1][0].tobytes()
                assert results[0][1] == results[1][1]
    for kernel in kernels:
        assert kernel.evaluate_block({"x": np.array([1, 1], dtype=dtype)}) == (2 if op == "sum" else 1)


@pytest.mark.parametrize("dtype", ["float32", "float64", "uint32", "uint64"])
def test_checked_range_bounds(dtype):
    source = "def k(x):\n    s = 0\n    for i in range(x):\n        s = s + i\n    return s\n"
    kernels = [portable_kernel(source, {"x": dtype}, "int64", jit=jit) for jit in (True, False)]
    x = np.array([0, 1, 3, 4], dtype=dtype)
    for kernel in kernels:
        np.testing.assert_array_equal(kernel.evaluate_block({"x": x}), [0, 0, 3, 6])
    invalid = (
        [2**63, 2**64 - 1]
        if dtype == "uint64"
        else ([np.nan, np.inf, -np.inf, float(2**63)] if dtype.startswith("float") else [])
    )
    for value in invalid:
        for kernel in kernels:
            with pytest.raises(blosc2.PortableArtifactError):
                kernel.evaluate_block({"x": np.array([value], dtype=dtype)})
            np.testing.assert_array_equal(kernel.evaluate_block({"x": x}), [0, 0, 3, 6])
    if dtype.startswith("float"):
        for kernel in kernels:
            np.testing.assert_array_equal(
                kernel.evaluate_block({"x": np.array([3.9, -2.9], dtype=dtype)}), [3, 0]
            )


@pytest.mark.parametrize(
    "body",
    [
        "if x < 0:\n        return x\n    else:\n        y = x + 1\n    return y",
        "if x < 0:\n        return x\n    elif x == 0:\n        return x\n    else:\n        y = x + 1\n    return y",
        "for i in range(2):\n        if x < 0:\n            continue\n        else:\n            y = x + 1\n        return y\n    return x",
    ],
)
def test_definite_assignment_at_live_joins(body):
    source = f"def k(x):\n    {body}\n"
    kernels = [portable_kernel(source, {"x": "float64"}, "float64", jit=jit) for jit in (True, False)]
    x = np.array([-2.0, 0.0, 3.0])
    results = [kernel.evaluate_block({"x": x}, return_status=True) for kernel in kernels]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]


def test_scalar_elif_reduction_participation():
    source = (
        "def k(x):\n    s = block_sum(x)\n    if s > 0:\n        return s\n"
        "    elif block_sum(int64(x * 1e100)) > 0:\n        return s\n    return s\n"
    )
    kernels = [
        portable_kernel(source, {"x": "float64"}, "float64", jit=jit, cardinality="block_scalar")
        for jit in (True, False)
    ]
    for kernel in kernels:
        assert kernel.evaluate_block({"x": np.array([1.0, 2.0])}) == 3
        with pytest.raises(blosc2.PortableArtifactError):
            kernel.evaluate_block({"x": np.array([-1.0, -2.0])})
        assert kernel.evaluate_block({"x": np.array([1.0, 2.0])}) == 3


@pytest.mark.parametrize("op", ["sum", "prod"])
def test_integer_reduction_modular_map_checked_accumulator(op):
    source = f"def k(x):\n    return block_{op}(x + 1)\n"
    kernels = [
        portable_kernel(source, {"x": "int8"}, "int64", jit=jit, cardinality="block_scalar")
        for jit in (True, False)
    ]
    for kernel in kernels:
        assert kernel.evaluate_block({"x": np.array([127, 0], dtype="int8")}) == (
            -127 if op == "sum" else -128
        )


@pytest.mark.parametrize(
    "body",
    [
        "if x > 0:\n        return block_sum(x)\n    return block_sum(x)",
        "s = block_sum(x)\n    for i in range(2):\n        s = block_sum(x)\n    return s",
    ],
)
def test_reductions_in_varying_branches_or_loops_still_rejected(body):
    dsl = blosc2.DSLKernel.from_source(f"def k(x):\n    {body}\n")
    with pytest.raises(blosc2.PortableArtifactError, match="not supported inside"):
        dsl.export({"x": "float64"}, "float64", version="1.0", cardinality="block_scalar")


@pytest.mark.parametrize("bounds", ["x, 5.0, 1.0", "0.0, 5.0, x", "5.0, 0.0, x"])
def test_checked_range_start_and_step(bounds):
    source = f"def k(x):\n    s = 0\n    for i in range({bounds}):\n        s = s + i\n    return s\n"
    kernels = [portable_kernel(source, {"x": "float64"}, "int64", jit=jit) for jit in (True, False)]
    x = np.array([1.9, -1.9, 2.1])
    np.testing.assert_array_equal(kernels[0].evaluate_block({"x": x}), kernels[1].evaluate_block({"x": x}))
    for value in (np.nan, np.inf, 1e100, *([0.9] if bounds.endswith("x") else [])):
        for kernel in kernels:
            with pytest.raises(blosc2.PortableArtifactError):
                kernel.evaluate_block({"x": np.array([value])})
            # Nonparticipating invalid bounds never execute conversion/step checks.
            kernel.evaluate_block({"x": np.array([value, 1.0])}, valid_mask=np.array([False, True]))


def test_explicit_modular_range_cast_remains_explicit():
    source = "def k(x):\n    s = 0\n    for i in range(int64(x)):\n        s = s + i\n    return s\n"
    for jit in (True, False):
        kernel = portable_kernel(source, {"x": "uint64"}, "int64", jit=jit)
        assert kernel.evaluate_block({"x": np.array([2**64 - 1], dtype="uint64")}) == 0


def test_possible_empty_loop_local_still_falls_back():
    source = "def k(x):\n    for i in range(x):\n        y = i\n    return y\n"
    artifact = blosc2.DSLKernel.from_source(source).export({"x": "int64"}, "int64", version="1.0")
    assert not blosc2.PortableKernel.from_json(artifact, jit=True).has_jit


@pytest.mark.parametrize(
    "dtype",
    ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64"],
)
@pytest.mark.parametrize("op", ["min", "max", "any", "all"])
@pytest.mark.parametrize("mapped", [False, True])
def test_scalar_extrema_truth_recipes(dtype, op, mapped):
    operand = "where(x >= 0, x, x)" if mapped else "x"
    source = f"def k(x):\n    s = block_{op}({operand})\n    return s\n"
    output = "bool" if op in ("any", "all") else dtype
    kernel = portable_kernel(source, {"x": dtype}, output, cardinality="block_scalar")
    reference = portable_kernel(source, {"x": dtype}, output, jit=False, cardinality="block_scalar")
    if dtype.startswith("float"):
        cases = [[-0.0, 0.0, -0.0], [1, np.nan, 2, -np.nan], [np.inf, -np.inf], [0, 0], []]
        snan = np.array([1.0, 0.0, 2.0], dtype=dtype)
        snan.view("uint32" if dtype == "float32" else "uint64")[1] = (
            0x7F800001 if dtype == "float32" else 0x7FF0000000000001
        )
        cases.append(snan)
    elif dtype == "bool":
        cases = [[False, True, False], [False, False], [True, True], []]
    else:
        limits = np.iinfo(dtype)
        cases = [[limits.min, limits.max, 0], [0, 0], [1, 1], []]
    for values in cases:
        x = np.array(values, dtype=dtype)
        for mask in (None, np.arange(x.size) % 2 == 0, np.zeros(x.size, dtype=bool)):
            if op in ("min", "max") and (x.size == 0 or (mask is not None and not mask.any())):
                errors = []
                for k in (kernel, reference):
                    with pytest.raises(blosc2.PortableArtifactError) as failure:
                        k.evaluate_block({"x": x}, valid_mask=mask)
                    errors.append(failure.value)
                assert errors[0].status == errors[1].status
                continue
            actual, status = kernel.evaluate_block({"x": x}, valid_mask=mask, return_status=True)
            expected, expected_status = reference.evaluate_block(
                {"x": x}, valid_mask=mask, return_status=True
            )
            assert actual.dtype == expected.dtype
            if not (np.isnan(actual).all() and np.isnan(expected).all()):
                assert actual.tobytes() == expected.tobytes()
            assert status == expected_status
    x = np.array([1, 2, 0, 3, 0, 1], dtype=dtype)
    np.testing.assert_array_equal(
        kernel.lazy({"x": x}, partitions=(4,))[:], reference.lazy({"x": x}, partitions=(4,))[:]
    )


@pytest.mark.parametrize(("op", "values"), [("any", [1.0, -1.0]), ("all", [0.0, -1.0])])
def test_truth_reductions_do_not_skip_late_map_errors(op, values):
    source = f"def k(x):\n    return block_{op}(int64(x))\n"
    kernel = portable_kernel(source, {"x": "float64"}, "bool", cardinality="block_scalar")
    reference = portable_kernel(source, {"x": "float64"}, "bool", jit=False, cardinality="block_scalar")
    x = np.array([values[0], np.nan])
    for k in (kernel, reference):
        with pytest.raises(blosc2.PortableArtifactError):
            k.evaluate_block({"x": x})
        assert k.evaluate_block({"x": np.array(values)}) == (op == "any")


@pytest.mark.parametrize(("op", "values"), [("any", [1.0, -1.0]), ("all", [0.0, -1.0])])
def test_truth_reductions_keep_late_fp_status(op, values):
    source = f"def k(x):\n    return block_{op}(sqrt(x))\n"
    kernels = [
        portable_kernel(source, {"x": "float64"}, "bool", jit=jit, cardinality="block_scalar")
        for jit in (True, False)
    ]
    results = [k.evaluate_block({"x": np.array(values)}, return_status=True) for k in kernels]
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    if sys.platform != "emscripten":
        assert results[0][1]["flags"] & 1


@pytest.mark.parametrize("op", ["min", "max"])
def test_lazy_extrema_do_not_raise_for_unselected_empty_group(op):
    source = f"def k(x, select):\n    return where(select, block_{op}(x), 7.0)\n"
    for jit in (True, False):
        kernel = portable_kernel(
            source,
            {"x": "float64"},
            "float64",
            jit=jit,
            constants={"select": False},
            cardinality="block_scalar",
        )
        result, status = kernel.evaluate_block({"x": np.array([], dtype="float64")}, return_status=True)
        assert result == 7
        assert not status["flags"]


@pytest.mark.parametrize("op", ["min", "max", "any", "all"])
@pytest.mark.parametrize("capture", [0.0, -2.0])
def test_extrema_truth_uniform_capture(op, capture):
    source = f"def k(x, c):\n    return block_{op}(c)\n"
    output = "bool" if op in ("any", "all") else "float64"
    kernels = [
        portable_kernel(
            source,
            {"x": "float64"},
            output,
            jit=jit,
            constants={"c": capture},
            cardinality="block_scalar",
        )
        for jit in (True, False)
    ]
    for mask in (None, np.array([True, False, True]), np.zeros(3, dtype=bool)):
        if op in ("min", "max") and mask is not None and not mask.any():
            for kernel in kernels:
                with pytest.raises(blosc2.PortableArtifactError):
                    kernel.evaluate_block({"x": np.ones(3)}, valid_mask=mask)
            continue
        results = [
            kernel.evaluate_block({"x": np.ones(3)}, valid_mask=mask, return_status=True)
            for kernel in kernels
        ]
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
