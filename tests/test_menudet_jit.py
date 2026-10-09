"""Actual portable JIT execution, not accelerated-request interpreter fallback."""

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2


def make(expression, dtype, *, jit, output=None, inputs=None):
    inputs = inputs or {"x": dtype, "y": dtype}
    source = f"def k({', '.join(inputs)}):\n    return {expression}\n"
    author = blosc2.DSLKernel.from_source(source)
    artifact = author.export(inputs, output or dtype, version="1.1", casting="unsafe")
    return blosc2.PortableKernel.from_json(artifact, jit=jit)


@pytest.fixture(params=["tcc", "cc", "clang"])
def backend(request, monkeypatch, tmp_path):
    monkeypatch.setenv("ME_DSL_JIT_COMPILER", "cc" if request.param == "clang" else request.param)
    monkeypatch.setenv("ME_DSL_JIT_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CFLAGS", raising=False)
    monkeypatch.delenv("ME_DSL_JIT_TCC_OPTIONS", raising=False)
    if request.param == "cc":
        compiler = os.environ.get("MENUDET_GCC", shutil.which("gcc-16") or shutil.which("gcc") or "cc")
        monkeypatch.setenv("CC", compiler)
    elif request.param == "clang":
        compiler = os.environ.get("MENUDET_CLANG", shutil.which("clang"))
        if not compiler:
            pytest.skip("Clang is not installed")
        monkeypatch.setenv("CC", compiler)
    probe = make("x + y", "float64", jit=True)
    if not probe.has_jit:
        if os.environ.get("MENUDET_REQUIRE_JIT"):
            pytest.fail(f"{request.param} did not produce actual portable JIT")
        pytest.skip(f"Portable {request.param} JIT unavailable in this runtime")
    return request.param


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("expression", "boolean"),
    [
        ("x * 2 + y / 3 - 1", False),
        ("-x", False),
        ("(x + y) * (x - y)", False),
        ("x < y", True),
        ("x <= y", True),
        ("x > y", True),
        ("x >= y", True),
        ("x == y", True),
        ("x != y", True),
        ("x / y > 0", True),
        ("where(x > 0, y / x, x * 2 + y)", False),
        ("where(x != 0, where(y > 0, x / y, -x), y)", False),
    ],
)
def test_bitwise_values_and_status(backend, dtype, expression, boolean):
    accelerated = make(expression, dtype, jit=True, output="bool" if boolean else dtype)
    reference = make(expression, dtype, jit=False, output="bool" if boolean else dtype)
    assert accelerated.has_jit
    assert not reference.has_jit
    rng = np.random.Generator(np.random.PCG64(20261009))
    x = rng.normal(size=257).astype(dtype)
    y = rng.normal(size=257).astype(dtype)
    x[:9] = [0.0, -0.0, np.inf, -np.inf, np.nan, np.finfo(dtype).max, np.finfo(dtype).tiny, 1.0, -1.0]
    y[:9] = [
        0.0,
        0.0,
        np.inf,
        -np.inf,
        1.0,
        np.finfo(dtype).max,
        np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype)),
        3.0,
        3.0,
    ]
    actual, actual_status = accelerated.evaluate({"x": x, "y": y}, return_status=True)
    expected, expected_status = reference.evaluate({"x": x, "y": y}, return_status=True)
    np.testing.assert_array_equal(actual, expected)
    assert actual_status == expected_status
    finite = ~np.isnan(expected)
    assert actual[finite].tobytes() == expected[finite].tobytes()
    for tile in (1, 17, 1024):
        values = accelerated.evaluate_array({"x": x[::-1], "y": y[::-1]}, tile_items=tile)
        np.testing.assert_array_equal(values, expected[::-1])


def test_lazy_mask_raise_recovery_and_threads(backend):
    k = make("where(x != 0, y / x, y)", "float64", jit=True)
    assert k.has_jit
    bindings = {"x": np.array([0.0, 2.0]), "y": np.array([1.0, 4.0])}
    values, status = k.evaluate(bindings, return_status=True, fp_errors="raise")
    np.testing.assert_array_equal(values, [1.0, 2.0])
    assert status["flags"] == 0
    div = make("y / x", "float64", jit=True)
    _, report = div.evaluate_array(
        bindings, reduction="sum", where=np.array([False, True]), return_report=True
    )
    assert report["fp_flags"] == 0
    with pytest.raises(blosc2.PortableArtifactError) as error:
        div.evaluate(bindings, fp_errors="raise")
    assert error.value.fp_status["flags"] == 2
    values, status = div.evaluate({"x": np.ones(2), "y": np.ones(2)}, return_status=True)
    assert status["flags"] == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: k.evaluate(bindings, return_status=True), range(16)))
    assert all(status["flags"] == 0 for _, status in results)


def test_fail_closed_and_casts(backend, monkeypatch):
    assert not make("x + y", "int64", jit=True).has_jit
    assert not make("sin(x)", "float64", jit=True).has_jit
    assert not make("x // y", "float64", jit=True).has_jit
    k = make("x + y", "float32", jit=True, output="float64")
    assert k.has_jit
    x = np.array([1.0, 2.0**24], dtype="float32")
    np.testing.assert_array_equal(
        k.evaluate({"x": x, "y": np.ones(2, dtype="float32")}),
        (x + np.ones(2, dtype="float32")).astype("float64"),
    )
    monkeypatch.setenv("CFLAGS", "-ffast-math")
    assert not make("x + y", "float64", jit=True).has_jit


def test_native_graph_acceleration_and_reuse(backend):
    from blosc2.native_graph import accelerated_plan

    x = np.arange(32, dtype="float64")
    with blosc2.expression_evaluation("safe"):
        expr = blosc2.lazyexpr("x * 2 + 1", {"x": x})
    result = expr.compute(_require_native=True, jit=True)
    assert expr._native_execution_report["backend"] == "portable-jit"
    np.testing.assert_array_equal(result[:], x * 2 + 1)
    hits = accelerated_plan.cache_info().hits
    x[:] = 4
    np.testing.assert_array_equal(expr.compute(_require_native=True, jit=True)[:], x * 2 + 1)
    assert accelerated_plan.cache_info().hits > hits


def test_inactive_union_bits_do_not_raise(backend):
    # Finite double whose low float32 word looks like a signaling NaN. The
    # interpreter must not speculatively widen the inactive union member.
    x = np.array([0x3FF000007F800001], dtype="uint64").view("float64")
    for jit in (False, True):
        k = make("x < y", "float64", jit=jit, output="bool")
        _, status = k.evaluate({"x": x, "y": np.ones(1)}, return_status=True)
        assert status["flags"] == 0


def test_constant_operations_keep_runtime_diagnostics(backend):
    for expression in ("x + (1.0 / 0.0)", "where(x > 0, 1.0 / 0.0, x)"):
        compiled = make(expression, "float64", jit=True)
        reference = make(expression, "float64", jit=False)
        assert compiled.has_jit
        for value in (-1.0, 1.0):
            bindings = {"x": np.array([value]), "y": np.array([1.0])}
            actual, status = compiled.evaluate(bindings, return_status=True)
            expected, expected_status = reference.evaluate(bindings, return_status=True)
            np.testing.assert_array_equal(actual, expected)
            assert status == expected_status


def test_boolean_short_circuit_and_mixed_float_inputs(backend):
    inputs = {"x": "float32", "y": "float64"}
    k = make("where(x > 0, x + y, y)", "float64", jit=True, inputs=inputs)
    assert k.has_jit
    bindings = {"x": np.array([-1.0, 2.0], dtype="float32"), "y": np.array([3.0, 4.0])}
    np.testing.assert_array_equal(k.evaluate(bindings), [3.0, 6.0])
    for expr in ("(x == 0) or (y / x > 1)", "(x != 0) and (y / x > 1)"):
        k = make(expr, "float64", jit=True, output="bool")
        assert k.has_jit
        _, status = k.evaluate({"x": np.array([0.0]), "y": np.array([1.0])}, return_status=True)
        assert status["flags"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("op", ["==", "!=", "<", "<=", ">", ">="])
def test_inline_comparison_nan_payloads_and_status(backend, dtype, op):
    # Construct payloads without floating arithmetic that could quiet sNaNs.
    if dtype == "float32":
        bits = [0x7FC00001, 0xFFC01234, 0x7F800001, 0xFF800123]
        uint = "uint32"
    else:
        bits = [0x7FF8000000000001, 0xFFF8000000001234, 0x7FF0000000000001, 0xFFF0000000000123]
        uint = "uint64"
    nan = np.array(bits, dtype=uint).view(dtype)
    compiled = make(f"x {op} y", dtype, jit=True, output="bool")
    reference = make(f"x {op} y", dtype, jit=False, output="bool")
    assert compiled.has_jit
    for i in range(len(nan)):
        value = nan[i : i + 1]
        ordinary = np.array([0.0], dtype=dtype)
        for x, y in ((value, ordinary), (ordinary, value), (value, value)):
            bindings = {"x": x, "y": y}
            actual, status = compiled.evaluate(bindings, return_status=True)
            expected, expected_status = reference.evaluate(bindings, return_status=True)
            np.testing.assert_array_equal(actual, expected)
            assert status == expected_status
            # Caller recovery and selected-lane masks must not leak NaN flags.
            _, report = compiled.evaluate_array(
                bindings, reduction="any", where=np.array([False]), return_report=True
            )
            assert report["fp_flags"] == 0


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_inline_comparison_nan_in_unselected_branch(backend, dtype):
    uint = "uint32" if dtype == "float32" else "uint64"
    bits = 0x7F800001 if dtype == "float32" else 0x7FF0000000000001
    x = np.array([bits], dtype=uint).view(dtype)
    k = make("where(y != 0, x > 0, y > 0)", dtype, jit=True, output="bool")
    assert k.has_jit
    values, status = k.evaluate({"x": x, "y": np.zeros(1, dtype=dtype)}, return_status=True)
    np.testing.assert_array_equal(values, [False])
    assert status["flags"] == 0
