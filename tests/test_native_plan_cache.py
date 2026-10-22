"""Prepared native-plan reuse stays bounded, operand-free and output-safe."""

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

import blosc2
from blosc2 import native_graph


def _plan_cache():
    return native_graph._native_plan_cache


def test_plan_cache_rebinds_same_signature(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    x = np.arange(1000, dtype="float64")
    out = np.empty_like(x)
    blosc2.evaluate("x * 2 + 1", local_dict={"x": x}, out=out)
    np.testing.assert_array_equal(out, x * 2 + 1)
    assert len(_plan_cache()) == 1
    plan, names, reduction = next(iter(_plan_cache().values()))
    assert len(names) == 1
    assert not reduction
    del plan

    y = x + 7
    blosc2.evaluate("x * 2 + 1", local_dict={"x": y}, out=out)
    np.testing.assert_array_equal(out, y * 2 + 1)
    assert len(_plan_cache()) == 1


def test_plan_cache_key_changes_with_dtype_and_shape(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    for value in (
        np.arange(12, dtype="float64"),
        np.arange(12, dtype="float32"),
        np.arange(13, dtype="float64"),
    ):
        blosc2.evaluate("x + 1", local_dict={"x": value})
    assert len(_plan_cache()) == 3


def test_plan_cache_does_not_retain_operands(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    x = np.arange(32, dtype="float64")
    reference = weakref.ref(x)
    blosc2.evaluate("x + 1", local_dict={"x": x})
    del x
    gc.collect()
    assert reference() is None
    assert len(_plan_cache()) == 1


def test_cached_plan_rejects_aliased_output(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    x = np.arange(32, dtype="float64")
    blosc2.evaluate("x + 1", local_dict={"x": x})
    assert len(_plan_cache()) == 1
    expected = x + 1
    blosc2.evaluate("x + 1", local_dict={"x": x}, out=x)
    np.testing.assert_array_equal(x, expected)


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("setting", ["ME_SIMD_MATH_BACKEND", "ME_DSL_FP_MODE"])
def test_math_policy_invalidates_preparation_cache(monkeypatch, jit, setting):
    monkeypatch.delenv(setting, raising=False)
    first = native_graph._graph_configuration(jit)
    operand = np.arange(16.0)
    expr = SimpleNamespace(expression="x + 1", operands={"x": operand})
    key = native_graph._native_plan_cache_key(expr, jit, {})
    monkeypatch.setenv(setting, "system" if setting == "ME_SIMD_MATH_BACKEND" else "strict")
    assert native_graph._graph_configuration(jit) != first
    assert native_graph._native_plan_cache_key(expr, jit, {}) != key


def test_cache_rejects_lookalike_operand_types():
    lookalike = type("NDArray", (), {})()
    expr = SimpleNamespace(expression="x + 1", operands={"x": lookalike})
    assert native_graph._native_plan_cache_key(expr, False, {}) is None


def test_cache_eviction_is_bounded(monkeypatch, native_graph_runtime):
    native_graph._native_plan_cache.clear()
    monkeypatch.setattr(native_graph, "_NATIVE_PLAN_CACHE_SIZE", 2)
    x = np.arange(32.0)
    inserted = []
    for constant in range(4):
        np.testing.assert_array_equal(blosc2.evaluate(f"x + {constant}", local_dict={"x": x}), x + constant)
        assert len(_plan_cache()) <= 2
        inserted.append(next(reversed(_plan_cache())))
    assert list(_plan_cache()) == inserted[-2:]


def test_cached_plan_rebinds_strided_inputs_and_outputs(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    base = np.arange(64.0)
    for x in (base[:32].copy(), base[::2], base[:32][::-1]):
        out = np.empty_like(x)[::-1]
        assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=out) is out
        np.testing.assert_array_equal(out, x + 1)
    assert len(_plan_cache()) == 1


@pytest.mark.skipif(blosc2.IS_WASM, reason="Host thread pools unavailable on WASM")
def test_cached_plan_concurrent_bindings(native_graph_runtime):
    native_graph._native_plan_cache.clear()
    blosc2.evaluate("x + 1", local_dict={"x": np.arange(4096.0)})

    def run(value):
        x = np.full(4096, value, dtype="float64")
        out = np.empty_like(x)
        assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=out) is out
        np.testing.assert_array_equal(out, value + 1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, range(24)))
    # Explicit output and allocating calls have separate invocation signatures.
    assert len(_plan_cache()) <= 2
