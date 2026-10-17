"""Cached preparation never retains operands, outputs or mutable execution state."""

import gc
import importlib
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

module = importlib.import_module("blosc2.lazyexpr")


@pytest.fixture(autouse=True)
def empty_recipe_cache():
    with module._evaluate_recipe_lock:
        module._evaluate_recipe_cache.clear()
    yield
    with module._evaluate_recipe_lock:
        module._evaluate_recipe_cache.clear()


@pytest.mark.parametrize("kind", ["numpy", "blosc2", "mixed"])
@pytest.mark.parametrize("mode", ["full", "safe"])
def test_evaluate_cache_rebinds_inputs_and_fresh_outputs(kind, mode, monkeypatch):
    def inputs(value):
        x = np.arange(257, dtype=np.float64) + value
        y = np.ones(257, dtype=np.float64)
        if kind != "numpy":
            x = blosc2.asarray(x, chunks=(128,), blocks=(32,))
        if kind == "blosc2":
            y = blosc2.asarray(y, chunks=(128,), blocks=(32,))
        return {"x": x, "y": y}

    with blosc2.expression_evaluation(mode):
        first_inputs = inputs(0)
        first = blosc2.evaluate("x + (y - 1)", local_dict=first_inputs)
        assert len(module._evaluate_recipe_cache) == 1

        def forbidden(*args, **kwargs):
            pytest.fail("Recipe hit repeated full LazyExpr construction")

        monkeypatch.setattr(blosc2.LazyExpr, "_new_expr", forbidden)
        second_inputs = inputs(10)
        second = blosc2.evaluate("x + (y - 1)", local_dict=second_inputs)
        assert second is not first
        assert not np.shares_memory(first, second)
        np.testing.assert_array_equal(first, np.arange(257))
        np.testing.assert_array_equal(second, np.arange(257) + 10)
        if kind == "numpy":
            second_inputs["x"][:] = 99
            np.testing.assert_array_equal(blosc2.evaluate("x + (y - 1)", local_dict=second_inputs), 99)


def test_evaluate_cache_signature_and_policy_changes():
    for dtype, shape in [("float32", (12,)), ("float64", (12,)), ("float64", (3, 4))]:
        x = np.arange(12, dtype=dtype).reshape(shape)
        result = blosc2.evaluate("x + 1", local_dict={"x": x})
        assert result.dtype == np.dtype(dtype)
        assert result.shape == shape
        np.testing.assert_array_equal(result, x + 1)
    assert len(module._evaluate_recipe_cache) == 3
    with blosc2.expression_evaluation("safe"):
        np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}), x + 1)
    assert len(module._evaluate_recipe_cache) == 4


def test_evaluate_cache_caller_and_explicit_namespace_resolution():
    x = np.arange(12, dtype=np.float64)
    np.testing.assert_array_equal(blosc2.evaluate("x + 1"), x + 1)
    x = np.arange(12, dtype=np.float64) + 10
    np.testing.assert_array_equal(blosc2.evaluate("x + 1"), x + 1)
    np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={}, global_dict={"x": x}), x + 1)
    assert len(module._evaluate_recipe_cache) == 1


@pytest.mark.parametrize("kind", ["numpy", "blosc2"])
def test_evaluate_cache_does_not_retain_arrays(kind):
    x = np.arange(257, dtype=np.float64)
    if kind == "blosc2":
        x = blosc2.asarray(x, chunks=(128,), blocks=(32,))
    reference = weakref.ref(x)
    output = blosc2.evaluate("x + 1", local_dict={"x": x})
    output_ref = weakref.ref(output)
    assert module._evaluate_recipe_cache
    del x, output
    gc.collect()
    assert reference() is None
    assert output_ref() is None


def test_evaluate_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(module, "_EVALUATE_RECIPE_CACHE_SIZE", 3)
    x = np.arange(12, dtype=np.float64)
    for value in range(6):
        np.testing.assert_array_equal(blosc2.evaluate(f"x + {value}", local_dict={"x": x}), x + value)
    assert len(module._evaluate_recipe_cache) == 3
    assert [key[0] for key in module._evaluate_recipe_cache] == ["x + 3", "x + 4", "x + 5"]


@pytest.mark.parametrize("mode", ["full", "safe"])
@pytest.mark.skipif(blosc2.IS_WASM, reason="Concurrent cache calls require host threads")
def test_evaluate_cache_concurrent_calls(mode):
    def run(value):
        x = np.arange(257, dtype=np.float64) + value
        with blosc2.expression_evaluation(mode):
            result = blosc2.evaluate("x + 1", local_dict={"x": x})
        np.testing.assert_array_equal(result, x + 1)
        return result

    # Exercise racing misses as well as repeated hits on an immutable recipe.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, range(24)))
    assert len(module._evaluate_recipe_cache) == 1
    assert len({id(result) for result in results}) == 24


def test_evaluate_cache_excludes_captures_options_and_aliases():
    x = np.arange(12, dtype=np.float64)
    np.testing.assert_array_equal(blosc2.evaluate("x + c", local_dict={"x": x, "c": 2}), x + 2)
    np.testing.assert_array_equal(blosc2.evaluate("x + c", local_dict={"x": x, "c": 3}), x + 3)
    np.testing.assert_array_equal(blosc2.evaluate("x + y", local_dict={"x": x, "y": x}), x + x)
    assert not module._evaluate_recipe_cache
    out = np.empty_like(x)
    assert blosc2.evaluate("x + 1", local_dict={"x": x}, out=out) is out
    np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}, order="C"), x + 1)
    assert len(module._evaluate_recipe_cache) == 1  # Explicit output is rebound, not cached.


def test_evaluate_cache_checks_current_array_layout():
    x = np.arange(12, dtype=np.float64)
    blosc2.evaluate("x + 1", local_dict={"x": x})
    x = np.arange(24, dtype=np.float64)[::2]
    np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": x}), x + 1)
    assert len(module._evaluate_recipe_cache) == 1


def test_evaluate_cache_storage_layout_changes():
    x = np.arange(257, dtype=np.float64)
    for chunks, blocks in [(128, 32), (100, 25)]:
        source = blosc2.asarray(x, chunks=(chunks,), blocks=(blocks,))
        np.testing.assert_array_equal(blosc2.evaluate("x + 1", local_dict={"x": source}), x + 1)
    assert len(module._evaluate_recipe_cache) == 2


def test_evaluate_cache_does_not_retain_dtype_metadata_owners():
    owner = np.arange(12)
    dtype = np.dtype("float64", metadata={"owner": owner})
    x = np.arange(12, dtype=dtype)
    reference = weakref.ref(owner)
    result = blosc2.evaluate("x + 1", local_dict={"x": x})
    np.testing.assert_array_equal(result, np.arange(12) + 1)
    assert not module._evaluate_recipe_cache
    del x, owner, dtype, result
    gc.collect()
    assert reference() is None


def test_evaluate_recipe_has_no_mutable_execution_state():
    x = np.arange(257, dtype=np.float64)
    first = module._prepare_cached_evaluate("x + 1", {"x": x}, None)
    first.compute(_getitem=True)
    second = module._prepare_cached_evaluate("x + 1", {"x": x + 10}, None)
    assert second is not first
    assert second.operands is not first.operands
    assert not hasattr(second, "_native_execution_report")
    np.testing.assert_array_equal(second.compute(_getitem=True), x + 11)
