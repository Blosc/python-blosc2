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


def test_simple_numpy_map_prepares_without_dummy_evaluation(monkeypatch, native_graph_runtime):
    x = np.arange(100_000, dtype="float64")
    y = x + 1

    def forbidden(*args, **kwargs):
        pytest.fail("Simple NumPy map evaluated synthetic operands")

    monkeypatch.setattr(module, "_numpy_eval_expr", forbidden)
    expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
    assert expr.expression == "(o0 + (o1 - 1))"
    np.testing.assert_array_equal(expr[:], x + (y - 1))
    assert expr._native_execution_report["evaluated_tiles"] == 2
    np.testing.assert_array_equal(blosc2.evaluate("x + (y - 1)", local_dict={"x": x, "y": y}), x + (y - 1))


def test_plain_arithmetic_shares_preparation(monkeypatch, native_graph_runtime):
    x = np.arange(1000, dtype="int64")
    y = x + 1

    def forbidden(*args, **kwargs):
        pytest.fail("Reviewed arithmetic preparation repeated legacy validation")

    monkeypatch.setattr(module, "validate_expr", forbidden)
    np.testing.assert_array_equal(blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})[:], x + (y - 1))

    first = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
    second = blosc2.lazyexpr("x + (y - 1)", {"x": x + 10, "y": y})
    assert first._resident_native[1] is second._resident_native[1]
    np.testing.assert_array_equal(second[:], x + 10 + (y - 1))


@pytest.mark.parametrize("names", [("x", "e3"), ("α", "β")])
def test_numpy_rebasing_preserves_numeric_literals_and_aliases(names, native_graph_runtime):
    x = np.arange(17.0)
    left, right = names
    expression = f"{left} + ({right} - 1.e+3)"
    expr = blosc2.lazyexpr(expression, {left: x, right: x})
    assert expr.expression == "(o0 + (o0 - 1000.0))"
    assert tuple(expr.operands) == ("o0",)
    np.testing.assert_array_equal(expr[:], x + (x - 1000.0))


@pytest.mark.parametrize("engine", ["lazyexpr", "evaluate"])
def test_numpy_preparation_shares_readonly_syntax(monkeypatch, native_graph_runtime, engine):
    import ast

    import blosc2.expression_graph as graph
    import blosc2.native_graph as native

    for function in (
        graph.expression_syntax,
        graph.parse_expression,
        native._native_first_syntax,
        module._evaluate_recipe_names,
    ):
        function.cache_clear()
    module._evaluate_recipe_cache.clear()
    native._native_plan_cache.clear()
    monkeypatch.setenv("BLOSC_ME_JIT", "0")
    original = ast.parse
    parsed = []

    def record(*args, **kwargs):
        parsed.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(ast, "parse", record)
    text = "x + (y - 1)"
    x = np.arange(1000, dtype="float64")
    y = x + 1
    with blosc2.jit_options(jit=None):
        if engine == "lazyexpr":
            result = blosc2.lazyexpr(text, {"x": x, "y": y})[:]
        else:
            result = blosc2.evaluate(text, local_dict={"x": x, "y": y})
        np.testing.assert_array_equal(result, x + (y - 1))
        assert parsed == [text]
        syntax, nodes = graph.expression_syntax(text)
        assert {node.id for node in nodes if isinstance(node, ast.Name)} == {"x", "y"}
        snapshot = ast.dump(syntax, include_attributes=True)
        # Aliases change renaming but must leave the cached source untouched.
        np.testing.assert_array_equal(blosc2.lazyexpr(text, {"x": y, "y": y})[:], y + (y - 1))
        assert ast.dump(syntax, include_attributes=True) == snapshot


def test_prepared_numpy_plan_rebinds_and_invalidates(monkeypatch, native_graph_runtime):
    import blosc2.native_graph as native

    monkeypatch.setenv("BLOSC_ME_JIT", "0")
    x = np.arange(100_000, dtype="float64")
    y = x + 1
    with blosc2.jit_options(jit=None):
        expr = blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y})
        assert hasattr(expr, "_prepared_native")
        native._native_plan_cache.clear()

        with monkeypatch.context() as patch:
            patch.setattr(native, "lower_native_graph", lambda *args, **kwargs: pytest.fail("Plan rebuilt"))
            expr.operands["o0"] = x + 10
            np.testing.assert_array_equal(expr[:], x + 10 + (y - 1))

        original = native.lower_native_graph
        calls = []

        def record(*args, **kwargs):
            calls.append(None)
            return original(*args, **kwargs)

        monkeypatch.setattr(native, "lower_native_graph", record)
        expr.operands["o0"] = x.astype("float32")
        expr.operands["o1"] = y.astype("float32")
        native._native_plan_cache.clear()
        np.testing.assert_array_equal(expr[:], x.astype("float32") + (y.astype("float32") - 1))
        assert calls

        calls.clear()
        native._native_plan_cache.clear()
        with blosc2.jit_options(jit=False):
            np.testing.assert_array_equal(expr[:], x.astype("float32") + (y.astype("float32") - 1))
        assert calls

    with blosc2.jit_options(jit=True):
        assert not hasattr(blosc2.lazyexpr("x + (y - 1)", {"x": x, "y": y}), "_prepared_native")


def test_prepared_plan_without_destination_does_not_respecialize(monkeypatch, native_graph_runtime):
    import blosc2.native_graph as native

    monkeypatch.setenv("BLOSC_ME_JIT", "0")
    x = np.arange(17.0)
    with blosc2.jit_options(jit=None):
        expr = blosc2.lazyexpr("x + 1", {"x": x})
        plan = expr._prepared_native[1][0]
        native._native_plan_cache.clear()

        def forbidden(*args, **kwargs):
            pytest.fail("Metadata-identical plan specialized just for admission")

        with monkeypatch.context() as patch:
            patch.setattr(plan, "specialize", forbidden)
            prepared = native.prepare_native_first(expr, (), jit=None, options={"_getitem": True})
        assert prepared[0] is plan
        np.testing.assert_array_equal(expr[:], x + 1)


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
    plan, inputs = first
    plan[2].execute(inputs)
    second = module._prepare_cached_evaluate("x + 1", {"x": x + 10}, None)
    assert second is not first
    assert second[1] is not first[1]
    assert second[0] is first[0]
    np.testing.assert_array_equal(second[0][2].execute(second[1])[0], x + 11)
