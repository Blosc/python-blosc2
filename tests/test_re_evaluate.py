"""Thread-local convenience state is independent of explicit prepared recipes."""

import gc
import importlib
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import blosc2

module = importlib.import_module("blosc2.prepared_evaluate")


@pytest.fixture(autouse=True)
def clean_last_expression():
    previous = getattr(module._evaluation_state, "last", None)
    module._evaluation_state.last = None
    yield
    module._evaluation_state.last = previous


def test_requires_previous_success():
    with pytest.raises(RuntimeError, match="evaluate"):
        blosc2.re_evaluate()


def test_caller_scope_and_explicit_output_rebinding():
    x = np.arange(12.0)
    out = np.empty_like(x)
    assert blosc2.evaluate("x + 1", out=out) is out
    x = x + 10
    assert blosc2.re_evaluate() is out
    np.testing.assert_array_equal(out, x + 1)
    x = x + 20
    assert blosc2.re_evaluate(local_dict={"x": x}, global_dict={}) is out
    np.testing.assert_array_equal(out, x + 1)


def test_global_lookup_and_frame_depth():
    blosc2.evaluate("x + y", local_dict={"x": np.ones(3)}, global_dict={"y": 2})
    result = blosc2.re_evaluate(local_dict={"x": np.ones(3)}, global_dict={"y": 4})
    np.testing.assert_array_equal(result, 5)
    x, y = np.arange(3.0), 8

    def wrapper():
        return blosc2.re_evaluate(_frame_depth=3)

    np.testing.assert_array_equal(wrapper(), x + y)


def test_arrays_not_retained_and_dead_output_allocates_fresh():
    x = np.arange(7.0)
    out = np.empty_like(x)
    references = [weakref.ref(x), weakref.ref(out)]
    blosc2.evaluate("x + 1", local_dict={"x": x}, out=out)
    del x, out
    gc.collect()
    assert all(reference() is None for reference in references)
    result = blosc2.re_evaluate(local_dict={"x": np.arange(7.0)}, global_dict={})
    np.testing.assert_array_equal(result, np.arange(7.0) + 1)


def test_fresh_output_and_option_reuse():
    x = np.arange(12.0).reshape(3, 4)
    first = blosc2.evaluate("x + 1", local_dict={"x": x}, order="F", disable_cache=True)
    second = blosc2.re_evaluate(local_dict={"x": x + 1}, global_dict={})
    assert second is not first
    assert second.flags.f_contiguous
    np.testing.assert_array_equal(second, x + 2)


def test_failed_call_and_prepared_evaluation_preserve_state():
    x = np.arange(5.0)
    blosc2.evaluate("x + 1", local_dict={"x": x})
    with pytest.raises(ValueError):
        blosc2.evaluate("x + 2", local_dict={"x": x}, order="invalid")
    blosc2.prepare_evaluate("x * 100")({"x": x})
    np.testing.assert_array_equal(blosc2.re_evaluate({"x": x}), x + 1)


def test_rebinding_is_checked_and_failure_does_not_destroy_state():
    x = np.arange(5.0)
    out = np.empty_like(x)
    blosc2.evaluate("x + 1", local_dict={"x": x}, out=out)
    with pytest.raises(ValueError):
        blosc2.re_evaluate({"x": np.ones(9)})
    assert blosc2.re_evaluate({"x": x}) is out
    np.testing.assert_array_equal(out, x + 1)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Host thread pools are unavailable on WASM")
def test_thread_isolation():
    barrier = threading.Barrier(2)

    def run(offset):
        with pytest.raises(RuntimeError):
            blosc2.re_evaluate()
        x = np.arange(5.0)
        blosc2.evaluate(f"x + {offset}", local_dict={"x": x})
        barrier.wait(timeout=10)
        result = blosc2.re_evaluate({"x": x + 10})
        np.testing.assert_array_equal(result, x + 10 + offset)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, [1, 100]))
    with pytest.raises(RuntimeError):
        blosc2.re_evaluate()


def test_nested_evaluation_does_not_replace_state_on_failure(monkeypatch):
    lazy = importlib.import_module("blosc2.lazyexpr")
    x = np.arange(5.0)
    blosc2.evaluate("x + 1", local_dict={"x": x})
    original = lazy._prepare_cached_evaluate

    def prepare(expression, *args):
        if expression == "x + 2":
            blosc2.evaluate("x * 100", local_dict={"x": x})
            raise ValueError("outer failure")
        return original(expression, *args)

    monkeypatch.setattr(lazy, "_prepare_cached_evaluate", prepare)
    with pytest.raises(ValueError, match="outer failure"):
        blosc2.evaluate("x + 2", local_dict={"x": x})
    np.testing.assert_array_equal(blosc2.re_evaluate({"x": x}), x + 1)


def test_saved_safe_mode_is_preserved(monkeypatch):
    lazy = importlib.import_module("blosc2.lazyexpr")
    x = np.arange(5.0)
    with blosc2.expression_evaluation("safe"):
        blosc2.evaluate("x + 1", local_dict={"x": x}, sanitize=False)
    original = lazy.evaluate
    modes = []

    def record(*args, **kwargs):
        modes.append(lazy.evaluation_mode())
        return original(*args, **kwargs)

    monkeypatch.setattr(lazy, "evaluate", record)
    with blosc2.expression_evaluation("full"):
        blosc2.re_evaluate({"x": x})
    assert modes == ["safe"]
