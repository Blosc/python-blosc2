"""Fallback scheduling amortizes metadata without retaining chunk arrays."""

import gc
import importlib
import weakref

import numpy as np
import pytest

import blosc2

lazy = importlib.import_module("blosc2.lazyexpr")
native = importlib.import_module("blosc2.native_graph")


@pytest.fixture(autouse=True)
def typed_mode(monkeypatch):
    monkeypatch.setenv("BLOSC2_EXPRESSION_EVALUATOR", "typed")


def test_nonflat_private_output_avoids_general_assignment(monkeypatch):
    values = np.arange(9 * 13, dtype=np.float32).reshape(9, 13)
    a = blosc2.asarray(values, chunks=(4, 5), blocks=(2, 3))
    expr = a * 2 + 1
    original_getitem = blosc2.NDArray.__getitem__

    def forbidden(*args, **kwargs):
        pytest.fail("Private chunk output used general assignment")

    def getitem(self, key):
        if self is a:
            pytest.fail("Scheduled input read repeated public indexing")
        return original_getitem(self, key)

    monkeypatch.setattr(blosc2.NDArray, "__setitem__", forbidden)
    monkeypatch.setattr(blosc2.NDArray, "__getitem__", getitem)
    np.testing.assert_array_equal(expr.compute()[:], values * 2 + 1)


def test_nonflat_user_output_keeps_general_assignment(monkeypatch):
    values = np.arange(9 * 13, dtype=np.float32).reshape(9, 13)
    a = blosc2.asarray(values, chunks=(4, 5), blocks=(2, 3))
    out = blosc2.zeros_like(a)
    expr = a * 2 + 1
    calls = []
    original = blosc2.NDArray.__setitem__

    def record(self, key, value):
        if self is out:
            calls.append(key)
        return original(self, key, value)

    monkeypatch.setattr(blosc2.NDArray, "__setitem__", record)
    # Explicitly select the chunk scheduler, not a different automatic native sink.
    assert expr.compute(out=out, _require_native=False) is out
    assert calls
    np.testing.assert_array_equal(out[:], values * 2 + 1)


@pytest.mark.parametrize("operation", ["compute", "sum"])
def test_persisted_edge_chunks_are_read_once(tmp_path, monkeypatch, operation):
    if blosc2.IS_WASM:
        pytest.skip("Prefetch reader requires host threads")
    values = np.arange(7 * 9, dtype=np.float32).reshape(7, 9)
    a = blosc2.asarray(values, chunks=(3, 4), blocks=(2, 3), urlpath=str(tmp_path / "a.b2nd"))
    expr = a + 1
    reads = []
    original = blosc2.NDArray.__getitem__

    def record(self, key):
        if self is a:
            reads.append(key)
        return original(self, key)

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", record)
    actual = getattr(expr, operation)()
    expected = values + 1
    if operation == "compute":
        np.testing.assert_array_equal(actual[:], expected)
    else:
        np.testing.assert_allclose(actual, expected.sum())
    assert len(reads) == a.schunk.nchunks
    assert lazy._chunk_read_state.get() is None


def test_persisted_zero_edges_keep_reduction_extent(tmp_path):
    a = blosc2.zeros(
        (7, 9), dtype=np.float32, chunks=(3, 4), blocks=(2, 3), urlpath=str(tmp_path / "zeros.b2nd")
    )
    assert (a + 1).sum() == 63


def test_native_chunk_schedules_rebind_and_release(automatic_native_runtime, monkeypatch):
    values = np.arange(9 * 13, dtype=np.float32).reshape(9, 13)
    a = blosc2.asarray(values, chunks=(4, 5), blocks=(2, 3))
    expr = a * 2 + 1
    calls, references, states = [], [], []
    original_lower = native.lower_native_graph
    original_evaluate = lazy.evaluate_chunk

    def lower(stub, **kwargs):
        if all(type(value) is np.ndarray for value in stub.operands.values()):
            calls.append(tuple(value.shape for value in stub.operands.values()))
            references.extend(weakref.ref(value) for value in stub.operands.values())
        return original_lower(stub, **kwargs)

    def evaluate(*args, **kwargs):
        states.append(lazy._chunk_read_state.get())
        return original_evaluate(*args, **kwargs)

    monkeypatch.setattr(native, "lower_native_graph", lower)
    monkeypatch.setattr(lazy, "evaluate_chunk", evaluate)
    with blosc2.jit_options(jit=True):
        np.testing.assert_array_equal(expr.compute()[:], values * 2 + 1)
    # Four distinct valid edge shapes, not one preparation per physical chunk.
    assert len(calls) == 4
    assert len(set(calls)) == 4
    assert states
    assert all(not state.schedules for state in states)
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize(("expression", "numpy_expected"), [("x * 2 + 1", True), ("sin(x)", False)])
def test_small_chunk_numpy_policy_is_arithmetic_only(
    automatic_native_runtime, monkeypatch, expression, numpy_expected
):
    values = np.arange(17, dtype=np.float32) * 0.01
    expected = lazy.evaluate_chunk(expression, {"x": values})
    calls = []
    original = lazy._numpy_eval_datetime_aware

    def record(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lazy, "_numpy_eval_datetime_aware", record)
    state = lazy._ChunkReadState()
    token = lazy._chunk_read_state.set(state)
    try:
        with blosc2.jit_options(jit=None, jit_backend=None):
            actual = lazy.evaluate_chunk(expression, {"x": values})
    finally:
        state.close()
        lazy._chunk_read_state.reset(token)
    np.testing.assert_array_equal(actual, expected)
    assert bool(calls) is numpy_expected
