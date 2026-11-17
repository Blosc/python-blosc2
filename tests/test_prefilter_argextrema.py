"""Binding, dispatch, and scheduling contracts; native tests own dtype matrices."""

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def argextrema_runtime(automatic_native_runtime, monkeypatch):
    if (
        blosc2.IS_WASM
        or not getattr(blosc2.blosc2_ext, "menudet_shared_reduction_rules_available", lambda: False)()
    ):
        pytest.skip("Requires native shared extrema primitives")
    import blosc2.native_graph as native

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    previous = blosc2.nthreads
    blosc2.set_nthreads(2)
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize("op", ["argmin", "argmax"])
@pytest.mark.parametrize("mapped", [False, True])
def test_argextrema_dispatch(op, mapped, monkeypatch):
    from blosc2.storage_readers import StorageReader

    values = np.zeros(10003, dtype=np.float32)
    values[1000] = values[9000] = -10 if op == "argmin" else 10
    array = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    expr = blosc2.lazyexpr("a + a" if mapped else "a", {"a": array})

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible arg-extrema must not materialize or replay the expression")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    previous = blosc2.nthreads
    try:
        for threads in (2, 4):
            blosc2.set_nthreads(threads)
            result = getattr(expr, op)()
            assert result.shape == ()
            assert result.dtype == np.dtype(blosc2.DEFAULT_INDEX)
            assert result == 1000
            report = expr._native_execution_report
            assert report["route"] == "parallel-prefilter-argextrema"
            assert report["identity_map_skipped"] == (not mapped)
            assert report["input_materialization_bytes"] == 0
            assert report["reduction_buffer_bytes"] <= 1024 * 1024
    finally:
        blosc2.set_nthreads(previous)


def test_argextrema_nan_tail_and_keepdims():
    values = np.arange(2051, dtype=np.float64)
    values[1024] = values[-1] = np.nan
    array = blosc2.asarray(values, chunks=(1024,), blocks=(256,))
    for op in ("argmin", "argmax"):
        for axis in (None, 0, -1):
            result = getattr(array, op)(axis=axis, keepdims=True)
            assert result.shape == (1,)
            assert result[0] == 1024
    expr = blosc2.lazyexpr("a", {"a": array})
    assert expr.argmax() == 1024
    assert expr._native_execution_report["route"] == "parallel-prefilter-argextrema"


def test_argextrema_integer_binding():
    # Above 2**53: a float conversion would lose the correct comparison.
    values = np.array([2**63 + 2, 2**63 + 3, 2**63 + 3], dtype=np.uint64)
    array = blosc2.asarray(values, chunks=(2,), blocks=(1,))
    assert array.argmax() == 1
    assert array.argmin() == 0


def test_argextrema_fallback_and_no_replay(monkeypatch):
    values = np.arange(1000, dtype=np.float64)
    array = blosc2.asarray(values, chunks=(100,), blocks=(10,))
    # Non-full-domain requests remain on the existing fallback.
    expr = blosc2.lazyexpr("a", {"a": array})
    assert expr.argmax(item=slice(0, 30)) == 29
    assert not hasattr(expr, "_native_execution_report")
    original = blosc2.NDArray._execute_menudet_prefilter

    def fail(self, *args, **kwargs):
        if kwargs.get("arg_indices") is not None:
            raise RuntimeError("arg-extrema execution probe")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", fail)
    with pytest.raises(RuntimeError, match="arg-extrema execution probe"):
        array.argmax()


def test_native_partial_buffer_validation():
    with pytest.raises(ValueError, match="partials"):
        blosc2.blosc2_ext.menudet_combine_argextrema(np.ones(2), np.array(0, dtype=np.int64), True)


def test_argextrema_serial_and_scratch_fallback():
    previous = blosc2.nthreads
    try:
        blosc2.set_nthreads(1)
        array = blosc2.arange(1000, dtype=np.float64, chunks=(100,), blocks=(10,))
        expr = blosc2.lazyexpr("a", {"a": array})
        assert expr.argmax() == 999
        assert not hasattr(expr, "_native_execution_report")
        blosc2.set_nthreads(4)
        # Each partial requires both a value and an index.
        array = blosc2.arange(100000, dtype=np.float64, chunks=(1000,), blocks=(1,))
        expr = blosc2.lazyexpr("a", {"a": array})
        assert expr.argmax() == 99999
        assert not hasattr(expr, "_native_execution_report")
    finally:
        blosc2.set_nthreads(previous)
