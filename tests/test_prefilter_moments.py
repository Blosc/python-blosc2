"""Moment bindings/dispatch contracts; exhaustive numerical tests live in miniexpr."""

import numpy as np
import pytest

import blosc2


@pytest.fixture(autouse=True)
def moments_runtime(automatic_native_runtime, monkeypatch):
    if blosc2.IS_WASM or not getattr(blosc2.blosc2_ext, "menudet_moments_available", lambda: False)():
        pytest.skip("Requires native moments primitives")
    import blosc2.native_graph as native

    monkeypatch.setattr(native, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    previous = blosc2.nthreads
    blosc2.set_nthreads(2)
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize(
    ("op", "dtype", "mapped"),
    [
        ("var", "float32", False),
        ("var", "float64", True),
        ("std", "float32", True),
        ("std", "float64", False),
    ],
)
def test_moments_dispatch(op, dtype, mapped, monkeypatch):
    from blosc2.storage_readers import StorageReader

    values = (np.arange(10003) % 97).astype(dtype) + 10000
    array = blosc2.asarray(values, chunks=(1000,), blocks=(100,))
    expr = blosc2.lazyexpr("a + a" if mapped else "a", {"a": array})
    expected = getattr(np, op)(values * 2 if mapped else values, ddof=1)

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible moments must not materialize or replay the expression")

    monkeypatch.setattr(blosc2.NDArray, "__getitem__", forbidden)
    monkeypatch.setattr(StorageReader, "read", forbidden)
    monkeypatch.setattr(blosc2.LazyExpr, "_compute_expr", forbidden)
    for threads in (1, 2, 4):
        blosc2.set_nthreads(threads)
        result = getattr(expr, op)(ddof=1)
        assert result.shape == ()
        assert result.dtype == np.dtype(dtype)
        np.testing.assert_allclose(result, expected, rtol=2e-6)
        report = expr._native_execution_report
        assert report["route"] == "parallel-prefilter-moments"
        assert report["identity_map_skipped"] == (not mapped)
        assert report["input_materialization_bytes"] == 0
        assert report["reduction_buffer_bytes"] == 10 * 32


def test_moments_keepdims_out_and_axis():
    values = np.arange(2051, dtype=np.float64)
    array = blosc2.asarray(values, chunks=(1024,), blocks=(256,))
    for op in ("var", "std"):
        for axis in (None, 0, -1):
            out = np.empty(1)
            assert getattr(array, op)(axis=axis, keepdims=True, ddof=0.5, out=out) is out
            np.testing.assert_allclose(out, getattr(np, op)(values, ddof=0.5))


def test_moments_fallback_and_no_replay(monkeypatch):
    values = np.arange(1000, dtype=np.float64)
    array = blosc2.asarray(values, chunks=(100,), blocks=(10,))
    expr = blosc2.lazyexpr("a", {"a": array})
    np.testing.assert_allclose(expr.var(item=slice(0, 30)), np.var(values[:30]))
    original = blosc2.NDArray._execute_menudet_prefilter

    def fail(self, *args, **kwargs):
        # A moments slot is four doubles, not one scalar per block.
        if kwargs.get("reduction_out") is not None and kwargs["reduction_out"].size == 4:
            raise RuntimeError("moments execution probe")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", fail)
    with pytest.raises(RuntimeError, match="moments execution probe"):
        expr.var()


def test_moments_capability_fallback(monkeypatch):
    monkeypatch.setattr(blosc2.blosc2_ext, "menudet_moments_available", lambda: False)
    array = blosc2.arange(1003, dtype=np.float64, chunks=(100,), blocks=(10,))
    expr = blosc2.lazyexpr("a", {"a": array})
    np.testing.assert_allclose(expr.var(), np.var(np.arange(1003)))
    assert getattr(expr, "_native_execution_report", {}).get("route") != "parallel-prefilter-moments"


def test_moments_partial_validation():
    for partials in (np.ones(2), np.ones(4, dtype=np.float32), np.ones((1, 4))):
        with pytest.raises(ValueError, match="partials"):
            blosc2.blosc2_ext.menudet_combine_moments(partials)
    # Result narrowing belongs inside the isolated floating environment too.
    result, flags = blosc2.blosc2_ext.menudet_combine_moments(np.array([1, 0, 0, 1e80]), dtype="float32")
    assert result.dtype == np.dtype("float32")
    assert np.isinf(result)
    assert flags != 0


def test_moments_output_validation():
    from blosc2.native_graph import lower_native_graph

    array = blosc2.arange(8, dtype=np.float64, chunks=(8,), blocks=(4,))
    source = np.arange(8, dtype=np.float64)
    plan, inputs, _ = lower_native_graph(blosc2.lazyexpr("a + b", {"a": array, "b": source}))
    artifact = plan._handle.reduction_artifact()
    with pytest.raises(ValueError, match="overlap"):
        artifact.evaluate_prefilter(inputs, _reduction_op="moments", _reduction_out=source[:4])
    with pytest.raises(ValueError, match="output"):
        artifact.evaluate_prefilter(inputs, _reduction_op="moments", _reduction_out=np.empty(2))
    with pytest.raises(ValueError, match="output"):
        artifact.evaluate_prefilter(inputs, _reduction_op="moments", _reduction_out=np.empty(8))


def test_mean_uses_accelerated_sum(monkeypatch):
    from blosc2.storage_readers import StorageReader

    values = np.arange(131075, dtype=np.float32)
    array = blosc2.asarray(values, chunks=(8192,), blocks=(256,))
    expr = blosc2.lazyexpr("a", {"a": array})

    def forbidden(*args, **kwargs):
        pytest.fail("Eligible mean must use the direct pairwise sum")

    monkeypatch.setattr(StorageReader, "read", forbidden)
    result = expr.mean(keepdims=True)
    assert result.shape == (1,)
    assert result.dtype == np.dtype("float32")
    np.testing.assert_allclose(result, values.mean())
    assert expr._native_execution_report["route"] == "parallel-prefilter-pairwise-sum"


@pytest.mark.parametrize("gate", ["dtype", "axis", "accuracy", "scratch"])
def test_moments_metadata_fallback(gate):
    from blosc2.native_graph import compute_prefilter_moments_if_eligible

    array = blosc2.arange(
        33001 if gate == "scratch" else 1003,
        dtype=np.float64,
        chunks=(33001 if gate == "scratch" else 100,),
        blocks=(1,),
    )
    expr = blosc2.lazyexpr("a", {"a": array})
    assert (
        compute_prefilter_moments_if_eligible(
            expr,
            axis=(0,) if gate == "axis" else None,
            dtype=np.float32 if gate == "dtype" else None,
            ddof=0,
            keepdims=False,
            stddev=False,
            fp_accuracy=blosc2.FPAccuracy.DEFAULT if gate == "accuracy" else None,
            execution_kwargs={},
        )
        is None
    )


def test_moments_streams_across_many_chunks():
    values = (np.arange(33001) % 113).astype(np.float64)
    array = blosc2.asarray(values, chunks=(100,), blocks=(1,))
    for op in ("var", "std"):
        expr = blosc2.lazyexpr("a", {"a": array})
        result = getattr(expr, op)(ddof=1)
        expected = getattr(np, op)(values, ddof=1)
        np.testing.assert_allclose(result, expected, rtol=1e-12)
        report = expr._native_execution_report
        assert report["route"] == "parallel-prefilter-moments"
        assert report["reduction_buffer_bytes"] == 100 * 32
