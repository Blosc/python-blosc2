"""Automatic dispatch and storage/thread invariance, not native numeric matrices."""

import math

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM or not getattr(blosc2.blosc2_ext, "menudet_pairwise_sum_available", lambda: False)(),
    reason="Requires native fixed-leaf pairwise sums",
)


@pytest.fixture
def automatic_pairwise(monkeypatch, automatic_native_runtime):
    from blosc2 import native_graph, resident_execution

    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    monkeypatch.setattr(resident_execution, "MINIMUM_WORKER_ITEMS", 1)
    previous = blosc2.nthreads
    yield
    blosc2.set_nthreads(previous)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mapped", [False, True])
def test_pairwise_storage_and_thread_invariance(automatic_pairwise, dtype, mapped):
    values = np.random.default_rng(42).normal(size=2051).astype(dtype)
    values[::3] *= 1e6
    expected = math.fsum(float(x) for x in (values + values if mapped else values))
    first_bits = None
    # Sweep workers on a fixed layout, then layouts with a fixed worker count.
    for chunk, block, worker_counts in ((1024, 256, (1, 2, 8)), (1000, 100, (2,)), (65, 5, (2,))):
        arr = blosc2.asarray(values, chunks=(chunk,), blocks=(block,))
        for threads in worker_counts:
            blosc2.set_nthreads(threads)
            expr = blosc2.lazyexpr("a + a" if mapped else "a", {"a": arr})
            result = expr.sum()
            assert result.shape == ()
            assert result.dtype == np.dtype(dtype)
            report = expr._native_execution_report
            assert report["route"] == "parallel-prefilter-pairwise-sum"
            assert report["summation"] == "fixed-pairwise"
            assert report["identity_map_skipped"] == (not mapped)
            assert report["input_materialization_bytes"] == 0
            if first_bits is None:
                first_bits = result.tobytes()
            else:
                assert result.tobytes() == first_bits
            np.testing.assert_allclose(result, expected, rtol=1e-6 if dtype == "float32" else 1e-14)


def test_pairwise_large_accumulator(automatic_pairwise):
    # The total exceeds 2**53 without allocating a billion-element Python test.
    count, start = 131072, 10**12
    arr = blosc2.arange(start, start + count, dtype="float64", chunks=(16000,), blocks=(1000,))
    expected = count * (2 * start + count - 1) // 2
    for keepdims in (False, True):
        expr = blosc2.lazyexpr("a", {"a": arr})
        result = expr.sum(keepdims=keepdims)
        assert result.shape == ((1,) if keepdims else ())
        assert result == float(expected)
        assert expr._native_execution_report["route"] == "parallel-prefilter-pairwise-sum"


def test_pairwise_execution_failure_never_replays(automatic_pairwise, monkeypatch):
    from blosc2 import native_graph

    arr = blosc2.arange(10000, dtype="float64", chunks=(1000,), blocks=(100,))
    original = blosc2.NDArray._execute_menudet_prefilter

    def fail(self, *args, **kwargs):
        if kwargs.get("pairwise_sum"):
            raise RuntimeError("pairwise execution probe")
        return original(self, *args, **kwargs)

    def no_legacy(*args, **kwargs):
        pytest.fail("Execution failures must not replay legacy reductions")

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", fail)
    monkeypatch.setattr(native_graph, "evaluate_graph_sources", no_legacy)
    with pytest.raises(RuntimeError, match="pairwise execution probe"):
        arr.sum()


@pytest.mark.parametrize("invalid", [False, True])
def test_pairwise_merge_fp_flags(automatic_pairwise, invalid):
    values = np.zeros(1024)
    # Put the operands in separate leaves: flags must include the merge itself,
    # not only map evaluation and worker-local leaf arithmetic.
    values[0] = np.inf if invalid else np.finfo("float64").max
    values[128] = -np.inf if invalid else np.finfo("float64").max
    arr = blosc2.asarray(values, chunks=(256,), blocks=(128,))
    expr = blosc2.lazyexpr("a", {"a": arr})
    result = expr.sum()
    assert np.isnan(result) if invalid else np.isinf(result)
    report = expr._native_execution_report
    assert report["route"] == "parallel-prefilter-pairwise-sum"
    assert report["identity_map_skipped"]
    if report["fp_supported"]:
        assert report["fp_flags"] & (1 if invalid else 4)
