"""Direct compressed reductions preserve native reducer semantics."""

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph
from blosc2.native_graph import _parallel_compressed_reduction


@pytest.mark.skipif(blosc2.IS_WASM, reason="Requires native prefilters")
@pytest.mark.parametrize(
    ("op", "mapped", "case"),
    [
        ("min", False, "nan"),
        ("min", True, "zeros"),
        ("min", False, "finite"),
        ("max", True, "nan"),
        ("max", False, "zeros"),
        ("max", True, "finite"),
        ("any", False, "zeros"),
        ("any", True, "nan"),
        ("all", True, "zeros"),
        ("all", False, "finite"),
    ],
)
def test_prefilter_reduction(op, mapped, case, monkeypatch):
    import blosc2.native_graph as native_graph

    if op in {"min", "max"} and not blosc2.blosc2_ext.menudet_shared_reduction_rules_available():
        pytest.skip("Requires native shared extrema primitives")
    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    values = np.arange(1, 400004, dtype=np.float64)
    if case == "nan":
        if op == "any":  # Only the NaN can make this true.
            values[:] = 0.0
        values[128:131] = [np.nan, -0.0, 0.0]
    elif case == "zeros":
        values[:] = 0.0
        values[::2] = -0.0
    # Min and truth reducers require enough items per chunk for all workers.
    array = blosc2.asarray(values, chunks=(262144,), blocks=(1024,))
    expression = f"{op}(a + a)" if mapped else f"{op}(a)"
    plan = NativeGraph.from_expression(expression, {"a": "float64"}, jit=False)
    previous = blosc2.nthreads
    try:
        blosc2.set_nthreads(4)
        actual, report = _parallel_compressed_reduction(plan, {"a": array}, values.shape, 65536)
        expected = plan.evaluate({"a": values})
        assert actual.tobytes() == expected.tobytes()
        assert report["route"] == "parallel-prefilter-reduction"
        assert report["identity_map_skipped"] == (not mapped)
        assert report["reduction_buffer_bytes"] <= 1024 * 1024
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Requires native prefilters")
def test_partial_storage_cap_retains_streaming_route(monkeypatch):
    import blosc2.native_graph as native_graph

    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    values = np.arange(262145, dtype="float64")
    # Small blocks exceed the partial-buffer cap without a large input.
    array = blosc2.asarray(values, chunks=(8192,), blocks=(1,))
    plan = NativeGraph.from_expression("max(a)", {"a": "float64"}, jit=False)
    previous = blosc2.nthreads
    try:
        blosc2.set_nthreads(2)
        result, report = _parallel_compressed_reduction(plan, {"a": array}, values.shape, 65536)
        assert result == 262144
        assert report["route"] == "parallel-graph-bounded-readers"
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Requires native parallel scheduling")
def test_small_chunks_use_parallel_regions(monkeypatch):
    import blosc2.native_graph as native_graph

    monkeypatch.setattr(native_graph, "MINIMUM_COMPRESSED_GRAPH_ITEMS", 1)
    values = np.arange(400003, dtype=np.float64)
    array = blosc2.asarray(values, chunks=(16384,), blocks=(1024,))
    plan = NativeGraph.from_expression("min(a)", {"a": "float64"}, jit=False)
    previous = blosc2.nthreads
    try:
        blosc2.set_nthreads(4)
        result, report = _parallel_compressed_reduction(plan, {"a": array}, values.shape, 65536)
        assert result == 0
        assert report["route"] == "parallel-graph-bounded-readers"
    finally:
        blosc2.set_nthreads(previous)
