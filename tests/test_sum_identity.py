"""Proven identity maps bypass evaluation without changing sum semantics."""

import numpy as np
import pytest

import blosc2
from blosc2.graph_plan import NativeGraph

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM or not getattr(blosc2.blosc2_ext, "menudet_pairwise_sum_available", lambda: False)(),
    reason="Requires native fixed-leaf sums",
)


@pytest.mark.parametrize(
    ("dtype", "chunks", "blocks", "codec"),
    [
        ("float32", 1024, 256, blosc2.Codec.BLOSCLZ),
        ("float64", 1000, 100, blosc2.Codec.ZSTD),
        ("float32", 65, 5, blosc2.Codec.ZSTD),
    ],
)
def test_identity_sum(dtype, chunks, blocks, codec):
    values = np.random.default_rng(42).normal(size=2051).astype(dtype)
    array = blosc2.asarray(values, chunks=(chunks,), blocks=(blocks,), cparams={"codec": codec})
    plan = NativeGraph.from_expression("sum(a)", {"a": dtype}, jit=False)
    kernel = plan._handle.reduction_artifact()
    baseline = None
    previous = blosc2.nthreads
    try:
        for threads in (1, 4):
            blosc2.set_nthreads(threads)
            for skip_identity in (False, True):
                result, report = kernel.evaluate_prefilter(
                    {"a": array},
                    _reduction_op="sum",
                    _reduction_out=np.empty((), dtype=dtype),
                    _pairwise_sum=True,
                    _skip_identity=skip_identity,
                )
                if baseline is None:
                    baseline = result.tobytes()
                assert result.tobytes() == baseline
                assert report["identity_map_skipped"] == skip_identity
                assert report["evaluated_blocks"] == (values.size + blocks - 1) // blocks
            np.testing.assert_array_equal(array[:], values)
            assert not array.schunk._has_postfilter
    finally:
        blosc2.set_nthreads(previous)


@pytest.mark.parametrize(
    ("clevel", "value"), [(0, 0.0), (0, 1.0), (1, 0.0), (1, 1.0), (1, np.inf), (1, np.nan)]
)
def test_identity_special_chunks(clevel, value):
    array = blosc2.full(
        (2051,), value, dtype="float64", chunks=(1000,), blocks=(100,), cparams={"clevel": clevel}
    )
    kernel = NativeGraph.from_expression("sum(a)", {"a": "float64"}, jit=False)._handle.reduction_artifact()
    results = []
    for skip_identity in (False, True):
        result, _ = kernel.evaluate_prefilter(
            {"a": array},
            _reduction_op="sum",
            _reduction_out=np.empty(()),
            _pairwise_sum=True,
            _skip_identity=skip_identity,
        )
        results.append(result.copy())
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])


def test_identity_delta_filter():
    values = np.arange(10003, dtype="float64") + 1000
    array = blosc2.asarray(
        values,
        chunks=(1000,),
        blocks=(100,),
        cparams={"filters": [blosc2.Filter.DELTA, blosc2.Filter.SHUFFLE]},
    )
    kernel = NativeGraph.from_expression("sum(a)", {"a": "float64"}, jit=False)._handle.reduction_artifact()
    for skip_identity in (False, True):
        result, _ = kernel.evaluate_prefilter(
            {"a": array},
            _reduction_op="sum",
            _reduction_out=np.empty(()),
            _pairwise_sum=True,
            _skip_identity=skip_identity,
        )
        assert result == values.sum()
    np.testing.assert_array_equal(array[:], values)


@pytest.mark.parametrize("invalid", [False, True])
def test_pairwise_fp_flags_with_map_evaluation(invalid):
    # The automatic pairwise test exercises the same merge with the map skipped.
    values = np.zeros(1024)
    values[0] = np.inf if invalid else np.finfo("float64").max
    values[128] = -np.inf if invalid else np.finfo("float64").max
    array = blosc2.asarray(values, chunks=(256,), blocks=(128,))
    kernel = NativeGraph.from_expression("sum(a)", {"a": "float64"}, jit=False)._handle.reduction_artifact()
    result, report = kernel.evaluate_prefilter(
        {"a": array},
        _reduction_op="sum",
        _reduction_out=np.empty(()),
        _pairwise_sum=True,
        _skip_identity=False,
    )
    assert not report["identity_map_skipped"]
    assert np.isnan(result) if invalid else np.isinf(result)
    if report["fp_supported"]:
        assert report["fp_flags"] & (1 if invalid else 4)
