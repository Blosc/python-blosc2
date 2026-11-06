"""Generator scheduling tests; scalar numerical matrices belong to miniexpr."""

import numpy as np
import pytest

import blosc2

pytestmark = pytest.mark.skipif(
    blosc2.IS_WASM or not getattr(blosc2.blosc2_ext, "menudet_prefilter_available", lambda: False)(),
    reason="Native generator prefilter is not available",
)


@blosc2.menudet(ndim=1, jit=False)
def grouped_index() -> np.int64:
    return _i0 + block_sum(_flat_idx) + _n0  # noqa: F821


def expected_groups(size, partition):
    expected = np.arange(size, dtype=np.int64) + size
    for start in range(0, size, partition):
        expected[start : start + partition] += np.arange(start, min(start + partition, size)).sum()
    return expected


@pytest.mark.parametrize(("clevel", "threads"), [(0, 1), (0, 4), (5, 4)])
def test_generator_prefilter_preserves_context_and_whole_groups(tmp_path, clevel, threads):
    lazy = grouped_index.specialize().lazy({}, shape=(37,), partitions=(8,))
    expected = expected_groups(37, 8)
    output = lazy.compute(
        chunks=(24,),
        blocks=(8,),
        cparams={"clevel": clevel, "nthreads": threads},
        urlpath=tmp_path / "generated.b2nd",
        mode="w",
    )
    np.testing.assert_array_equal(output[:], expected)
    np.testing.assert_array_equal(blosc2.open(tmp_path / "generated.b2nd")[:], expected)
    report = lazy._source_execution_report
    assert report["route"] == "menudet-prefilter"
    assert report["evaluated_blocks"] == 5
    assert report["threads"] == threads
    assert report["input_buffer_bytes"] == 0
    assert report["output_copy_bytes"] == 0
    assert report["logical_partitions"] == (8,)


@pytest.mark.parametrize(("chunks", "blocks"), [(24, 4), (10, 8)])
def test_generator_prefilter_rejects_repartitioning(chunks, blocks):
    lazy = grouped_index.specialize().lazy({}, shape=(37,), partitions=(8,))
    output = lazy.compute(chunks=(chunks,), blocks=(blocks,), cparams={"clevel": 0})
    np.testing.assert_array_equal(output[:], expected_groups(37, 8))
    assert not hasattr(lazy, "_source_execution_report")


def test_generator_partial_selection_keeps_original_groups():
    lazy = grouped_index.specialize().lazy({}, shape=(37,), partitions=(8,))
    result = lazy.compute(item=slice(3, 21), chunks=(16,), blocks=(8,), cparams={"clevel": 0})
    np.testing.assert_array_equal(result[:], expected_groups(37, 8)[3:21])
    assert not hasattr(lazy, "_source_execution_report")


def test_generator_prefilter_raises_native_error_and_cleans_output_context():
    @blosc2.menudet(ndim=1, jit=False)
    def checked() -> np.int64:
        if _flat_idx >= 8:  # noqa: F821
            return int64(1e100)  # noqa: F821
        return _flat_idx  # noqa: F821

    output = blosc2.uninit((37,), dtype=np.int64, chunks=(24,), blocks=(8,), cparams={"clevel": 0})
    lazy = checked.specialize().lazy({}, shape=(37,), partitions=(8,))
    with pytest.raises(blosc2.MenudetError):
        lazy.compute(out=output)
    # The prefilter's temporary native state must not survive either outcome.
    output[:] = np.arange(37)
    np.testing.assert_array_equal(output[:], np.arange(37))
    valid = grouped_index.specialize().lazy({}, shape=(37,), partitions=(8,))
    assert valid.compute(out=output) is output
    np.testing.assert_array_equal(output[:], expected_groups(37, 8))


def test_generator_prefilter_aggregates_floating_status():
    @blosc2.menudet(ndim=1, jit=False)
    def reciprocal() -> np.float64:
        return 1.0 / (float64(_flat_idx) - 8.0)  # noqa: F821

    kernel = reciprocal.specialize()
    lazy = kernel.lazy({}, shape=(37,), partitions=(8,))
    result = lazy.compute(chunks=(24,), blocks=(8,), cparams={"clevel": 0, "nthreads": 4})
    expected = []
    flags = 0
    for origin in range(0, 37, 8):
        block, status = kernel.evaluate_block(
            {},
            block_shape=(min(8, 37 - origin),),
            logical_shape=(37,),
            block_origin=(origin,),
            return_status=True,
        )
        expected.append(block)
        flags |= status["flags"]
    np.testing.assert_array_equal(result[:], np.concatenate(expected))
    assert lazy._source_execution_report["fp_flags"] == flags
    assert lazy._source_execution_report["fp_supported"] == status["supported"]
