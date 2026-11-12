#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import importlib
import random

import numpy as np
import pytest

import blosc2
from blosc2.dsl_kernel import SourceKernel

ndarray_module = importlib.import_module("blosc2.ndarray")


def test_diagonal_chunk_indices_reference():
    rng = random.Random(2026)
    for _ in range(200):
        rows, cols = rng.randrange(101), rng.randrange(101)
        chunk_rows, chunk_cols = rng.randrange(1, 40), rng.randrange(1, 40)
        k = rng.randrange(-120, 121)
        column_chunks = (cols + chunk_cols - 1) // chunk_cols
        expected = sorted(
            {
                row // chunk_rows * column_chunks + (row + k) // chunk_cols
                for row in range(rows)
                if 0 <= row + k < cols
            }
        )
        assert list(ndarray_module._diagonal_chunk_indices((rows, cols), (chunk_rows, chunk_cols), k)) == (
            expected
        )
    assert list(ndarray_module._diagonal_chunk_indices((30_000, 30_000), (1024, 1024), 0)) == [
        i * 31 for i in range(30)
    ]


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks", "k", "clevel"),
    [
        ((17, 23), (7, 9), (4, 5), 0, 5),
        ((17, 23), (7, 9), (4, 5), 7, 0),
        ((23, 17), (7, 9), (4, 5), -7, 5),
        ((3, 31), (11, 7), (4, 3), 29, 5),
    ],
)
def test_eye_only_updates_diagonal_chunks(monkeypatch, shape, chunks, blocks, k, clevel):
    original = blosc2.NDArray._execute_menudet_prefilter
    reports = []
    expected = np.eye(*shape, k=k, dtype=np.float64)

    def record(self, *args, **kwargs):
        selected = list(kwargs["chunk_indices"])
        kwargs["chunk_indices"] = iter(selected)
        untouched = {i: self.schunk.get_chunk(i) for i in range(self.schunk.nchunks) if i not in selected}
        result, report = original(self, *args, **kwargs)
        assert report["processed_chunks"] == len(selected) < self.schunk.nchunks
        assert report["threads"] == 2
        assert report["output_destination"] == "blosc2"
        for i, header in untouched.items():
            assert len(header) == blosc2.MAX_OVERHEAD
            assert self.schunk.get_chunk(i) == header
        reports.append(report)
        return result, report

    monkeypatch.setattr(blosc2.NDArray, "_execute_menudet_prefilter", record)
    result = blosc2.eye(
        *shape,
        k=k,
        chunks=chunks,
        blocks=blocks,
        jit=False,
        cparams={"codec": blosc2.Codec.LZ4, "nthreads": 2, "clevel": clevel},
    )
    assert len(reports) == 1
    np.testing.assert_array_equal(result[:], expected)


def test_eye_selected_chunks_fallback(monkeypatch):
    monkeypatch.setattr(
        blosc2.blosc2_ext, "menudet_prefilter_inputs_eligible", lambda *args, **kwargs: False
    )
    result = blosc2.eye(17, 23, k=7, chunks=(7, 9), blocks=(4, 5), jit=False)
    np.testing.assert_array_equal(result[:], np.eye(17, 23, k=7))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
@pytest.mark.parametrize("contiguous", [False, True])
def test_eye_selected_chunks_persistent(tmp_path, contiguous):
    path = tmp_path / "eye.b2nd"
    result = blosc2.eye(
        23,
        17,
        k=-3,
        chunks=(7, 9),
        blocks=(4, 5),
        urlpath=path,
        contiguous=contiguous,
        jit=False,
    )
    expected = np.eye(23, 17, k=-3)
    np.testing.assert_array_equal(result[:], expected)
    del result
    np.testing.assert_array_equal(blosc2.open(path)[:], expected)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
@pytest.mark.parametrize("indices", [[1], []])
def test_selected_prefilter_preserves_unselected_output(indices):
    author = SourceKernel.from_source("def fill():\n    return 7.0\n")
    kernel = blosc2.MenudetKernel.from_json(author.export({}, "float64", ndim=1), jit=False)
    output = blosc2.full((17,), 5.0, chunks=(8,), blocks=(4,), cparams={"nthreads": 2})
    result, report = kernel._handle.evaluate_prefilter({}, out=output, _chunk_indices=iter(indices))
    assert result is output
    assert report["processed_chunks"] == len(indices)
    expected = np.full(17, 5.0)
    if indices:
        expected[8:16] = 7
    np.testing.assert_array_equal(output[:], expected)


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
def test_selected_prefilter_with_compressed_input():
    author = SourceKernel.from_source("def increment(x):\n    return x + 1\n")
    kernel = blosc2.MenudetKernel.from_json(author.export({"x": "float64"}, "float64"), jit=False)
    values = np.arange(17, dtype=np.float64)
    source = blosc2.asarray(values, chunks=(8,), blocks=(4,))
    output = blosc2.full((17,), 5.0, chunks=(8,), blocks=(4,), cparams={"nthreads": 2})
    result, report = kernel._handle.evaluate_prefilter({"x": source}, out=output, _chunk_indices=iter([1]))
    assert result is output
    assert report["processed_chunks"] == report["input_chunk_fetches"] == 1
    expected = np.full(17, 5.0)
    expected[8:16] = values[8:16] + 1
    np.testing.assert_array_equal(output[:], expected)
    with pytest.raises(ValueError, match="Selected prefilter chunks require"):
        kernel._handle.evaluate_prefilter({"x": source}, _chunk_indices=iter([1]))


@pytest.mark.skipif(blosc2.IS_WASM, reason="Native prefilter scheduling is unavailable on WASM")
@pytest.mark.parametrize("bad_index", [-1, 3])
def test_selected_prefilter_bounds_and_cleanup(bad_index):
    author = SourceKernel.from_source("def fill():\n    return 7.0\n")
    kernel = blosc2.MenudetKernel.from_json(author.export({}, "float64", ndim=1), jit=False)
    output = blosc2.full((17,), 5.0, chunks=(8,), blocks=(4,), cparams={"nthreads": 2})
    with pytest.raises(ValueError, match="outside the output domain"):
        kernel._handle.evaluate_prefilter({}, out=output, _chunk_indices=iter([1, bad_index]))
    # A failed selected update must restore the ordinary compression context.
    output[0] = 3
    expected = np.full(17, 5.0)
    expected[0] = 3
    expected[8:16] = 7
    np.testing.assert_array_equal(output[:], expected)
