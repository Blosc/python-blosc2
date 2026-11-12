#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import math

import numpy as np
import pytest

import blosc2
from blosc2 import core


@pytest.fixture
def fixed_chunksize(monkeypatch):
    monkeypatch.setattr(core, "get_chunksize", lambda *args, **kwargs: 2**24)


def chunk_count(shape, chunks):
    return math.prod((s + c - 1) // c for s, c in zip(shape, chunks, strict=True))


def padded_chunk_bytes(chunks, blocks, dtype):
    return (
        math.prod((c + b - 1) // b * b for c, b in zip(chunks, blocks, strict=True))
        * np.dtype(dtype).itemsize
    )


@pytest.mark.usefixtures("fixed_chunksize")
def test_automatic_frame_chunks_grow_only_when_needed():
    small = blosc2.compute_chunks_blocks((2**48,), blocks=(2**14,), dtype=np.float64, contiguous=True)
    large = blosc2.compute_chunks_blocks((2**49,), blocks=(2**14,), dtype=np.float64, contiguous=True)
    assert small == ((2**21,), (2**14,))
    assert large == ((2**22,), (2**14,))
    # In-memory non-contiguous storage has no frame offsets payload.
    assert blosc2.compute_chunks_blocks((2**49,), blocks=(2**14,), dtype=np.float64) == small


@pytest.mark.parametrize("explicit_blocks", [False, True])
def test_explicit_frame_chunks_report_offsets_limit(explicit_blocks):
    kwargs = {"blocks": (2**14,)} if explicit_blocks else {}
    with pytest.raises(ValueError, match=r"Frame requires.*offsets limit.*Increase chunks"):
        blosc2.zeros(2**49, dtype=np.float64, contiguous=True, chunks=(2**21,), **kwargs)


@pytest.mark.parametrize("as_mapping", [False, True])
def test_compute_layout_recognizes_storage_argument(as_mapping):
    storage = {"contiguous": True} if as_mapping else blosc2.Storage(contiguous=True)
    chunks, blocks = blosc2.compute_chunks_blocks((2**49,), dtype=np.float64, storage=storage)
    assert chunk_count((2**49,), chunks) <= blosc2.MAX_BUFFERSIZE // 8
    assert padded_chunk_bytes(chunks, blocks, np.float64) <= blosc2.MAX_BUFFERSIZE


@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("as_storage", [False, True])
def test_persistent_frame_chunks(tmp_path, contiguous, as_storage):
    urlpath = tmp_path / "huge.b2nd"
    kwargs = {"urlpath": urlpath, "contiguous": contiguous}
    if as_storage:
        kwargs = {"storage": blosc2.Storage(**kwargs)}
    a = blosc2.zeros(2**49, dtype=np.float64, **kwargs)
    chunks = a.chunks
    assert a.schunk.nchunks <= blosc2.MAX_BUFFERSIZE // 8
    assert a[0] == a[-1] == 0
    del a
    reopened = blosc2.open(urlpath)
    assert reopened.shape == (2**49,)
    assert reopened.chunks == chunks
    assert reopened[0] == reopened[-1] == 0


def test_huge_contiguous_zeros():
    a = blosc2.zeros(2**49, dtype=np.float64, contiguous=True)
    assert a.shape == (2**49,)
    assert a.schunk.nchunks <= blosc2.MAX_BUFFERSIZE // 8
    assert len(a.to_cframe()) < 1024
    assert a[0] == a[-1] == 0


def test_multidimensional_edge_chunks_and_padding(monkeypatch):
    # A tiny simulated limit exercises the same growth without huge arrays.
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 1024)
    shape, chunks, blocks = (251, 253), (17, 19), (4, 6)
    result = core._frame_safe_chunks(shape, chunks, blocks, np.uint8, True, contiguous=True)
    assert chunk_count(shape, result) <= 128
    assert padded_chunk_bytes(result, blocks, np.uint8) <= 1024
    assert tuple(chunks) == (17, 19)  # Do not mutate input layouts.
    assert all(c <= s for c, s in zip(result, shape, strict=True))


def test_growth_uses_remaining_padded_byte_budget(monkeypatch):
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 1024)
    # Doubling 80 would exceed the buffer; a partial increase to 128 fits.
    result = core._frame_safe_chunks((16000,), (80,), (16,), np.float64, True, contiguous=True)
    assert result == [128]


def test_offsets_boundary_counts_partial_chunks(monkeypatch):
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 1024)
    chunks = (8,)
    assert core._frame_safe_chunks((1024,), chunks, (1,), np.uint8, False, contiguous=True) == [8]
    with pytest.raises(ValueError, match=r"129 chunks.*offsets limit of 128"):
        core._frame_safe_chunks((1025,), chunks, (1,), np.uint8, False, contiguous=True)


def test_unrepresentable_automatic_frame_layout():
    with pytest.raises(ValueError, match=r"Cannot select automatic chunks.*padded chunk size limit"):
        blosc2.zeros(2**57, dtype=np.float64, contiguous=True)


@pytest.mark.parametrize("shape", [(0,), (0, 100), ()])
def test_empty_and_scalar_frame_layouts(shape):
    a = blosc2.zeros(shape, dtype=np.float64, contiguous=True)
    assert a.shape == shape
