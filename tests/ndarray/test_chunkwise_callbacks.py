#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2

# Whole arrays, aligned whole chunks, mixed whole/partial chunks, and one partial chunk.
KEYS = [slice(None), slice(8, 24), slice(4, 28), slice(4, 8)]


def make_array(values, chunks=(8,), blocks=(4,)):
    return blosc2.asarray(
        values, chunks=chunks, blocks=blocks, cparams={"nthreads": 1}, dparams={"nthreads": 1}
    )


@pytest.mark.parametrize("key", KEYS)
def test_assignment_applies_source_postfilter(key):
    expected = np.zeros(32, dtype=np.int32)
    values = np.arange(expected[key].size, dtype=np.int32)
    src = make_array(values)
    dst = make_array(expected)

    @src.schunk.postfilter(np.int32, np.int32)
    def copy_source_postfilter(inp, out, offset):
        out[:] = inp + 100

    try:
        expected[key] = values + 100
        dst[key] = src
        np.testing.assert_array_equal(dst[:], expected)
    finally:
        src.schunk.remove_postfilter("copy_source_postfilter")


@pytest.mark.parametrize("scalar", [True, False], ids=["scalar", "array"])
@pytest.mark.parametrize("key", KEYS)
def test_assignment_applies_destination_prefilter(key, scalar):
    expected = np.zeros(32, dtype=np.int32)
    dst = make_array(expected)
    reference = make_array(expected)
    values = 7 if scalar else np.arange(expected[key].size, dtype=np.int32)
    src = values if scalar else make_array(values)

    @dst.schunk.prefilter(np.int32, np.int32)
    def copy_destination_prefilter(inp, out, offset):
        out[:] = inp + 100

    @reference.schunk.prefilter(np.int32, np.int32)
    def reference_destination_prefilter(inp, out, offset):
        out[:] = inp + 100

    try:
        # NumPy input takes the ordinary set_slice path. A partial write recompresses its
        # containing chunks, so their existing items also pass through the prefilter.
        reference[key] = np.full(expected[key].shape, values, dtype=np.int32) if scalar else values
        dst[key] = src
        np.testing.assert_array_equal(dst[:], reference[:])
        np.testing.assert_array_equal(dst[key], np.broadcast_to(values + 100, expected[key].shape))
    finally:
        dst.schunk.remove_prefilter("copy_destination_prefilter")
        reference.schunk.remove_prefilter("reference_destination_prefilter")


def test_large_asarray_copy_applies_source_postfilter():
    # Exactly 16 MiB reaches asarray's chunk-wise path, rather than its small-array path.
    values = np.zeros(2**22, dtype=np.int32)
    src = make_array(values, chunks=(2**18,), blocks=(2**16,))

    @src.schunk.postfilter(np.int32, np.int32)
    def large_copy_source_postfilter(inp, out, offset):
        out[:] = inp + 100

    try:
        dst = blosc2.asarray(src, copy=True, cparams={"nthreads": 1}, dparams={"nthreads": 1})
        np.testing.assert_array_equal(dst[:], values + 100)
    finally:
        src.schunk.remove_postfilter("large_copy_source_postfilter")


@pytest.mark.parametrize("callback", ["prefilter", "postfilter"])
def test_unrelated_callbacks_keep_chunkwise_fast_paths(callback, monkeypatch):
    owner = make_array(np.zeros(32, dtype=np.int32))
    src = make_array(np.arange(32, dtype=np.int32))
    dst = make_array(np.zeros(32, dtype=np.int32))

    @getattr(owner.schunk, callback)(np.int32, np.int32)
    def unrelated_copy_callback(inp, out, offset):
        out[:] = inp + 100

    special_chunks = []
    copied_chunks = []
    original_constant_chunk = blosc2.NDArray._constant_chunk
    original_get_chunk = blosc2.SChunk.get_chunk

    def constant_chunk(self, fill):
        special_chunks.append(self)
        return original_constant_chunk(self, fill)

    def get_chunk(self, nchunk):
        if self is src.schunk:
            copied_chunks.append(nchunk)
        return original_get_chunk(self, nchunk)

    monkeypatch.setattr(blosc2.NDArray, "_constant_chunk", constant_chunk)
    monkeypatch.setattr(blosc2.SChunk, "get_chunk", get_chunk)
    try:
        dst[:] = 7
        np.testing.assert_array_equal(dst[:], np.full(32, 7, dtype=np.int32))
        assert special_chunks == [dst]
        dst[:] = src
        np.testing.assert_array_equal(dst[:], np.arange(32, dtype=np.int32))
        assert copied_chunks == [0, 1, 2, 3]
    finally:
        getattr(owner.schunk, "remove_" + callback)("unrelated_copy_callback")


@pytest.mark.parametrize("callback", ["prefilter", "postfilter"])
def test_handle_callback_flags_follow_registration(callback):
    owner = make_array(np.zeros(32, dtype=np.int32))
    other = make_array(np.zeros(32, dtype=np.int32))
    flag = "_has_" + callback
    assert not getattr(owner.schunk, flag)

    @getattr(owner.schunk, callback)(np.int32, np.int32)
    def handle_copy_callback(inp, out, offset):
        out[:] = inp

    try:
        assert getattr(owner.schunk, flag)
        assert not getattr(other.schunk, flag)
    finally:
        getattr(owner.schunk, "remove_" + callback)("handle_copy_callback")
    assert not getattr(owner.schunk, flag)
