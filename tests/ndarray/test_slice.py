#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import ctypes
import gc
import sys

import numpy as np
import pytest

import blosc2
from blosc2.ndarray import detect_aligned_chunks

# --- direct unit tests for detect_aligned_chunks ----------------------------
# Regression coverage for a bug found via Caterva2: n_chunks per dimension was
# computed with floor division, undercounting any dimension whose size isn't
# an exact multiple of its chunk size. That corrupted the flat chunk-index
# math for any aligned slice with a nonzero start in an earlier dimension,
# silently mapping it onto a *different* chunk instead of raising or
# returning []. These test the helper directly (not just through
# NDArray.slice(), the only current call site), including consecutive=True,
# which has no other coverage at all since no call site uses it yet.


def test_detect_aligned_chunks_exact_multiple_shape():
    # Sanity: exact-multiple shape, was never affected by the bug.
    assert detect_aligned_chunks((slice(5, 10), slice(0, 10)), (10, 20), (5, 10)) == [2]


def test_aligned_chunks_non_exact_multiple():
    # The bug repro: dim 1 (100_003) isn't a multiple of its chunk (40_000),
    # so its true chunk count is 3, not 100_003 // 40_000 == 2. Before the
    # fix this returned [2] (row 0, col chunk 2) instead of the correct [3]
    # (row 1, col chunk 0).
    assert detect_aligned_chunks((slice(1, 2), slice(0, 40_000)), (2, 100_003), (1, 40_000)) == [3]


def test_aligned_chunks_unaligned_gives_empty():
    # A slice boundary that isn't a chunk multiple must short-circuit to [],
    # regardless of the n_chunks bug (this check runs before n_chunks is
    # even computed).
    assert detect_aligned_chunks((slice(1, 2), slice(0, 40_001)), (2, 100_003), (1, 40_000)) == []


def test_aligned_chunks_middle_dim_non_exact():
    # 3D, non-exact-multiple dim in the *middle* (not last) position, offset
    # in the first dim -- pins that the fix's multiplier chain is right in
    # general, not just for the 2D case (last dim == only non-first dim)
    # that surfaced the bug.
    key = (slice(2, 4), slice(0, 3), slice(0, 5))
    assert detect_aligned_chunks(key, (4, 7, 5), (2, 3, 5)) == [3]


def test_aligned_chunks_several_non_exact_dims():
    # Both non-first dims are non-exact-multiple at once.
    key = (slice(2, 4), slice(0, 3), slice(0, 4))
    assert detect_aligned_chunks(key, (4, 7, 11), (2, 3, 4)) == [9]


def test_detect_aligned_chunks_consecutive_true():
    # consecutive=True has no call site today (NDArray.slice always passes
    # False), so this is its only coverage. Exact-multiple shape: the
    # requested region is the whole array, so all 4 chunks are consecutive.
    key = (slice(0, 10), slice(0, 20))
    assert detect_aligned_chunks(key, (10, 20), (5, 10), consecutive=True) == [0, 1, 2, 3]


def test_aligned_chunks_consecutive_flag_false():
    # Same grid, a region whose chunks are NOT consecutive in flat order.
    key = (slice(0, 5), slice(0, 10))
    assert detect_aligned_chunks(key, (10, 30), (5, 10), consecutive=True) == [0]


def test_aligned_chunks_consecutive_non_exact():
    # The bug pattern (non-exact-multiple trailing dim) under
    # consecutive=True: before the fix, the corrupted flat indices could
    # come out consecutive when they shouldn't (or vice versa), since the
    # consecutive check runs on the same (wrong) flat_index values. Here the
    # true flat indices for row 0 and row 1's first chunk are 0 and 3 -- not
    # consecutive -- so this must return [].
    key = (slice(0, 2), slice(0, 40_000))
    assert detect_aligned_chunks(key, (2, 100_003), (1, 40_000), consecutive=True) == []
    # Same key with consecutive=False must still find both chunks.
    assert detect_aligned_chunks(key, (2, 100_003), (1, 40_000), consecutive=False) == [0, 3]


argnames = "shape, chunks, blocks, slices, dtype"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    ([12, 13, 14, 15, 16], [5, 5, 5, 5, 5], [2, 2, 2, 2, 2], (slice(1, 3), ..., slice(3, 6)), np.float32),
    # Consecutive slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 10), slice(0, 100), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 100), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 50)), np.int32),
    # Aligned slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 50), slice(25, 100), slice(50, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 40), slice(25, 75), slice(100, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 35), slice(50, 75), slice(100, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 25), slice(25, 50), slice(50, 100)), np.int32),
    # Aligned slices on shapes NOT an exact multiple of chunks in some dim
    # (regression: detect_aligned_chunks used floor division to count chunks
    # per dimension, undercounting a dim with a trailing partial chunk; this
    # corrupted the flat chunk-index math for any aligned slice with a
    # nonzero start in an earlier dimension, silently returning a different
    # chunk's data).
    ((2, 100_003), (1, 40_000), (1, 10_000), (slice(1, 2), slice(0, 40_000)), np.float64),
    ((2, 100_003), (1, 40_000), (1, 10_000), (slice(1, 2), slice(40_000, 80_000)), np.float64),
    ((2, 100_003), (1, 40_000), (1, 10_000), (slice(1, 2), slice(80_000, 100_003)), np.float64),
    ((6, 100_003), (1, 40_000), (1, 10_000), (slice(3, 5), slice(0, 40_000)), np.float64),
    # Same bug, but with the non-exact-multiple dim in the *middle* of a 3D
    # array (not the last dim) and multiple non-exact-multiple dims at once
    # -- pins that the fix generalizes past the 2D case that was found by.
    ((4, 7, 5), (2, 3, 5), (1, 1, 5), (slice(2, 4), slice(0, 3), slice(0, 5)), np.float64),
    ((4, 7, 11), (2, 3, 4), (1, 1, 4), (slice(2, 4), slice(0, 3), slice(0, 4)), np.float64),
    # Non-consecutive slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 10), slice(0, 100), slice(0, 300 - 1)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 100 - 1), slice(0, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5 - 1), slice(0, 25), slice(0, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(0, 5), slice(0, 25), slice(0, 50 - 1)), np.int32),
    # Non-aligned slices
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 50 - 1), slice(25, 100), slice(50, 300)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(10, 40), slice(25, 75 - 1), slice(100, 200)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20, 35), slice(50, 75), slice(100, 300 - 1)), np.int32),
    ((10, 100, 300), (5, 25, 50), (1, 5, 10), (slice(20 + 1, 25), slice(25, 50), slice(50, 100)), np.int32),
    # int indices drop their dims
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, slice(3, 17), slice(5, 29)), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (slice(1, 3), slice(3, 17), 4), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, 5, slice(5, 29)), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, 5, 7), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, slice(5, 6), slice(5, 29)), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (slice(2, 3), slice(3, 17), slice(5, 29)), np.int32),
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, slice(8, 16), slice(8, 16)), np.int32),  # aligned
    ((2, 2, 2), (1, 1, 1), (1, 1, 1), (1, 0, 1), np.int32),  # aligned
    ((7, 20, 30), (1, 8, 8), (1, 4, 4), (2, slice(3, 17, 2), slice(5, 29)), np.int32),  # step
]


@pytest.mark.parametrize(argnames, argvalues)
def test_slice(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = a.slice(slices)
    np_slice = a[slices]
    assert b.shape == np_slice.shape
    np.testing.assert_almost_equal(b[...], np_slice)
    assert b.base is None  # a new array, not a view


@pytest.mark.parametrize(argnames, argvalues)
def test_slice_codec_and_clevel(shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(
        nparray,
        chunks=chunks,
        blocks=blocks,
        cparams={"codec": blosc2.Codec.LZ4, "clevel": 6, "filters": [blosc2.Filter.BITSHUFFLE]},
    )

    b = a.slice(slices)
    assert b.cparams.codec == a.cparams.codec
    assert b.cparams.clevel == a.cparams.clevel
    assert b.cparams.filters == a.cparams.filters


argnames = "shape, chunks, blocks, slices, dtype, chunks2, blocks2"
argvalues = [
    ([456], [258], [73], slice(0, 1), np.int32, [1], [1]),
    ([77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64, [3, 50], None),
    (
        [12, 13, 14, 15, 16],
        [5, 5, 5, 5, 5],
        [2, 2, 2, 2, 2],
        (slice(1, 3), ..., slice(3, 6)),
        np.float32,
        None,
        [2, 3, 3, 5, 2],
    ),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_slice_chunks_blocks(shape, chunks, blocks, chunks2, blocks2, slices, dtype):
    size = int(np.prod(shape))
    nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = a.slice(slices, chunks=chunks2, blocks=blocks2)
    np_slice = a[slices]
    np.testing.assert_almost_equal(b[...], np_slice)


# --- int indices and storage arguments --------------------------------------
# The result of slice() owns its data and has the storage asked for.

_INT_KEY = (2, slice(3, 17), slice(5, 29))  # not chunk-aligned
_SLICES_KEY = (slice(1, 3), slice(3, 17), slice(5, 29))  # not chunk-aligned
_ALIGNED_INT_KEY = (2, slice(8, 16), slice(8, 16))
_ALIGNED_SLICES_KEY = (slice(2, 4), slice(8, 16), slice(8, 16))


def _array_3d():
    nparray = np.arange(7 * 20 * 30, dtype=np.int32).reshape(7, 20, 30)
    return nparray, blosc2.asarray(nparray, chunks=(1, 8, 8), blocks=(1, 4, 4))


_ALL_KEYS = [_INT_KEY, _SLICES_KEY, _ALIGNED_INT_KEY, _ALIGNED_SLICES_KEY]


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_slice_storage(tmp_path, key):
    nparray, a = _array_3d()
    urlpath = str(tmp_path / "slice.b2nd")
    b = a.slice(key, urlpath=urlpath, mode="w", meta={"m": [1, 2]})
    assert b.schunk.urlpath == urlpath
    assert b.schunk.contiguous
    for c in (b, blosc2.open(urlpath)):
        assert c.shape == nparray[key].shape
        assert c.schunk.meta["m"] == [1, 2]
        np.testing.assert_array_equal(c[...], nparray[key])
    assert a.slice(key, contiguous=True).schunk.contiguous


@pytest.mark.parametrize("key", _ALL_KEYS)
def test_slice_cparams_dparams(key):
    nparray, a = _array_3d()
    cparams = {"codec": blosc2.Codec.LZ4, "clevel": 1, "filters": [blosc2.Filter.NOFILTER], "nthreads": 1}
    b = a.slice(key, cparams=cparams, dparams={"nthreads": 1})
    assert (b.cparams.codec, b.cparams.clevel, b.cparams.filters[-1]) == (
        blosc2.Codec.LZ4,
        1,
        blosc2.Filter.NOFILTER,
    )
    assert b.cparams.nthreads == 1
    assert b.dparams.nthreads == 1
    np.testing.assert_array_equal(b[...], nparray[key])


@pytest.mark.parametrize(
    ("key", "chunks", "blocks", "chunks2", "blocks2"),
    [
        (_INT_KEY, (7, 12), (7, 4), (7, 12), (7, 4)),
        (_INT_KEY, [2, 7, 12], [2, 7, 4], (7, 12), (7, 4)),  # with the dropped dim
        ((2, slice(5, 6), slice(5, 29)), [1, 12], [1, 4], (1, 12), (1, 4)),
        (_ALIGNED_INT_KEY, (4, 8), (2, 4), (4, 8), (2, 4)),
    ],
)
def test_slice_int_indices_chunks_blocks(key, chunks, blocks, chunks2, blocks2):
    nparray, a = _array_3d()
    b = a.slice(key, chunks=chunks, blocks=blocks)
    assert b.chunks == chunks2
    assert b.blocks == blocks2
    np.testing.assert_array_equal(b[...], nparray[key])


class _Mallinfo2(ctypes.Structure):
    # all fields: mallinfo2() returns the struct by value
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in "arena ordblks smblks hblks hblkhd usmblks fsmblks uordblks fordblks keepcost".split()
    ]


_mallinfo2 = getattr(ctypes.CDLL(None), "mallinfo2", None) if sys.platform.startswith("linux") else None
if _mallinfo2 is not None:  # glibc >= 2.33
    _mallinfo2.restype = _Mallinfo2


def _malloc_in_use():
    gc.collect()
    info = _mallinfo2()
    return info.uordblks + info.hblkhd


@pytest.mark.skipif(_mallinfo2 is None, reason="needs glibc's mallinfo2()")
@pytest.mark.parametrize("key", [_INT_KEY, _SLICES_KEY])
def test_slice_frees_sliced_array(key):
    # The sliced array (about 75 KiB here) was kept for every slice that was not chunk-aligned
    _, a = _array_3d()
    for _ in range(20):  # warm-up
        a.slice(key)
    growth = []
    for _ in range(3):
        before = _malloc_in_use()
        for _ in range(100):
            a.slice(key)
        growth.append((_malloc_in_use() - before) / 100)
    assert min(growth) < 256
