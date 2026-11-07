#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

# Orthogonal selections copy typesize bytes per item in C into a NumPy buffer allocated for
# the array's dtype.  Arrays whose dtype has smaller items than the typesize must raise
# instead of overflowing that buffer.

import shutil
from pathlib import Path

import numpy as np
import pytest

import blosc2

# Written by c-blosc2's examples/b2nd/example_frame_generator.c (c-blosc2 2.7-2.16 store
# "|u1" in the b2nd metalayer of arrays created without a dtype): float32 items, typesize 4
LEGACY_U1_FILE = Path(__file__).parent.parent / "data" / "legacy_u1_dtype" / "example_day_month_temp.b2nd"


def with_dtype_string(arr, old, new):
    """`arr` with dtype string `old` replaced by `new` (same length) in its b2nd metalayer"""
    assert len(old) == len(new)
    cframe = arr.to_cframe()
    # The metalayers come before the chunks, so the first match is the dtype
    return blosc2.ndarray_from_cframe(cframe.replace(old.encode(), new.encode(), 1), copy=True)


def assert_oindex_raises(arr):
    """The paths to the C orthogonal selection raise ValueError and leave the array unchanged"""
    before = arr.tobytes()
    full = [list(range(n)) for n in arr.shape]
    match = f"typesize is {arr.schunk.typesize}"
    with pytest.raises(ValueError, match=match):
        arr.get_oselection_numpy(full)
    with pytest.raises(ValueError, match=match):
        arr.get_oselection_numpy([[0]] * arr.ndim)  # one item: a few bytes too many
    with pytest.raises(ValueError, match=match):
        arr.oindex[tuple(full)] = np.zeros(arr.shape, dtype=arr.dtype)
    if arr.ndim > 1:
        with pytest.raises(ValueError, match=match):
            arr[[0, 1]]  # through take(axis=0)
    assert arr.tobytes() == before


def test_legacy_u1_file(tmp_path):
    path = tmp_path / LEGACY_U1_FILE.name
    shutil.copy(LEGACY_U1_FILE, path)
    arr = blosc2.open(path, mode="a")
    assert arr.dtype == np.uint8
    assert arr.schunk.typesize == 4
    assert arr.shape == (400, 3)
    assert_oindex_raises(arr)
    assert path.read_bytes() == LEGACY_U1_FILE.read_bytes()


@pytest.mark.parametrize(
    ("dtype", "shape", "chunks", "blocks", "stored"),
    [
        # As c-blosc2's plugins/test_data/example_item_prices.b2nd
        (np.float32, (12, 25, 250), (8, 10, 50), (4, 5, 10), "|u1"),
        (np.float64, (100,), (30,), (7,), "|u1"),
        # A damaged dtype string
        (np.float64, (6, 10), (4, 4), (2, 2), "<f2"),
    ],
)
def test_dtype_smaller_than_typesize(dtype, shape, chunks, blocks, stored):
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    arr = with_dtype_string(blosc2.asarray(data, chunks=chunks, blocks=blocks), data.dtype.str, stored)
    assert arr.dtype == np.dtype(stored)
    assert arr.schunk.typesize == data.itemsize
    assert_oindex_raises(arr)


def test_oindex_buffer_size():
    shape = (6, 10)
    data = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    arr = blosc2.asarray(data, chunks=(4, 4), blocks=(2, 2))
    key = [[5, 0, 2], [9, 1]]
    expected = data[np.ix_(*key)]
    out = np.empty(expected.shape, dtype=arr.dtype)
    np.testing.assert_array_equal(arr.get_oindex_numpy(out, key), expected)
    with pytest.raises(ValueError, match="destination buffer is smaller"):
        arr.get_oindex_numpy(np.empty(expected.size - 1, dtype=arr.dtype), key)
    with pytest.raises(ValueError, match="source buffer is smaller"):
        arr.set_oindex_numpy(key, np.empty(expected.size - 1, dtype=arr.dtype))
    np.testing.assert_array_equal(arr[...], data)
    # A larger buffer is fine, as for slices
    out = np.zeros(expected.size + 3, dtype=arr.dtype)
    arr.get_oindex_numpy(out, key)
    np.testing.assert_array_equal(out[: expected.size].reshape(expected.shape), expected)
    # Empty selections need no buffer
    arr.get_oindex_numpy(np.empty(0, dtype=arr.dtype), [[], [1]])
    arr.set_oindex_numpy([[2], []], np.empty(0, dtype=arr.dtype))
    np.testing.assert_array_equal(arr[...], data)


@pytest.mark.parametrize(
    "dtype",
    ["?", ">i4", "S7", "<U100", [("a", "<i4"), ("b", "<f8")]],
)
def test_oindex_valid_dtypes(dtype):
    # The typesize of valid arrays is the dtype's itemsize, also above 255 bytes ("<U100")
    dtype = np.dtype(dtype)
    shape = (5, 7)
    data = np.arange(np.prod(shape)).astype(dtype).reshape(shape)
    arr = blosc2.asarray(data, chunks=(3, 4), blocks=(2, 2))
    assert arr.schunk.typesize == dtype.itemsize
    key = [[4, 0, 2], [6, 1, 1]]
    np.testing.assert_array_equal(arr.oindex[tuple(key)], data[np.ix_(*key)])
    key = [[4, 0, 2], [6, 1, 3]]
    value = data[np.ix_([0, 1, 2], [0, 1, 2])]
    arr.oindex[tuple(key)] = value
    expected = data.copy()
    expected[np.ix_(*key)] = value
    np.testing.assert_array_equal(arr[...], expected)
