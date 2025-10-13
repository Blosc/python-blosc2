#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

"""
Tests for Unicode path support (Issue #502)

This module tests that blosc2 can properly handle file paths containing
non-ASCII characters (e.g., Chinese, emoji, accented characters).
"""

import os
import sys

import numpy as np
import pytest

import blosc2

# Test filenames with various Unicode characters
UNICODE_FILENAMES = [
    "test_ascii.b2nd",  # Baseline: ASCII
    "test_测试.b2nd",  # Chinese characters
    "test_тест.b2nd",  # Cyrillic characters
    "test_δοκιμή.b2nd",  # Greek characters
    "test_テスト.b2nd",  # Japanese characters
    "test_café.b2nd",  # Accented characters
    "test_🎉.b2nd",  # Emoji (may not work on all systems)
]


@pytest.mark.parametrize("filename", UNICODE_FILENAMES)
def test_unicode_path_schunk(tmp_path, filename):
    """Test SChunk creation and opening with Unicode paths"""
    # Skip emoji test on Windows due to C runtime limitations
    if "🎉" in filename and sys.platform == "win32":
        pytest.skip("Emoji paths may not work on Windows C runtime")

    urlpath = str(tmp_path / filename)

    # Create a SChunk
    data = b"Hello, World! " * 100
    schunk = blosc2.SChunk(
        chunksize=len(data) // 4,
        data=data,
        urlpath=urlpath,
        mode="w",
        contiguous=True,
    )

    # Verify it was created
    assert os.path.exists(urlpath), f"File not created: {urlpath}"

    # Open the SChunk
    schunk2 = blosc2.open(urlpath)

    # Verify data integrity
    assert schunk.nchunks == schunk2.nchunks
    assert schunk[:] == schunk2[:]


@pytest.mark.parametrize("filename", UNICODE_FILENAMES)
def test_unicode_path_ndarray(tmp_path, filename):
    """Test NDArray creation and opening with Unicode paths"""
    # Skip emoji test on Windows due to C runtime limitations
    if "🎉" in filename and sys.platform == "win32":
        pytest.skip("Emoji paths may not work on Windows C runtime")

    urlpath = str(tmp_path / filename)

    # Create an NDArray
    data = np.arange(1000, dtype=np.int32)
    arr = blosc2.asarray(data, urlpath=urlpath, mode="w")

    # Verify it was created
    assert os.path.exists(urlpath), f"File not created: {urlpath}"

    # Open the NDArray
    arr2 = blosc2.open(urlpath)

    # Verify data integrity
    assert arr.shape == arr2.shape
    assert np.array_equal(arr[:], arr2[:])


@pytest.mark.parametrize(
    "dirname",
    [
        "测试目录",  # Chinese directory name
        "тест_папка",  # Cyrillic directory name
        "café_dir",  # Accented directory name
    ],
)
def test_unicode_directory_path(tmp_path, dirname):
    """Test creating files in directories with Unicode names"""
    dir_path = tmp_path / dirname
    dir_path.mkdir()

    urlpath = str(dir_path / "test_array.b2nd")

    # Create an NDArray in the Unicode-named directory
    data = np.arange(100, dtype=np.int32)
    arr = blosc2.asarray(data, urlpath=urlpath, mode="w")

    # Open the NDArray
    arr2 = blosc2.open(urlpath)

    # Verify data integrity
    assert np.array_equal(arr[:], arr2[:])


def test_unicode_path_remove(tmp_path):
    """Test that remove_urlpath works with Unicode paths"""
    filename = "测试_remove.b2nd"
    urlpath = str(tmp_path / filename)

    # Create a file
    data = np.arange(100, dtype=np.int32)
    blosc2.asarray(data, urlpath=urlpath, mode="w")

    # Verify it exists
    assert os.path.exists(urlpath)

    # Remove it using blosc2's remove_urlpath
    blosc2.remove_urlpath(urlpath)

    # Verify it was removed
    assert not os.path.exists(urlpath)


def test_unicode_path_modes(tmp_path):
    """Test different modes (r, w, a) with Unicode paths"""
    filename = "测试_modes.b2nd"
    urlpath = str(tmp_path / filename)

    # Test 'w' mode - create new
    data1 = np.arange(100, dtype=np.int32)
    arr1 = blosc2.asarray(data1, urlpath=urlpath, mode="w")
    del arr1

    # Test 'r' mode - read only
    arr2 = blosc2.open(urlpath, mode="r")
    assert np.array_equal(arr2[:], data1)
    del arr2

    # Test 'a' mode - append/modify
    arr3 = blosc2.open(urlpath, mode="a")
    assert np.array_equal(arr3[:], data1)
    del arr3


@pytest.mark.parametrize("mmap_mode", ["r", "r+", "c"])
def test_unicode_path_mmap(tmp_path, mmap_mode):
    """Test memory-mapped mode with Unicode paths"""
    if sys.platform == "win32" and mmap_mode == "c":
        pytest.skip("Cannot test mmap_mode 'c' on Windows")

    filename = "测试_mmap.b2frame"
    urlpath = str(tmp_path / filename)

    # Create a SChunk with mmap_mode
    data = b"Test data " * 100
    schunk = blosc2.SChunk(
        chunksize=len(data) // 4,
        data=data,
        urlpath=urlpath,
        mmap_mode="w+",
        contiguous=True,
    )
    del schunk

    # Open with specified mmap_mode
    schunk2 = blosc2.open(urlpath, mmap_mode=mmap_mode)

    # Verify we can read the data
    assert len(schunk2[:]) > 0


def test_unicode_path_pathlib(tmp_path):
    """Test that pathlib.Path objects with Unicode work"""

    filename = "测试_pathlib.b2nd"
    urlpath = tmp_path / filename

    # Create with pathlib.Path
    data = np.arange(100, dtype=np.int32)
    arr = blosc2.asarray(data, urlpath=urlpath, mode="w")

    # Open with pathlib.Path
    arr2 = blosc2.open(urlpath)

    # Verify
    assert np.array_equal(arr[:], arr2[:])


def test_unicode_path_mixed_characters(tmp_path):
    """Test path with mixed Unicode scripts"""
    filename = "测试_test_テスト_тест_δοκιμή.b2nd"
    urlpath = str(tmp_path / filename)

    # Create an array
    data = np.arange(100, dtype=np.int32)
    arr = blosc2.asarray(data, urlpath=urlpath, mode="w")

    # Open and verify
    arr2 = blosc2.open(urlpath)
    assert np.array_equal(arr[:], arr2[:])


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
