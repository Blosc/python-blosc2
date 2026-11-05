#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import concurrent.futures
import os
import signal
import sys
import time

import numpy as np
import pytest

import blosc2

# Slices: everything, inside one chunk, across chunks, one item, through the chunks that stay special values
SLICES = [
    np.s_[:],
    np.s_[11:19, 21:39, 17:31],
    np.s_[3:37, 7:44, 9:58],
    np.s_[39, 49, 60],
    np.s_[:, 30:, :30],
]


@pytest.fixture
def on_disk(tmp_path):
    path = str(tmp_path / "prefetch.b2nd")
    a = blosc2.zeros((40, 50, 61), dtype=np.int32, chunks=(10, 20, 16), blocks=(5, 8, 8), urlpath=path)
    # data in part of the array only, so that the other chunks stay special values (zeros) in the index
    values = np.arange(40 * 30 * 61, dtype=np.int64).reshape(40, 30, 61) * 2654435761 % 1000003
    a[:, :30] = values.astype(np.int32)
    expected = a[:]
    del a
    return path, expected


def evict(path):
    """Drop the file from the page cache, so that read-ahead has something to read (where the OS lets us)."""
    if hasattr(os, "posix_fadvise"):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


@pytest.fixture
def prefetch():
    yield blosc2.set_prefetch
    blosc2.set_prefetch(0)


# (nthreads, max_ahead, piece_size): several threads; a budget smaller than any chunk; the smallest pieces;
# pieces larger than the file
SETTINGS = [(4, 0, 0), (1, 1, 0), (3, 0, 64 << 10), (2, 0, 4 << 20)]


def read_all_ways(path, mmap_mode):
    """Every slice through the paths that read ahead: [], slice(), orthogonal selection; and SChunk slices."""
    a = blosc2.open(path, mode="r", mmap_mode=mmap_mode)
    got = []
    for sl in SLICES:
        got.append(a[sl])
        if all(isinstance(k, slice) for k in np.index_exp[sl]):
            got.append(a.slice(sl)[:])
            key = (np.index_exp[sl] + (slice(None),) * 3)[:3]
            got.append(
                a.get_oselection_numpy(
                    [list(range(*k.indices(n))) for k, n in zip(key, a.shape, strict=True)]
                )
            )
    nitems = a.schunk.nbytes // a.schunk.typesize
    got.extend(a.schunk[start : min(start + 20000, nitems)] for start in (0, 7919, nitems - 5000))
    del a
    return got


def assert_same(got, expected):
    assert len(got) == len(expected)
    for g, e in zip(got, expected, strict=True):
        if isinstance(e, bytes):
            assert g == e
        else:
            np.testing.assert_array_equal(g, e)


@pytest.mark.skipif(sys.platform == "win32", reason="read-ahead is a no-op on Windows")
@pytest.mark.parametrize("mmap_mode", [None, "r"])
@pytest.mark.parametrize(("nthreads", "max_ahead", "piece_size"), SETTINGS)
def test_slices(on_disk, prefetch, mmap_mode, nthreads, max_ahead, piece_size):
    path, _ = on_disk
    prefetch(0)
    expected = read_all_ways(path, mmap_mode)
    prefetch(nthreads, max_ahead, piece_size)
    evict(path)
    for _ in range(2):  # cold, then from the page cache
        assert_same(read_all_ways(path, mmap_mode), expected)


@pytest.mark.skipif(sys.platform == "win32", reason="read-ahead is a no-op on Windows")
@pytest.mark.skipif(blosc2.IS_WASM, reason="needs threads")
@pytest.mark.parametrize("mmap_mode", [None, "r"])
def test_threads(on_disk, prefetch, mmap_mode):
    # Several threads read at the same time, each with its own handle: each of their slices gets its own
    # read-ahead threads, so the pool grows with them
    path, _ = on_disk
    prefetch(0)
    expected = read_all_ways(path, mmap_mode)
    prefetch(2)
    evict(path)
    with concurrent.futures.ThreadPoolExecutor(6) as pool:
        for _ in range(2):  # cold, then from the page cache
            for got in pool.map(lambda _: read_all_ways(path, mmap_mode), range(6)):
                assert_same(got, expected)


@pytest.mark.skipif(sys.platform == "win32", reason="read-ahead is a no-op on Windows")
@pytest.mark.parametrize("mmap_mode", [None, "r"])
def test_partial_chunks(tmp_path, prefetch, mmap_mode):
    # Chunks of 2 MiB with 128 KiB blocks of random data, compressed with several threads (blocks stored out of
    # order): of the chunks a slice covers in part, only the blocks it needs are read ahead, in several pieces
    path = str(tmp_path / "partial.b2nd")
    data = np.random.default_rng(0).random((64, 256, 256), dtype=np.float32)
    blosc2.asarray(
        data, chunks=(32, 128, 128), blocks=(8, 32, 128), urlpath=path, mode="w", cparams={"nthreads": 4}
    )
    prefetch(4)
    for sl in [np.s_[5:40, 100:150, 10:200], np.s_[31:33, :, 120:140], np.s_[:, 3:4, :]]:
        evict(path)
        a = blosc2.open(path, mode="r", mmap_mode=mmap_mode)
        np.testing.assert_array_equal(a[sl], data[sl])
        np.testing.assert_array_equal(a.slice(sl)[:], data[sl])
        del a


def test_set_prefetch(prefetch):
    assert prefetch(2) == 0
    assert prefetch(0, 1 << 20) == 2
    assert prefetch(1, 0, 64 << 10) == 0
    with pytest.raises(ValueError):
        prefetch(-1)
    with pytest.raises(ValueError):
        prefetch(1, -1)
    with pytest.raises(ValueError):
        prefetch(1, 0, (64 << 10) - 1)


@pytest.mark.skipif(blosc2.IS_WASM or not hasattr(os, "fork"), reason="needs native fork()")
def test_fork(on_disk, prefetch):
    path, expected = on_disk
    prefetch(4)
    # One decompression thread: C-Blosc2's decompression threads do not survive fork() either, independently
    # of read-ahead (a child decompressing with nthreads > 1 after the parent did hangs)
    old_nthreads = blosc2.set_nthreads(1)
    try:
        evict(path)
        blosc2.open(path, mode="r")[:]  # the read-ahead threads run in this process now
        evict(path)
        _fork_and_read(path, expected)
    finally:
        blosc2.set_nthreads(old_nthreads)


def _fork_and_read(path, expected):
    pid = os.fork()
    if pid == 0:  # the child has to start its own threads
        try:
            got = blosc2.open(path, mode="r", mmap_mode="r")[3:37, 7:44, 9:58]
            os._exit(0 if np.array_equal(got, expected[3:37, 7:44, 9:58]) else 1)
        except BaseException:
            os._exit(2)
    for _ in range(600):
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        pytest.fail("the forked child hung")
    assert os.waitstatus_to_exitcode(status) == 0
