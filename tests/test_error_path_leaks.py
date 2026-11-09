#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Calls that fail must not keep C memory, Python buffers or files.

The C memory checks count the bytes that glibc's malloc has in use (mallinfo2), so they only run
on glibc.  Python objects come from pymalloc arenas, which malloc does not count: after a warm-up,
repeating a call that frees what it allocates leaves the count where it was."""

import ctypes
import gc
import os
import sys
import weakref

import numpy as np
import pytest

import blosc2


class _Mallinfo2(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_size_t)
        for name in (
            "arena",
            "ordblks",
            "smblks",
            "hblks",
            "hblkhd",
            "usmblks",
            "fsmblks",
            "uordblks",
            "fordblks",
            "keepcost",
        )
    ]


def _get_mallinfo2():
    if not sys.platform.startswith("linux"):
        return None
    try:
        mallinfo2 = ctypes.CDLL(None).mallinfo2  # glibc >= 2.33
    except (OSError, AttributeError):
        return None
    mallinfo2.restype = _Mallinfo2
    return mallinfo2


_mallinfo2 = _get_mallinfo2()


def _c_bytes_left_per_call(fn, ncalls=300, nrounds=3):
    """Smallest growth of malloc's in-use bytes per call of `fn` over `nrounds` rounds."""
    for _ in range(50):  # warm-up: caches and other one-time allocations
        fn()
    growth = []
    for _ in range(nrounds):
        gc.collect()
        info = _mallinfo2()
        before = info.uordblks + info.hblkhd
        for _ in range(ncalls):
            fn()
        gc.collect()
        info = _mallinfo2()
        growth.append((info.uordblks + info.hblkhd - before) / ncalls)
    return min(growth)


@pytest.fixture
def missing_dir_path(tmp_path):
    # A file cannot be created there: the C call that makes the array fails
    return str(tmp_path / "no_such_dir" / "array.b2nd")


@pytest.fixture
def arr():
    return blosc2.asarray(np.arange(2000, dtype=np.int32).reshape(100, 20), chunks=(50, 20), blocks=(10, 20))


@pytest.mark.skipif(_mallinfo2 is None, reason="needs glibc's mallinfo2()")
@pytest.mark.parametrize(
    ("call", "exc", "match", "limit"),
    [
        # The b2nd context (about 1 kB) was leaked when the call raised
        pytest.param(
            lambda arr, path: blosc2.concat([arr, arr], urlpath=path, mode="w"),
            RuntimeError,
            "Error while concatenating the arrays",
            256,
            id="concat",
        ),
        pytest.param(
            lambda arr, path: blosc2.empty((100,), dtype=np.int32, urlpath=path, mode="w"),
            RuntimeError,
            "Could not build empty array",
            256,
            id="empty",
        ),
        pytest.param(
            lambda arr, path: blosc2.full((100,), "abc", dtype=np.int32),
            ValueError,
            "invalid literal",
            256,
            id="full",
        ),
        pytest.param(
            lambda arr, path: arr.copy(urlpath=path, mode="w"),
            RuntimeError,
            "Error while copying the array",
            256,
            id="copy",
        ),
        # The index arrays and the output buffer (about 180 bytes) were leaked
        pytest.param(
            lambda arr, path: arr.oindex[[0, 1000], [0]],
            RuntimeError,
            "Error while getting orthogonal selection",
            64,
            id="oindex",
        ),
    ],
)
def test_failed_call_frees_c_memory(call, exc, match, limit, arr, missing_dir_path):
    def fail():
        with pytest.raises(exc, match=match):
            call(arr, missing_dir_path)

    assert _c_bytes_left_per_call(fail) < limit


@pytest.mark.skipif(_mallinfo2 is None, reason="needs glibc's mallinfo2()")
@pytest.mark.parametrize("meta", [None, {"m": 1, "n": [1, 2, 3]}])
def test_new_array_frees_its_parameters(meta):
    # Every array creation leaked the cparams and dparams it was made with (about 200 bytes), and
    # the names and contents of its metalayers
    assert _c_bytes_left_per_call(lambda: blosc2.empty((100,), dtype=np.int32, meta=meta)) < 64


def test_too_many_metalayers():
    # One more than fits overflowed a fixed-size table (the process aborted)
    with pytest.raises(ValueError, match="at most 15 metalayers"):
        blosc2.empty((10,), dtype=np.int32, meta={f"m{i}": i for i in range(16)})
    assert len(blosc2.empty((10,), dtype=np.int32, meta={f"m{i}": i for i in range(15)}).schunk.meta) == 16


def _assert_released(make_input, call):
    x = make_input()
    ref = weakref.ref(x)
    call(x)
    del x
    gc.collect()
    assert ref() is None, "the failed call still holds a buffer of its input"


def test_asarray_error_releases_input(missing_dir_path):
    def call(x):
        with pytest.raises(RuntimeError, match="Error while creating the NDArray"):
            blosc2.asarray(x, urlpath=missing_dir_path, mode="w")

    _assert_released(lambda: np.arange(1000, dtype=np.int32), call)


def test_setitem_error_releases_value(arr):
    def call(x):
        with pytest.raises((RuntimeError, ValueError)):
            arr[0:10, 0:20] = x

    _assert_released(lambda: np.zeros(4, dtype=np.int32), call)

    def call_c(x):
        with pytest.raises(RuntimeError, match="Error while setting the slice"):
            arr.set_slice(([0, 0], [10, 20]), x)

    _assert_released(lambda: np.zeros(4, dtype=np.int32), call_c)


def test_oindex_setitem_error_releases_value(arr):
    def call(x):
        with pytest.raises((RuntimeError, IndexError)):
            arr.oindex[[0, 1000], [0, 1]] = x

    _assert_released(lambda: np.zeros((2, 2), dtype=np.int32), call)

    def call_c(x):
        with pytest.raises(RuntimeError, match="orthogonal selection"):
            arr.set_oindex_numpy([[0, 1000], [0, 1]], x)

    _assert_released(lambda: np.zeros((2, 2), dtype=np.int32), call_c)


@pytest.mark.parametrize("func", [blosc2.schunk_from_cframe, blosc2.ndarray_from_cframe])
def test_from_cframe_error_releases_input(func):
    ba = bytearray(2000)
    with pytest.raises(RuntimeError, match="from the cframe"):
        func(ba)
    ba.extend(b"x")  # BufferError while a buffer of it is still exported


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="c-blosc2's blosc2_create_cctx() frees its _aligned_malloc'ed context with free() on this error path",
)
def test_compress2_error_releases_input():
    ba = bytearray(1000)
    with pytest.raises(RuntimeError, match="Could not create the compression context"):
        blosc2.compress2(ba, filters=[10])  # no such filter
    ba.extend(b"x")  # BufferError while a buffer of it is still exported


def test_schunk_buffer_errors_release_buffer():
    # A bytearray cannot be resized (BufferError) while a buffer of it is still exported
    schunk = blosc2.SChunk(chunksize=400, data=np.arange(1000, dtype=np.int32))
    dst = bytearray()
    with pytest.raises(ValueError, match="The dst length must be greater than 0"):
        schunk.decompress_chunk(0, dst)
    dst.extend(b"x")
    out = bytearray(4)
    with pytest.raises(ValueError, match="Not enough space"):
        schunk.get_slice(0, 100, out=out)
    out.extend(b"x")


def _open_fds():
    for d in ("/proc/self/fd", "/dev/fd"):
        if os.path.isdir(d):
            return len(os.listdir(d))
    pytest.skip("cannot count open file descriptors here")


def _bad_b2nd_meta(urlpath):
    # A frame with a b2nd metalayer that cannot be decoded
    blosc2.SChunk(chunksize=400, urlpath=urlpath, mode="w", meta={"b2nd": [1, 2, 3]})


def _ndarray(urlpath):
    blosc2.asarray(np.arange(100), urlpath=urlpath, mode="w")


def _schunk(urlpath):
    blosc2.SChunk(chunksize=400, urlpath=urlpath, mode="w", data=np.arange(100))


@pytest.mark.skipif(blosc2.IS_WASM, reason="no file descriptors to count")
@pytest.mark.parametrize(
    ("make", "kwargs", "exc", "match"),
    [
        pytest.param(_bad_b2nd_meta, {}, RuntimeError, "Could not create array from schunk", id="b2nd-meta"),
        pytest.param(
            _bad_b2nd_meta,
            {"mmap_mode": "r"},
            RuntimeError,
            "Could not create array from schunk",
            id="b2nd-meta-mmap",
        ),
        # check_schunk_params() raises after the array (or super-chunk) was made
        pytest.param(
            _ndarray,
            {"cparams": {"typesize": 2}, "typesize": 2},
            ValueError,
            "Cannot change typesize",
            id="params-ndarray",
        ),
        pytest.param(
            _schunk,
            {"cparams": {"typesize": 2}, "typesize": 2},
            ValueError,
            "Cannot change typesize",
            id="params-schunk",
        ),
    ],
)
def test_open_error_closes_file(tmp_path, make, kwargs, exc, match):
    # open() raised after it opened the file and kept it open
    urlpath = str(tmp_path / "frame.b2frame")
    make(urlpath)
    with pytest.raises(exc, match=match):
        blosc2.open(urlpath, **kwargs)
    gc.collect()
    nfds = _open_fds()
    for _ in range(5):
        with pytest.raises(exc, match=match):
            blosc2.open(urlpath, **kwargs)
    gc.collect()
    assert _open_fds() <= nfds
