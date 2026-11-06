#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""The decompression functions stay within the src and dst buffers they are given, and free what they
took when they fail."""

import array
import ctypes
import gc
import sys

import numpy as np
import pytest

import blosc2

NITEMS = 1 << 14
DATA = np.arange(NITEMS, dtype=np.int64)
NBYTES = DATA.nbytes
CHUNK = blosc2.compress2(DATA, codec=blosc2.Codec.LZ4, nthreads=1)

DECOMPRESSORS = [blosc2.decompress, blosc2.decompress2]


def _corrupt_src_error(decompress):
    return RuntimeError if decompress is blosc2.decompress else ValueError


@pytest.mark.parametrize("decompress", DECOMPRESSORS)
@pytest.mark.parametrize(
    "make_dst",
    [
        pytest.param(bytearray, id="bytearray"),
        pytest.param(lambda n: np.zeros(n // 8, dtype=np.int64), id="numpy"),
        pytest.param(lambda n: np.zeros((2, n // 16), dtype=np.int64), id="numpy-2d"),
        pytest.param(lambda n: np.zeros(n // 2, dtype=np.int16), id="numpy-int16"),
        pytest.param(lambda n: memoryview(np.zeros(n // 8, dtype=np.int64)), id="memoryview"),
        pytest.param(lambda n: array.array("q", bytes(n)), id="array"),
        # A part of a larger array, as when decompressing chunks one after another into their place
        pytest.param(
            lambda n: np.zeros(n // 4, dtype=np.int64)[n // 16 : n // 16 + n // 8], id="numpy-slice"
        ),
    ],
)
def test_dst_kinds(decompress, make_dst):
    exact = make_dst(NBYTES)
    assert decompress(CHUNK, exact) is None
    assert bytes(memoryview(exact).cast("B")) == DATA.tobytes()

    larger = make_dst(2 * NBYTES)
    assert decompress(CHUNK, larger) is None
    larger_bytes = memoryview(larger).cast("B")
    assert larger_bytes[:NBYTES] == DATA.tobytes()
    assert larger_bytes[NBYTES:] == bytes(NBYTES)


@pytest.mark.parametrize("decompress", DECOMPRESSORS)
@pytest.mark.parametrize("missing", [1, 8, NBYTES - 1])
def test_dst_too_small(decompress, missing):
    # dst is the start of a larger buffer: writing past its end shows in the rest
    whole = bytearray(NBYTES + 64)
    dst = memoryview(whole)[: NBYTES - missing]
    with pytest.raises(RuntimeError, match=rf"dst length \({NBYTES - missing} bytes\) is smaller"):
        decompress(CHUNK, dst)
    assert whole == bytes(len(whole))


@pytest.mark.parametrize("decompress", DECOMPRESSORS)
def test_empty_dst_is_released(decompress):
    dst = bytearray()
    with pytest.raises(ValueError, match="greater than 0"):
        decompress(CHUNK, dst)
    dst.extend(b"x")  # BufferError while the buffer of dst is still held


@pytest.mark.skipif(sys.maxsize < 2**32, reason="needs a 64-bit address space")
@pytest.mark.parametrize("decompress", DECOMPRESSORS)
def test_dst_larger_than_2gib(decompress):
    # np.zeros() gets zeroed pages from the OS: only the ones written are touched
    dst = np.zeros(2**31 + 8, dtype=np.uint8)
    decompress(CHUNK, dst)
    assert np.array_equal(dst[:NBYTES].view(np.int64), DATA)


@pytest.mark.skipif(sys.maxsize < 2**32, reason="needs a 64-bit address space")
@pytest.mark.parametrize("decompress", DECOMPRESSORS)
def test_src_larger_than_2gib(decompress):
    # A chunk at the start of a larger buffer (say, a memory-mapped file): only its own bytes are read
    src = np.zeros(2**31 + 8, dtype=np.uint8)
    src[: len(CHUNK)] = np.frombuffer(CHUNK, dtype=np.uint8)
    assert decompress(src) == DATA.tobytes()


@pytest.mark.parametrize("decompress", DECOMPRESSORS)
@pytest.mark.parametrize("missing", [1, 100, len(CHUNK) // 2])
def test_truncated_src(decompress, missing):
    # The bytes past the end of src are the rest of the chunk: reading them would succeed
    src = memoryview(bytearray(CHUNK))[: len(CHUNK) - missing]
    with pytest.raises(_corrupt_src_error(decompress)):
        decompress(src)
    with pytest.raises(_corrupt_src_error(decompress)):
        decompress(src, np.zeros(NITEMS, dtype=np.int64))


@pytest.mark.parametrize("decompress", DECOMPRESSORS)
def test_truncated_memcpyed_src(decompress):
    # Incompressible data: the chunk holds it as is after the header
    data = np.random.default_rng(0).integers(0, 256, 1 << 16, dtype=np.uint8)
    chunk = blosc2.compress2(data, typesize=1, clevel=0, nthreads=1)
    assert len(chunk) == data.nbytes + blosc2.MAX_OVERHEAD
    src = memoryview(bytearray(chunk))[: len(chunk) // 2]
    with pytest.raises(_corrupt_src_error(decompress)):
        decompress(src)


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


@pytest.mark.skipif(_mallinfo2 is None, reason="needs glibc's mallinfo2()")
@pytest.mark.parametrize(
    ("call", "exc"),
    [
        pytest.param(
            lambda: blosc2.blosc2_ext.vldecompress_block(b"abc", 0), ValueError, id="vlblock-short-src"
        ),
        pytest.param(
            lambda: blosc2.blosc2_ext.vldecompress_block(3, 0), TypeError, id="vlblock-no-buffer-src"
        ),
        pytest.param(lambda: blosc2.decompress2(CHUNK, 3), TypeError, id="decompress2-no-buffer-dst"),
        pytest.param(
            lambda: blosc2.decompress2(CHUNK, np.zeros(64, dtype=np.int64)[::2]),
            ValueError,
            id="decompress2-strided-dst",
        ),
    ],
)
def test_failed_call_frees_dctx(call, exc):
    def fail():
        with pytest.raises(exc):
            call()

    # The decompression context (about 1.3 kB) was leaked
    assert _c_bytes_left_per_call(fail) < 256
