#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import asyncio

import numpy as np
import pytest

import blosc2
from blosc2.utils import get_chunks_idx

argnames = "urlpath, shape, chunks, blocks, slices, dtype"
argvalues = [
    ("b2nd", [456], [258], [73], slice(0, 1), np.int32),
    ("b2nd", [456], [258], [73], slice(0, 3), "f4,f8,i4"),
    (None, [77, 134, 13], [31, 13, 5], [7, 8, 3], (slice(3, 7), slice(50, 100), 7), np.float64),
    (
        "b2nd",
        [12, 13, 14, 15, 16],
        [5, 5, 5, 5, 5],
        [2, 2, 2, 2, 2],
        (slice(1, 3), ..., slice(3, 6)),
        np.float32,
    ),
]


@pytest.mark.parametrize(argnames, argvalues)
def test_ndarray(urlpath, shape, chunks, blocks, slices, dtype):
    size = int(np.prod(shape))
    struct_dtype = False
    if isinstance(dtype, str) and "," in dtype:
        struct_dtype = True
        nparray = np.ones(size, dtype=dtype)
    else:
        nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks)
    b = blosc2.Proxy(a, urlpath=urlpath, mode="w")

    np_slice = a[slices]
    cache_slice = b[slices]
    assert cache_slice.shape == np_slice.shape
    if not struct_dtype:
        np.testing.assert_almost_equal(cache_slice, np_slice)
    else:
        assert cache_slice.dtype == np.dtype(dtype)
        assert b.fields.keys() == cache_slice.dtype.fields.keys()
        for field in cache_slice.dtype.fields:
            np.testing.assert_almost_equal(cache_slice[field], np_slice[field])

    a_slice = a.slice(slices)
    cache_slice = b.fetch(slices)
    assert cache_slice.shape == a.shape
    assert cache_slice.schunk.urlpath == urlpath
    if not struct_dtype:
        np.testing.assert_almost_equal(cache_slice[slices], a_slice[...])
    else:
        assert cache_slice.dtype == np.dtype(dtype)
        assert b.fields.keys() == cache_slice.fields.keys()
        for field in cache_slice.fields:
            np.testing.assert_almost_equal(cache_slice.fields[field][slices], a_slice.fields[field][...])

    cache_arr = b.fetch()
    assert cache_arr.schunk.urlpath == urlpath
    if not struct_dtype:
        np.testing.assert_almost_equal(cache_arr[...], a[...])
    else:
        assert cache_arr.dtype == np.dtype(dtype)
        assert b.fields.keys() == cache_arr.fields.keys()
        for field in cache_arr.fields:
            np.testing.assert_almost_equal(cache_arr.fields[field][...], a.fields[field][...])
    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(argnames, argvalues)
def test_open(urlpath, shape, chunks, blocks, slices, dtype):
    proxy_urlpath = "proxy.b2nd"
    size = int(np.prod(shape))
    struct_dtype = False
    if isinstance(dtype, str) and "," in dtype:
        struct_dtype = True
        nparray = np.ones(size, dtype=dtype)
    else:
        nparray = np.arange(size, dtype=dtype).reshape(shape)
    a = blosc2.asarray(nparray, chunks=chunks, blocks=blocks, urlpath=urlpath)
    b = blosc2.Proxy(a, urlpath=proxy_urlpath, mode="w")
    del a
    del b
    if urlpath is None:
        with pytest.raises(RuntimeError):
            _ = blosc2.open(proxy_urlpath, mode="a")
    else:
        b = blosc2.open(proxy_urlpath, mode="a")
        a = blosc2.open(urlpath, mode="r")
        if not struct_dtype:
            np.testing.assert_almost_equal(b[...], a[...])
        else:
            assert b.dtype == np.dtype(dtype)
            for field in b.fields:
                np.testing.assert_almost_equal(b.fields[field][...], a.fields[field][...])

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(proxy_urlpath)


def test_open_readonly_proxy_keeps_cache_and_source_readonly(tmp_path):
    source_path = tmp_path / "source.b2nd"
    proxy_path = tmp_path / "proxy.b2nd"
    data = np.arange(120, dtype=np.int32).reshape(12, 10)

    source = blosc2.asarray(data, chunks=(4, 5), blocks=(2, 5), urlpath=source_path, mode="w")
    proxy = blosc2.Proxy(source, urlpath=proxy_path, mode="w")
    proxy.fetch()
    cached_size = proxy_path.stat().st_size
    del proxy, source

    readonly = blosc2.open(proxy_path)

    assert readonly.schunk.mode == "r"
    assert readonly.schunk.vlmeta.mode == "r"
    assert readonly.src.schunk.mode == "r"
    np.testing.assert_array_equal(readonly[:], data)
    assert proxy_path.stat().st_size == cached_size

    with blosc2.open(proxy_path) as readonly_ctx:
        assert isinstance(readonly_ctx, blosc2.Proxy)
        np.testing.assert_array_equal(readonly_ctx[:], data)


# Test the ProxyNDSources interface
@pytest.mark.parametrize(
    ("shape", "chunks", "blocks"),
    [
        # One should be careful to choose aligned partitions for our source
        # E.g., the following is not aligned
        # ((10, 8), (4, 4), (2, 2))
        ((12,), (4,), (2,)),
        ((10, 8), (2, 8), (1, 4)),
        ((10, 8, 6), (2, 4, 3), (1, 2, 3)),
        ((4, 8, 6, 4), (2, 4, 3, 2), (1, 2, 3, 2)),
    ],
)
def test_proxy_source(shape, chunks, blocks):
    # Define an object that will be used as a source
    class Source(blosc2.ProxyNDSource):
        """
        A simple source that will be used to test the ProxyNDSource interface.

        """

        def __init__(self, data, chunks, blocks):
            self._data = data
            self._shape = data.shape
            self._dtype = data.dtype
            self._chunks = chunks
            self._chunksize = np.prod(self._chunks)
            self._blocks = blocks
            self._blocksize = np.prod(self._blocks) * self._dtype.itemsize
            self._chunks_idx, self._nchunks = get_chunks_idx(self._shape, self._chunks)
            aligned = blosc2.are_partitions_aligned(self._shape, self._chunks, self._blocks)
            if not aligned:
                raise ValueError("The partitions are not aligned")

        @property
        def shape(self) -> tuple:
            return self._shape

        @property
        def dtype(self):
            return self._dtype

        @property
        def chunks(self) -> tuple:
            return self._chunks

        @property
        def blocks(self) -> tuple:
            return self._blocks

        def get_chunk(self, nchunk):
            # Yep, this seems complex, but is one of the simplest possible implementations
            coords = tuple(np.unravel_index(nchunk, self._chunks_idx))
            slice_ = tuple(
                slice(c * s, min((c + 1) * s, self._shape[i]))
                for i, (c, s) in enumerate(zip(coords, self._chunks, strict=True))
            )
            data = self._data[slice_].tobytes()
            # Compress the data
            return blosc2.compress2(data, typesize=self._dtype.itemsize, blocksize=self._blocksize)

    data = np.arange(np.prod(shape), dtype="int32").reshape(shape)
    source = Source(data, chunks, blocks)
    proxy = blosc2.Proxy(source)
    result = proxy[...]
    np.testing.assert_array_equal(result, data)


def test_proxy_zeroshape():
    a1 = blosc2.ones(shape=(0, 100), chunks=(0, 9), blocks=(0, 1))
    na1 = a1[()]
    a1 = blosc2.Proxy(a1)
    sl = slice(100)
    np.testing.assert_allclose(a1[sl], na1[sl])


class _ConcurrencyTrackingSource:
    """A ProxyNDSource whose aget_chunk reports how many calls overlapped."""

    def __init__(self, array):
        self.array = array
        self.inflight = 0
        self.max_inflight = 0

    @property
    def shape(self):
        return self.array.shape

    @property
    def chunks(self):
        return self.array.chunks

    @property
    def blocks(self):
        return self.array.blocks

    @property
    def dtype(self):
        return self.array.dtype

    @property
    def cparams(self):
        return self.array.cparams

    def get_chunk(self, nchunk):
        return self.array.get_chunk(nchunk)

    async def aget_chunk(self, nchunk):
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        await asyncio.sleep(0.01)  # simulate a network round trip
        self.inflight -= 1
        return self.array.get_chunk(nchunk)


def test_proxy_afetch_max_concurrency():
    data = np.arange(10 * 10).reshape(10, 10)
    array = blosc2.asarray(data, chunks=(2, 10), blocks=(1, 10))  # 5 chunks
    source = _ConcurrencyTrackingSource(array)
    proxy = blosc2.Proxy(source)

    result = asyncio.run(proxy.afetch(max_concurrency=3))
    np.testing.assert_array_equal(result[:], data)
    assert source.max_inflight == 3  # bounded by the semaphore, not by the 5 chunks

    # Serial by default for non-remote sources
    source2 = _ConcurrencyTrackingSource(array)
    proxy2 = blosc2.Proxy(source2)
    asyncio.run(proxy2.afetch())
    assert source2.max_inflight == 1


def test_proxy_contiguous_kwarg(tmp_path):
    # Extra kwargs (e.g. contiguous) must be forwarded to the cache
    # container constructor, without needing the _cache= escape hatch.
    data = np.arange(20).reshape(4, 5)
    src = blosc2.asarray(data, chunks=[2, 5], blocks=[1, 5])
    urlpath = tmp_path / "cache.b2nd"
    proxy = blosc2.Proxy(src, urlpath=str(urlpath), contiguous=False)
    assert proxy.schunk.contiguous is False
    assert urlpath.is_dir()  # sparse frame is a directory of chunk files
    np.testing.assert_array_equal(proxy.fetch(())[:], data)
