#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# This source code is licensed under a BSD-style license (found in the
# LICENSE file in the root directory of this source tree)
#######################################################################

import numpy as np
import pytest

import blosc2
from blosc2.ndarray import get_chunks_idx

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
            _ = blosc2.open(proxy_urlpath)
    else:
        b = blosc2.open(proxy_urlpath)
        a = blosc2.open(urlpath)
        if not struct_dtype:
            np.testing.assert_almost_equal(b[...], a[...])
        else:
            assert b.dtype == np.dtype(dtype)
            for field in b.fields:
                np.testing.assert_almost_equal(b.fields[field][...], a.fields[field][...])

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(proxy_urlpath)


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
