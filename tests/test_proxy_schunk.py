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


@pytest.mark.parametrize(
    ("contiguous", "urlpath", "chunksize", "nchunks", "start", "stop"),
    [
        (True, None, 40_000, 10, 13, 59),
        (True, "b2frame", 20_000, 5, 0, 20_000 // 4 * 5),
        (False, None, 20_000, 20, 200, 20_000 // 4 + 349),
        (False, "b2frame", 40_000, 15, 40_000 // 4, 40_000 // 4 * 2),
    ],
)
def test_schunk_proxy(contiguous, urlpath, chunksize, nchunks, start, stop):
    kwargs = {"contiguous": contiguous, "cparams": {"typesize": 4}}
    num_elem = chunksize // 4 * nchunks
    data = np.arange(num_elem, dtype="int32")
    schunk = blosc2.SChunk(chunksize=chunksize, data=data, **kwargs)
    bytes_obj = data.tobytes()
    cache = blosc2.Proxy(schunk, urlpath=urlpath, mode="w")

    cache_slice = cache[slice(start, stop)]
    assert cache_slice == bytes_obj[start * data.dtype.itemsize : stop * data.dtype.itemsize]

    cache_slice = cache.fetch(slice(start, stop))
    assert cache_slice.urlpath == urlpath
    out = np.empty(stop - start, data.dtype)
    cache_slice.get_slice(start, stop, out)
    assert np.array_equal(out, data[start:stop])

    cache_eval = cache.fetch()
    assert cache_eval.urlpath == urlpath
    out = np.empty(data.shape, data.dtype)
    cache_eval.get_slice(0, None, out)
    assert np.array_equal(out, data)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("urlpath", "chunksize", "nchunks"),
    [
        (None, 40_000, 10),
        ("b2frame", 20_000, 5),
        (None, 20_000, 20),
        ("b2frame", 40_000, 15),
    ],
)
def test_open(urlpath, chunksize, nchunks):
    kwargs = {"urlpath": urlpath, "cparams": {"typesize": 4}}
    proxy_urlpath = "proxy.b2frame"
    blosc2.remove_urlpath(urlpath)
    num_elem = chunksize // 4 * nchunks
    data = np.arange(num_elem, dtype="int32")
    schunk = blosc2.SChunk(chunksize=chunksize, data=data, **kwargs)
    bytes_obj = data.tobytes()
    proxy = blosc2.Proxy(schunk, urlpath=proxy_urlpath, mode="w")
    del proxy
    del schunk
    if urlpath is None:
        with pytest.raises(RuntimeError):
            _ = blosc2.open(proxy_urlpath)
    else:
        proxy = blosc2.open(proxy_urlpath)
        assert proxy[0 : len(data) * 4] == bytes_obj

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(proxy_urlpath)


# Test the ProxySource class
def test_proxy_source():
    # Define an object that will be used as a source
    class Source(blosc2.ProxySource):
        def __init__(self, data):
            self._data = data
            self._nbytes = len(data) * 4
            self._typesize = 4
            self._chunksize = 20

        @property
        def nbytes(self) -> int:
            return self._nbytes

        @property
        def chunksize(self) -> int:
            return self._chunksize

        @property
        def typesize(self) -> int:
            return self._typesize

        def get_chunk(self, nchunk):
            data = self._data[nchunk * self.chunksize : (nchunk + 1) * self.chunksize]
            # Compress the data
            return blosc2.compress2(data, typesize=self._typesize)

    data = np.arange(100, dtype="int32").tobytes()
    source = Source(data)
    proxy = blosc2.Proxy(source)
    assert proxy[0:100] == data
