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
    "contiguous, urlpath, chunksize, nchunks, start, stop",
    [
        (True, None, 40_000, 10, 13, 59),
        (True, "b2frame", 20_000, 5, 0, 20_000 // 4 * 5),
        (False, None, 20_000, 20, 200, 20_000 // 4 + 349),
        (False, "b2frame", 40_000, 15, 40_000 // 4, 40_000 // 4 * 2),
    ],
)
def test_schunk_cache(contiguous, urlpath, chunksize, nchunks, start, stop):
    storage = {"contiguous": contiguous, "cparams": {'typesize': 4}}
    blosc2.remove_urlpath(urlpath)
    num_elem = chunksize // 4 * nchunks
    data = np.arange(num_elem, dtype="int32")
    schunk = blosc2.SChunk(chunksize=chunksize, data=data, **storage)
    bytes_obj = data.tobytes()
    cache = blosc2.ProxySChunk(schunk, urlpath=urlpath)

    cache_slice = cache[slice(start, stop)]
    assert cache_slice == bytes_obj[start * data.dtype.itemsize:stop * data.dtype.itemsize]

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
