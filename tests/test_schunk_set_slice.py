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


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize("mode", ["w", "a"])
@pytest.mark.parametrize(
    "cparams, dparams, nchunks, start, stop",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 1, 200 * 100 * 1, 200 * 100 * 2),
        ({"typesize": 4}, {"nthreads": 4}, 1, 200 * 100 * 1 - 233, 200 * 100 * 3 + 7),
        (
            {"splitmode": blosc2.SplitMode.ALWAYS_SPLIT, "nthreads": 5, "typesize": 4},
            {},
            5,
            21,
            200 * 2 * 100,
        ),
        ({"codec": blosc2.Codec.LZ4HC, "typesize": 4}, {}, 7, None, None),
        ({"typesize": 4, "blocksize": 200 * 100}, {}, 7, 3, -12),
        ({"blocksize": 200 * 100, "typesize": 4}, {}, 5, -2456, -234),
        ({"blocksize": 200 * 100 + 4 * 2, "typesize": 4}, {}, 2, -1, 200 * 100 * 3 + 7),
    ],
)
def test_schunk_set_slice(contiguous, urlpath, mode, cparams, dparams, nchunks, start, stop):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    data = np.arange(200 * 100 * nchunks, dtype="int32")
    schunk = blosc2.SChunk(chunksize=200 * 100 * 4, data=data, mode=mode, **storage)

    _start, _stop = start, stop
    if _start is None:
        _start = 0
    elif _start < 0:
        _start += data.size
    if _stop is None:
        _stop = data.size
    elif _stop < 0:
        _stop += data.size

    val = nchunks * np.arange(_stop - _start, dtype="int32")
    schunk[start:stop] = val

    out = np.empty(val.shape, dtype="int32")

    schunk.get_slice(_start, _stop, out)
    assert np.array_equal(val, out)

    blosc2.remove_urlpath(urlpath)


def test_schunk_set_slice_raises():
    storage = {"contiguous": True, "urlpath": "schunk.b2frame", "cparams": {"typesize": 4}, "dparams": {}}
    blosc2.remove_urlpath(storage["urlpath"])

    nchunks = 2
    data = np.arange(200 * 100 * nchunks, dtype="int32")
    blosc2.SChunk(chunksize=200 * 100 * 4, data=data, **storage)

    schunk = blosc2.open(storage["urlpath"], mode="r")
    start = 200 * 100
    stop = 200 * 100 * nchunks
    val = 3 * np.arange(start, stop, dtype="int32")

    with pytest.raises(ValueError):
        schunk[start:stop] = val

    schunk = blosc2.open(storage["urlpath"], mode="a")
    with pytest.raises(IndexError):
        schunk[start:stop:2] = val

    stop += 4
    with pytest.raises(ValueError):
        schunk[start:stop] = val

    start = -1
    stop = -4
    with pytest.raises(ValueError):
        schunk[start:stop] = val

    start = 200 * 100 * 2 + 1
    stop = 200 * 100 * 2 * 3
    with pytest.raises(ValueError):
        schunk[start:stop] = val

    blosc2.remove_urlpath(storage["urlpath"])
