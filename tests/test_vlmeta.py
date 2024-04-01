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
@pytest.mark.parametrize(
    "cparams, dparams, nchunks",
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 10),
    ],
)
def test_schunk_numpy(contiguous, urlpath, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **storage)
    for i in range(nchunks):
        buffer = i * np.arange(200 * 1000, dtype="int32")
        nchunks_ = schunk.append_data(buffer)
        assert nchunks_ == (i + 1)

    add(schunk)
    iter(schunk)
    delete(schunk)
    clear(schunk)

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    "nbytes, cparams, dparams, nchunks",
    [
        (136, {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 1}, {}, 10),
    ],
)
def test_schunk(contiguous, urlpath, nbytes, cparams, dparams, nchunks):
    storage = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

    blosc2.remove_urlpath(urlpath)
    schunk = blosc2.SChunk(chunksize=2 * nbytes, **storage)
    for i in range(nchunks):
        bytes_obj = b"i " * nbytes
        nchunks_ = schunk.append_data(bytes_obj)
        assert nchunks_ == (i + 1)

    add(schunk)
    to_dict(schunk)
    iter(schunk)
    delete(schunk)
    clear(schunk)

    blosc2.remove_urlpath(urlpath)


def add(schunk):
    schunk.vlmeta["vlmeta1"] = b"val1"
    schunk.vlmeta["vlmeta2"] = "val2"
    schunk.vlmeta["vlmeta3"] = {b"lorem": 4231}
    schunk.vlmeta["vlmeta4"] = [1, 2, 3]
    schunk.vlmeta["vlmeta5"] = (1, 2, 3)

    assert schunk.vlmeta["vlmeta1"] == b"val1"
    assert schunk.vlmeta["vlmeta2"] == "val2"
    assert schunk.vlmeta["vlmeta3"] == {b"lorem": 4231}
    assert schunk.vlmeta["vlmeta4"] == [1, 2, 3]
    assert schunk.vlmeta["vlmeta5"] == (1, 2, 3)
    assert "vlmeta1" in schunk.vlmeta
    assert len(schunk.vlmeta) == 5


def to_dict(schunk):
    assert schunk.vlmeta.to_dict() == {
        b"vlmeta1": b"val1",
        b"vlmeta2": "val2",
        b"vlmeta3": {b"lorem": 4231},
        b"vlmeta4": [1, 2, 3],
        b"vlmeta5": (1, 2, 3),
    }


def delete(schunk):
    # Remove one of them
    assert "vlmeta2" in schunk.vlmeta
    del schunk.vlmeta["vlmeta2"]
    assert "vlmeta2" not in schunk.vlmeta
    assert schunk.vlmeta["vlmeta1"] == b"val1"
    assert schunk.vlmeta["vlmeta3"] == {b"lorem": 4231}
    assert schunk.vlmeta["vlmeta4"] == [1, 2, 3]
    assert schunk.vlmeta["vlmeta5"] == (1, 2, 3)
    with pytest.raises(KeyError):
        schunk.vlmeta["vlmeta2"]
    assert len(schunk.vlmeta) == 4


def iter(schunk):
    keys = ["vlmeta1", "vlmeta2", "vlmeta3", "vlmeta4", "vlmeta5"]
    for i, vlmeta in enumerate(schunk.vlmeta):
        assert vlmeta == keys[i]


def clear(schunk):
    nparray = np.arange(start=0, stop=2)
    schunk.vlmeta["vlmeta2"] = nparray.tobytes()
    assert schunk.vlmeta["vlmeta2"] == nparray.tobytes()
    assert schunk.vlmeta.__len__() == 5

    schunk.vlmeta.clear()
    assert schunk.vlmeta.__len__() == 0
