#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import os
import sys

import numpy as np
import pytest

import blosc2


@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("urlpath", [None, "b2frame"])
@pytest.mark.parametrize(
    ("cparams", "dparams", "nchunks"),
    [
        ({"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 4}, {}, 10),
    ],
)
def test_schunk_numpy(contiguous, urlpath, cparams, dparams, nchunks):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}
    blosc2.remove_urlpath(urlpath)

    schunk = blosc2.SChunk(chunksize=200 * 1000 * 4, **kwargs)
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
    ("nbytes", "cparams", "dparams", "nchunks"),
    [
        (136, {"codec": blosc2.Codec.LZ4, "clevel": 6, "typesize": 1}, {}, 10),
    ],
)
def test_schunk(contiguous, urlpath, nbytes, cparams, dparams, nchunks):
    kwargs = {"contiguous": contiguous, "urlpath": urlpath, "cparams": cparams, "dparams": dparams}

    blosc2.remove_urlpath(urlpath)
    schunk = blosc2.SChunk(chunksize=2 * nbytes, **kwargs)
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


def test_vlmeta_supports_ref_roundtrip(tmp_path):
    schunk = blosc2.SChunk()
    array = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=str(tmp_path / "ref_array.b2nd"), mode="w")
    ref = blosc2.Ref.from_object(array)

    schunk.vlmeta["ref"] = ref

    restored = schunk.vlmeta["ref"]
    assert isinstance(restored, blosc2.Ref)
    assert restored == ref
    np.testing.assert_array_equal(restored.open()[:], array[:])

    bulk = schunk.vlmeta[:]
    assert isinstance(bulk["ref"], blosc2.Ref)
    assert bulk["ref"] == ref
    np.testing.assert_array_equal(bulk["ref"].open()[:], array[:])

    with pytest.raises(NotImplementedError, match="Slicing is not supported, unless"):
        _ = schunk.vlmeta[::2]
    with pytest.raises(NotImplementedError, match="Slicing is not supported, unless"):
        _ = schunk.meta[::2]


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


def test_vlmeta_empty_list_roundtrip():
    schunk = blosc2.SChunk()
    schunk.vlmeta["empty"] = []
    schunk.vlmeta["nested"] = {"rows": [[], ["x"]]}

    assert schunk.vlmeta["empty"] == []
    assert schunk.vlmeta["nested"] == {"rows": [[], ["x"]]}


def _rss_bytes():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def _rss_growth(op, n):
    for _ in range(10):
        op()
    rss = _rss_bytes()
    for _ in range(n):
        op()
    return _rss_bytes() - rss


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the RSS from /proc")
def test_vlmeta_read_frees_content():
    # Each read used to leak the 1 MiB content (200 MiB here)
    schunk = blosc2.SChunk()
    schunk.vlmeta["big"] = bytes(2**20)
    assert _rss_growth(lambda: schunk.vlmeta["big"], 200) < 50 * 2**20


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads the RSS from /proc")
def test_vlmeta_names_are_freed():
    # Each listing used to leak an array of 4096 pointers (32 MiB here)
    schunk = blosc2.SChunk()
    for i in range(4096):
        schunk.vlmeta[str(i)] = i
    assert _rss_growth(lambda: list(schunk.vlmeta), 1000) < 8 * 2**20
