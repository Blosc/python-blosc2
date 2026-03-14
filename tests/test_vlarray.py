#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest

import blosc2

VALUES = [
    b"bytes\x00payload",
    "plain text",
    42,
    3.5,
    True,
    None,
    [1, "two", b"three"],
    (1, 2, "three"),
    {"nested": [1, 2], "tuple": (3, 4)},
]


def _storage(contiguous, urlpath, mode="w"):
    return blosc2.Storage(contiguous=contiguous, urlpath=urlpath, mode=mode)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_vlarray.b2frame"),
        (False, "test_vlarray_s.b2frame"),
    ],
)
def test_vlarray_roundtrip(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    vlarray = blosc2.VLArray(storage=_storage(contiguous, urlpath))
    assert vlarray.meta["vlarray"]["serializer"] == "msgpack"

    for i, value in enumerate(VALUES, start=1):
        assert vlarray.append(value) == i

    assert len(vlarray) == len(VALUES)
    assert list(vlarray) == VALUES
    assert vlarray[-1] == VALUES[-1]

    expected = list(VALUES)
    expected[1] = {"updated": ("tuple", 7)}
    expected[-1] = "tiny"
    vlarray[1] = expected[1]
    vlarray[-1] = expected[-1]
    assert list(vlarray) == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert isinstance(reopened, blosc2.VLArray)
        assert list(reopened) == expected
        with pytest.raises(ValueError):
            reopened.append("nope")
        with pytest.raises(ValueError):
            reopened[0] = "nope"

        reopened_rw = blosc2.open(urlpath, mode="a")
        reopened_rw[0] = "changed"
        expected[0] = "changed"
        assert list(reopened_rw) == expected

        if contiguous:
            reopened_mmap = blosc2.open(urlpath, mode="r", mmap_mode="r")
            assert isinstance(reopened_mmap, blosc2.VLArray)
            assert list(reopened_mmap) == expected

    blosc2.remove_urlpath(urlpath)


def test_vlarray_from_cframe():
    vlarray = blosc2.VLArray()
    for value in VALUES[:4]:
        vlarray.append(value)

    restored = blosc2.from_cframe(vlarray.to_cframe())
    assert isinstance(restored, blosc2.VLArray)
    assert list(restored) == VALUES[:4]

    restored2 = blosc2.vlarray_from_cframe(vlarray.to_cframe())
    assert isinstance(restored2, blosc2.VLArray)
    assert list(restored2) == VALUES[:4]


def test_vlarray_constructor_kwargs():
    urlpath = "test_vlarray_kwargs.b2frame"
    blosc2.remove_urlpath(urlpath)

    vlarray = blosc2.VLArray(urlpath=urlpath, mode="w", contiguous=True)
    for value in VALUES[:3]:
        vlarray.append(value)

    reopened = blosc2.VLArray(urlpath=urlpath, mode="r", contiguous=True, mmap_mode="r")
    assert list(reopened) == VALUES[:3]

    blosc2.remove_urlpath(urlpath)


def test_vlarray_size_guard(monkeypatch):
    vlarray = blosc2.VLArray()
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 4)
    with pytest.raises(ValueError, match="Serialized objects cannot be larger"):
        vlarray.append("payload")
