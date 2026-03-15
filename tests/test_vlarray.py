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
    assert vlarray.insert(0, "head") == len(expected) + 1
    expected.insert(0, "head")
    assert vlarray.insert(-1, {"between": 5}) == len(expected) + 1
    expected.insert(-1, {"between": 5})
    assert vlarray.insert(999, "tail") == len(expected) + 1
    expected.insert(999, "tail")
    assert vlarray.delete(2) == len(expected) - 1
    del expected[2]
    del vlarray[-2]
    del expected[-2]
    assert list(vlarray) == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert isinstance(reopened, blosc2.VLArray)
        assert list(reopened) == expected
        with pytest.raises(ValueError):
            reopened.append("nope")
        with pytest.raises(ValueError):
            reopened[0] = "nope"
        with pytest.raises(ValueError):
            reopened.insert(0, "nope")
        with pytest.raises(ValueError):
            reopened.delete(0)
        with pytest.raises(ValueError):
            del reopened[0]
        with pytest.raises(ValueError):
            reopened.extend(["nope"])
        with pytest.raises(ValueError):
            reopened.pop()
        with pytest.raises(ValueError):
            reopened.clear()

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
    vlarray.extend(VALUES)
    vlarray.insert(1, {"inserted": True})
    del vlarray[3]
    expected = list(VALUES)
    expected.insert(1, {"inserted": True})
    del expected[3]

    restored = blosc2.from_cframe(vlarray.to_cframe())
    assert isinstance(restored, blosc2.VLArray)
    assert list(restored) == expected

    restored2 = blosc2.vlarray_from_cframe(vlarray.to_cframe())
    assert isinstance(restored2, blosc2.VLArray)
    assert list(restored2) == expected


def test_vlarray_constructor_kwargs():
    urlpath = "test_vlarray_kwargs.b2frame"
    blosc2.remove_urlpath(urlpath)

    vlarray = blosc2.VLArray(urlpath=urlpath, mode="w", contiguous=True)
    for value in VALUES:
        vlarray.append(value)

    reopened = blosc2.VLArray(urlpath=urlpath, mode="r", contiguous=True, mmap_mode="r")
    assert list(reopened) == VALUES

    blosc2.remove_urlpath(urlpath)


def test_vlarray_size_guard(monkeypatch):
    vlarray = blosc2.VLArray()
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 4)
    with pytest.raises(ValueError, match="Serialized objects cannot be larger"):
        vlarray.append("payload")


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_vlarray_list_ops.b2frame"),
        (False, "test_vlarray_list_ops_s.b2frame"),
    ],
)
def test_vlarray_list_like_ops(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    vlarray = blosc2.VLArray(storage=_storage(contiguous, urlpath))
    vlarray.extend([1, 2, 3])
    assert list(vlarray) == [1, 2, 3]
    assert vlarray.pop() == 3
    assert vlarray.pop(0) == 1
    assert list(vlarray) == [2]

    vlarray.clear()
    assert len(vlarray) == 0
    assert list(vlarray) == []

    vlarray.extend(["a", "b"])
    assert list(vlarray) == ["a", "b"]

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert list(reopened) == ["a", "b"]

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_vlarray_slices.b2frame"),
        (False, "test_vlarray_slices_s.b2frame"),
    ],
)
def test_vlarray_slices(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    expected = list(range(8))
    vlarray = blosc2.VLArray(storage=_storage(contiguous, urlpath))
    vlarray.extend(expected)

    assert vlarray[1:6:2] == expected[1:6:2]
    assert vlarray[::-2] == expected[::-2]

    vlarray[2:5] = ["a", "b"]
    expected[2:5] = ["a", "b"]
    assert list(vlarray) == expected

    vlarray[1:6:2] = [100, 101, 102]
    expected[1:6:2] = [100, 101, 102]
    assert list(vlarray) == expected

    del vlarray[::3]
    del expected[::3]
    assert list(vlarray) == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert reopened[::2] == expected[::2]
        with pytest.raises(ValueError):
            reopened[1:3] = [9]
        with pytest.raises(ValueError):
            del reopened[::2]

    blosc2.remove_urlpath(urlpath)


def test_vlarray_slice_errors():
    vlarray = blosc2.VLArray()
    vlarray.extend([0, 1, 2, 3])

    with pytest.raises(ValueError, match="extended slice"):
        vlarray[::2] = [9]
    with pytest.raises(TypeError):
        vlarray[1:2] = 3
    with pytest.raises(ValueError):
        _ = vlarray[::0]


def test_vlarray_copy():
    urlpath = "test_vlarray_copy.b2frame"
    copy_path = "test_vlarray_copy_out.b2frame"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.VLArray(urlpath=urlpath, mode="w", contiguous=True)
    original.extend(VALUES)
    original.insert(1, {"copy": True})

    copied = original.copy(
        urlpath=copy_path, contiguous=False, cparams={"codec": blosc2.Codec.LZ4, "clevel": 5}
    )
    assert list(copied) == list(original)
    assert copied.urlpath == copy_path
    assert copied.schunk.contiguous is False
    assert copied.cparams.codec == blosc2.Codec.LZ4
    assert copied.cparams.clevel == 5

    inmem = original.copy()
    assert list(inmem) == list(original)
    assert inmem.urlpath is None

    with pytest.raises(ValueError, match="meta should not be passed to copy"):
        original.copy(meta={})

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)


def test_vlarray_empty_list_roundtrip():
    values = [[], {"a": []}, [[], ["nested"]], None, ("tuple", []), {"rows": [[], []]}]
    vlarray = blosc2.VLArray()
    vlarray.extend(values)
    assert list(vlarray) == values


def test_vlarray_empty_tuple_roundtrip():
    values = [(), {"a": ()}, [(), ("nested",)], None, ("tuple", ()), {"rows": [[], ()]}]
    vlarray = blosc2.VLArray()
    vlarray.extend(values)
    assert list(vlarray) == values


def test_vlarray_insert_delete_errors():
    vlarray = blosc2.VLArray()
    vlarray.append("value")

    with pytest.raises(TypeError):
        vlarray.insert("0", "bad")
    with pytest.raises(IndexError):
        vlarray.delete(3)
    with pytest.raises(IndexError):
        blosc2.VLArray().pop()
    with pytest.raises(NotImplementedError):
        vlarray.pop(slice(0, 1))
