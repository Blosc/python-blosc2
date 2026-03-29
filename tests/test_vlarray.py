#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import numpy as np
import pytest

import blosc2
import blosc2.c2array as blosc2_c2array


@blosc2.dsl_kernel
def _kernel_add_twice(x, y):
    return x + y * 2


def _python_udf_add(inputs_tuple, output, offset):
    x, y = inputs_tuple
    output[:] = x + y


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


def _make_nested_blosc2_objects():
    ndarray = blosc2.arange(6, dtype=np.int32)

    schunk = blosc2.SChunk(chunksize=16)
    schunk.append_data(np.arange(4, dtype=np.int32))

    nested_vlarray = blosc2.VLArray()
    nested_vlarray.extend(["alpha", {"beta": 2}])

    nested_batchstore = blosc2.BatchStore(items_per_block=2)
    nested_batchstore.extend([[1, 2], ["x", {"y": 3}]])

    estore = blosc2.EmbedStore()
    estore["/node"] = blosc2.arange(3, dtype=np.int32)

    return ndarray, schunk, nested_vlarray, nested_batchstore, estore


def _make_c2array(monkeypatch, path="@public/examples/ds-1d.b2nd", urlbase="https://cat2.cloud/demo/"):
    def fake_info(path_, urlbase_, params=None, headers=None, model=None, auth_token=None):
        return {"schunk": {"cparams": dict(blosc2.cparams_dflts)}}

    monkeypatch.setattr(blosc2_c2array, "info", fake_info)
    return blosc2.C2Array(path, urlbase=urlbase)


def _make_persistent_lazyexpr(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=tmp_path / "a.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})
    expected = np.arange(5, dtype=np.int64) * 3
    return expr, expected


def _make_in_memory_lazyexpr():
    a = blosc2.asarray(np.arange(5, dtype=np.int64))
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2)
    return blosc2.lazyexpr("a + b", operands={"a": a, "b": b})


def _make_persistent_lazyudf(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.float32), urlpath=tmp_path / "a_udf.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.float32) * 2, urlpath=tmp_path / "b_udf.b2nd", mode="w")
    udf = blosc2.lazyudf(_kernel_add_twice, (a, b), dtype=a.dtype, shape=a.shape)
    expected = a[:] + b[:] * 2
    return udf, expected


def _make_persistent_python_lazyudf(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.float32), urlpath=tmp_path / "a_pyudf.b2nd", mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.float32) * 2, urlpath=tmp_path / "b_pyudf.b2nd", mode="w")
    return blosc2.lazyudf(_python_udf_add, (a, b), dtype=a.dtype, shape=a.shape)


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


def test_vlarray_msgpack_supports_blosc2_objects():
    ndarray, schunk, nested_vlarray, nested_batchstore, estore = _make_nested_blosc2_objects()

    vlarray = blosc2.VLArray()
    vlarray.append(
        {
            "ndarray": ndarray,
            "schunk": schunk,
            "vlarray": nested_vlarray,
            "batchstore": nested_batchstore,
            "estore": estore,
        }
    )

    restored = vlarray[0]

    assert isinstance(restored["ndarray"], blosc2.NDArray)
    assert np.array_equal(restored["ndarray"][:], ndarray[:])

    assert isinstance(restored["schunk"], blosc2.SChunk)
    assert restored["schunk"].decompress_chunk(0) == schunk.decompress_chunk(0)

    assert isinstance(restored["vlarray"], blosc2.VLArray)
    assert list(restored["vlarray"]) == list(nested_vlarray)

    assert isinstance(restored["batchstore"], blosc2.BatchStore)
    assert [batch[:] for batch in restored["batchstore"]] == [batch[:] for batch in nested_batchstore]

    assert isinstance(restored["estore"], blosc2.EmbedStore)
    assert list(restored["estore"].keys()) == ["/node"]
    assert np.array_equal(restored["estore"]["/node"][:], estore["/node"][:])


def test_vlarray_msgpack_supports_c2array(monkeypatch):
    c2array = _make_c2array(monkeypatch)

    vlarray = blosc2.VLArray()
    vlarray.append(c2array)

    restored = vlarray[0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == c2array.path
    assert restored.urlbase == c2array.urlbase
    assert restored.auth_token is None


def test_vlarray_msgpack_supports_lazyexpr(tmp_path):
    expr, expected = _make_persistent_lazyexpr(tmp_path)

    vlarray = blosc2.VLArray()
    vlarray.append(expr)

    restored = vlarray[0]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


def test_vlarray_msgpack_supports_lazyudf_dslkernel(tmp_path):
    udf, expected = _make_persistent_lazyudf(tmp_path)

    vlarray = blosc2.VLArray()
    vlarray.append(udf)
    restored = vlarray[0]

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], expected)


def test_vlarray_msgpack_rejects_lazyexpr_with_in_memory_operands():
    expr = _make_in_memory_lazyexpr()

    vlarray = blosc2.VLArray()
    with pytest.raises(ValueError, match="stored on disk/network"):
        vlarray.append(expr)


def test_vlarray_msgpack_rejects_plain_python_lazyudf(tmp_path):
    udf = _make_persistent_python_lazyudf(tmp_path)

    vlarray = blosc2.VLArray()
    with pytest.raises(TypeError, match="DSLKernel"):
        vlarray.append(udf)


@pytest.mark.network
def test_vlarray_msgpack_roundtrip_c2array_network(cat2_context):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    original = blosc2.C2Array(path)

    vlarray = blosc2.VLArray()
    vlarray.append(original)

    restored = vlarray[0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    np.testing.assert_allclose(restored[:], original[:])


def test_vlarray_info():
    vlarray = blosc2.VLArray()
    vlarray.extend(VALUES)

    assert vlarray.typesize == 1
    assert vlarray.contiguous == vlarray.schunk.contiguous
    assert vlarray.urlpath == vlarray.schunk.urlpath

    items = dict(vlarray.info_items)
    assert items["type"] == "VLArray"
    assert items["entries"] == len(VALUES)
    assert items["item_nbytes_min"] > 0
    assert items["item_nbytes_max"] >= items["item_nbytes_min"]
    assert items["chunk_cbytes_min"] > 0
    assert items["chunk_cbytes_max"] >= items["chunk_cbytes_min"]
    assert "urlpath" not in items
    assert "contiguous" not in items
    assert "typesize" not in items
    assert "(" in items["nbytes"]
    assert "(" in items["cbytes"]

    text = repr(vlarray.info)
    assert "type" in text
    assert "VLArray" in text
    assert "item_nbytes_avg" in text


def test_vlarray_zstd_uses_dict_by_default():
    vlarray = blosc2.VLArray()
    assert vlarray.cparams.codec == blosc2.Codec.ZSTD
    assert vlarray.cparams.use_dict is True


def test_vlarray_respects_explicit_use_dict_and_non_zstd():
    vlarray = blosc2.VLArray(cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
    assert vlarray.cparams.codec == blosc2.Codec.LZ4
    assert vlarray.cparams.use_dict is False

    vlarray = blosc2.VLArray(cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1, "use_dict": True})
    assert vlarray.cparams.codec == blosc2.Codec.LZ4HC
    assert vlarray.cparams.use_dict is True

    vlarray = blosc2.VLArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 0})
    assert vlarray.cparams.codec == blosc2.Codec.ZSTD
    assert vlarray.cparams.use_dict is False

    vlarray = blosc2.VLArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "use_dict": False})
    assert vlarray.cparams.use_dict is False

    vlarray = blosc2.VLArray(cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, use_dict=False))
    assert vlarray.cparams.use_dict is False


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
