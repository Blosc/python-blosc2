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

    nested_oarr = blosc2.ObjectArray()
    nested_oarr.extend(["alpha", {"beta": 2}])

    nested_batcharray = blosc2.BatchArray(items_per_block=2)
    nested_batcharray.extend([[1, 2], ["x", {"y": 3}]])

    estore = blosc2.EmbedStore()
    estore["/node"] = blosc2.arange(3, dtype=np.int32)

    return ndarray, schunk, nested_oarr, nested_batcharray, estore


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


def _make_persistent_ref(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=tmp_path / "a_ref.b2nd", mode="w")
    return blosc2.Ref.from_object(a), a[:]


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_objectarray.b2frame"),
        (False, "test_objectarray_s.b2frame"),
    ],
)
def test_objectarray_roundtrip(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    oarr = blosc2.ObjectArray(storage=_storage(contiguous, urlpath))
    assert oarr.meta["vlarray"]["serializer"] == "msgpack"

    for i, value in enumerate(VALUES, start=1):
        assert oarr.append(value) == i

    assert len(oarr) == len(VALUES)
    assert list(oarr) == VALUES
    assert oarr[-1] == VALUES[-1]

    expected = list(VALUES)
    expected[1] = {"updated": ("tuple", 7)}
    expected[-1] = "tiny"
    oarr[1] = expected[1]
    oarr[-1] = expected[-1]
    assert oarr.insert(0, "head") == len(expected) + 1
    expected.insert(0, "head")
    assert oarr.insert(-1, {"between": 5}) == len(expected) + 1
    expected.insert(-1, {"between": 5})
    assert oarr.insert(999, "tail") == len(expected) + 1
    expected.insert(999, "tail")
    assert oarr.delete(2) == len(expected) - 1
    del expected[2]
    del oarr[-2]
    del expected[-2]
    assert list(oarr) == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert isinstance(reopened, blosc2.ObjectArray)
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
            assert isinstance(reopened_mmap, blosc2.ObjectArray)
            assert list(reopened_mmap) == expected

    blosc2.remove_urlpath(urlpath)


def test_objectarray_from_cframe():
    oarr = blosc2.ObjectArray()
    oarr.extend(VALUES)
    oarr.insert(1, {"inserted": True})
    del oarr[3]
    expected = list(VALUES)
    expected.insert(1, {"inserted": True})
    del expected[3]

    restored = blosc2.from_cframe(oarr.to_cframe())
    assert isinstance(restored, blosc2.ObjectArray)
    assert list(restored) == expected

    restored2 = blosc2.objectarray_from_cframe(oarr.to_cframe())
    assert isinstance(restored2, blosc2.ObjectArray)
    assert list(restored2) == expected


def test_objectarray_msgpack_supports_blosc2_objects():
    ndarray, schunk, nested_oarr, nested_batcharray, estore = _make_nested_blosc2_objects()

    oarr = blosc2.ObjectArray()
    oarr.append(
        {
            "ndarray": ndarray,
            "schunk": schunk,
            "objectarray": nested_oarr,
            "batcharray": nested_batcharray,
            "estore": estore,
        }
    )

    restored = oarr[0]

    assert isinstance(restored["ndarray"], blosc2.NDArray)
    assert np.array_equal(restored["ndarray"][:], ndarray[:])

    assert isinstance(restored["schunk"], blosc2.SChunk)
    assert restored["schunk"].decompress_chunk(0) == schunk.decompress_chunk(0)

    assert isinstance(restored["objectarray"], blosc2.ObjectArray)
    assert list(restored["objectarray"]) == list(nested_oarr)

    assert isinstance(restored["batcharray"], blosc2.BatchArray)
    assert [batch[:] for batch in restored["batcharray"]] == [batch[:] for batch in nested_batcharray]

    assert isinstance(restored["estore"], blosc2.EmbedStore)
    assert list(restored["estore"].keys()) == ["/node"]
    assert np.array_equal(restored["estore"]["/node"][:], estore["/node"][:])


def test_objectarray_msgpack_supports_c2array(monkeypatch):
    c2array = _make_c2array(monkeypatch)

    oarr = blosc2.ObjectArray()
    oarr.append(c2array)

    restored = oarr[0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == c2array.path
    assert restored.urlbase == c2array.urlbase
    assert restored.auth_token is None


def test_objectarray_msgpack_supports_ref(tmp_path):
    ref, expected = _make_persistent_ref(tmp_path)

    oarr = blosc2.ObjectArray()
    oarr.append(ref)

    restored = oarr[0]

    assert isinstance(restored, blosc2.Ref)
    assert restored == ref
    np.testing.assert_array_equal(restored.open()[:], expected)


def test_objectarray_msgpack_supports_lazyexpr(tmp_path):
    expr, expected = _make_persistent_lazyexpr(tmp_path)

    oarr = blosc2.ObjectArray()
    oarr.append(expr)

    restored = oarr[0]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


def test_objectarray_msgpack_supports_lazyudf_dslkernel(tmp_path):
    udf, expected = _make_persistent_lazyudf(tmp_path)

    oarr = blosc2.ObjectArray()
    oarr.append(udf)
    restored = oarr[0]

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], expected)


def test_objectarray_msgpack_rejects_lazyexpr_with_in_memory_operands():
    expr = _make_in_memory_lazyexpr()

    oarr = blosc2.ObjectArray()
    with pytest.raises(ValueError, match="stored on disk/network"):
        oarr.append(expr)


def test_objectarray_msgpack_rejects_plain_python_lazyudf(tmp_path):
    udf = _make_persistent_python_lazyudf(tmp_path)

    oarr = blosc2.ObjectArray()
    with pytest.raises(TypeError, match="DSLKernel"):
        oarr.append(udf)


@pytest.mark.network
def test_objectarray_msgpack_roundtrip_c2array_network(cat2_context):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    original = blosc2.C2Array(path)

    oarr = blosc2.ObjectArray()
    oarr.append(original)

    restored = oarr[0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    np.testing.assert_allclose(restored[:], original[:])


def test_objectarray_info():
    oarr = blosc2.ObjectArray()
    oarr.extend(VALUES)

    assert oarr.typesize == 1
    assert oarr.contiguous == oarr.schunk.contiguous
    assert oarr.urlpath == oarr.schunk.urlpath

    items = dict(oarr.info_items)
    assert items["type"] == "ObjectArray"
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

    text = repr(oarr.info)
    assert "type" in text
    assert "ObjectArray" in text
    assert "item_nbytes_avg" in text


def test_objectarray_zstd_uses_dict_by_default():
    oarr = blosc2.ObjectArray()
    assert oarr.cparams.codec == blosc2.Codec.ZSTD
    assert oarr.cparams.use_dict is True


def test_objectarray_respects_explicit_use_dict_and_non_zstd():
    oarr = blosc2.ObjectArray(cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
    assert oarr.cparams.codec == blosc2.Codec.LZ4
    assert oarr.cparams.use_dict is False

    oarr = blosc2.ObjectArray(cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1, "use_dict": True})
    assert oarr.cparams.codec == blosc2.Codec.LZ4HC
    assert oarr.cparams.use_dict is True

    oarr = blosc2.ObjectArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 0})
    assert oarr.cparams.codec == blosc2.Codec.ZSTD
    assert oarr.cparams.use_dict is False

    oarr = blosc2.ObjectArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "use_dict": False})
    assert oarr.cparams.use_dict is False

    oarr = blosc2.ObjectArray(cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, use_dict=False))
    assert oarr.cparams.use_dict is False


def test_objectarray_constructor_kwargs():
    urlpath = "test_objectarray_kwargs.b2frame"
    blosc2.remove_urlpath(urlpath)

    oarr = blosc2.ObjectArray(urlpath=urlpath, mode="w", contiguous=True)
    for value in VALUES:
        oarr.append(value)

    reopened = blosc2.ObjectArray(urlpath=urlpath, mode="r", contiguous=True, mmap_mode="r")
    assert list(reopened) == VALUES

    blosc2.remove_urlpath(urlpath)


def test_objectarray_size_guard(monkeypatch):
    oarr = blosc2.ObjectArray()
    monkeypatch.setattr(blosc2, "MAX_BUFFERSIZE", 4)
    with pytest.raises(ValueError, match="Serialized objects cannot be larger"):
        oarr.append("payload")


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_objectarray_list_ops.b2frame"),
        (False, "test_objectarray_list_ops_s.b2frame"),
    ],
)
def test_objectarray_list_like_ops(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    oarr = blosc2.ObjectArray(storage=_storage(contiguous, urlpath))
    oarr.extend([1, 2, 3])
    assert list(oarr) == [1, 2, 3]
    assert oarr.pop() == 3
    assert oarr.pop(0) == 1
    assert list(oarr) == [2]

    oarr.clear()
    assert len(oarr) == 0
    assert list(oarr) == []

    oarr.extend(["a", "b"])
    assert list(oarr) == ["a", "b"]

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert list(reopened) == ["a", "b"]

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_objectarray_slices.b2frame"),
        (False, "test_objectarray_slices_s.b2frame"),
    ],
)
def test_objectarray_slices(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    expected = list(range(8))
    oarr = blosc2.ObjectArray(storage=_storage(contiguous, urlpath))
    oarr.extend(expected)

    assert oarr[1:6:2] == expected[1:6:2]
    assert oarr[::-2] == expected[::-2]

    oarr[2:5] = ["a", "b"]
    expected[2:5] = ["a", "b"]
    assert list(oarr) == expected

    oarr[1:6:2] = [100, 101, 102]
    expected[1:6:2] = [100, 101, 102]
    assert list(oarr) == expected

    del oarr[::3]
    del expected[::3]
    assert list(oarr) == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert reopened[::2] == expected[::2]
        with pytest.raises(ValueError):
            reopened[1:3] = [9]
        with pytest.raises(ValueError):
            del reopened[::2]

    blosc2.remove_urlpath(urlpath)


def test_objectarray_slice_errors():
    oarr = blosc2.ObjectArray()
    oarr.extend([0, 1, 2, 3])

    with pytest.raises(ValueError, match="extended slice"):
        oarr[::2] = [9]
    with pytest.raises(TypeError):
        oarr[1:2] = 3
    with pytest.raises(ValueError):
        _ = oarr[::0]


def test_objectarray_copy():
    urlpath = "test_objectarray_copy.b2frame"
    copy_path = "test_objectarray_copy_out.b2frame"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.ObjectArray(urlpath=urlpath, mode="w", contiguous=True)
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


def test_objectarray_empty_list_roundtrip():
    values = [[], {"a": []}, [[], ["nested"]], None, ("tuple", []), {"rows": [[], []]}]
    oarr = blosc2.ObjectArray()
    oarr.extend(values)
    assert list(oarr) == values


def test_objectarray_empty_tuple_roundtrip():
    values = [(), {"a": ()}, [(), ("nested",)], None, ("tuple", ()), {"rows": [[], ()]}]
    oarr = blosc2.ObjectArray()
    oarr.extend(values)
    assert list(oarr) == values


def test_objectarray_insert_delete_errors():
    oarr = blosc2.ObjectArray()
    oarr.append("value")

    with pytest.raises(TypeError):
        oarr.insert("0", "bad")
    with pytest.raises(IndexError):
        oarr.delete(3)
    with pytest.raises(IndexError):
        blosc2.ObjectArray().pop()
    with pytest.raises(NotImplementedError):
        oarr.pop(slice(0, 1))


def test_objectarray_delete_negative_step_slice():
    # Regression: negative-step slices used to delete in ascending order,
    # shifting indices and deleting the wrong chunks (or raising).
    oarr = blosc2.ObjectArray()
    oarr.extend(range(5))
    del oarr[3:0:-1]
    assert list(oarr) == [0, 4]

    oarr2 = blosc2.ObjectArray()
    oarr2.extend(range(5))
    del oarr2[::-1]
    assert len(oarr2) == 0
