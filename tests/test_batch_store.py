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
from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb


@blosc2.dsl_kernel
def _kernel_add_twice(x, y):
    return x + y * 2


def _python_udf_add(inputs_tuple, output, offset):
    x, y = inputs_tuple
    output[:] = x + y


BATCHES = [
    [b"bytes\x00payload", "plain text", 42],
    [{"nested": [1, 2]}, None, {"tail": True}],
    [(1, 2, "three"), 3.5, True],
]


def _make_payload(seed, size):
    base = bytes((seed + i) % 251 for i in range(251))
    reps = size // len(base) + 1
    return (base * reps)[:size]


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


def _make_persistent_ref(tmp_path):
    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=tmp_path / "a_ref.b2nd", mode="w")
    return blosc2.Ref.from_object(a), a[:]


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batchstore.b2b"),
        (False, "test_batchstore_s.b2b"),
    ],
)
def test_batchstore_roundtrip(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(storage=_storage(contiguous, urlpath))
    assert barray.meta["batchstore"]["serializer"] == "msgpack"

    for i, batch in enumerate(BATCHES, start=1):
        assert barray.append(batch) == i

    assert len(barray) == len(BATCHES)
    assert barray.items_per_block is not None
    assert 1 <= barray.items_per_block <= len(BATCHES[0])
    assert [batch[:] for batch in barray] == BATCHES
    assert barray.append([1, 2]) == len(BATCHES) + 1
    assert [batch[:] for batch in barray][-1] == [1, 2]

    batch0 = barray[0]
    assert isinstance(batch0, blosc2.Batch)
    assert len(batch0) == len(BATCHES[0])
    assert batch0[1] == BATCHES[0][1]
    assert batch0[:] == BATCHES[0]
    assert isinstance(batch0.lazybatch, bytes)
    assert batch0.nbytes > 0
    assert batch0.cbytes > 0
    assert batch0.cratio > 0

    expected = list(BATCHES)
    expected.append([1, 2])
    expected[1] = ["updated", {"tuple": (7, 8)}, 99]
    expected[-1] = ["tiny", False, "x"]
    barray[1] = expected[1]
    barray[-1] = expected[-1]
    assert barray.insert(0, ["head", 0, "x"]) == len(expected) + 1
    expected.insert(0, ["head", 0, "x"])
    assert barray.insert(-1, ["between", {"k": 5}, None]) == len(expected) + 1
    expected.insert(-1, ["between", {"k": 5}, None])
    assert barray.insert(999, ["tail", 1, 2]) == len(expected) + 1
    expected.insert(999, ["tail", 1, 2])
    assert barray.delete(2) == len(expected) - 1
    del expected[2]
    del barray[-2]
    del expected[-2]
    assert [batch[:] for batch in barray] == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert isinstance(reopened, blosc2.BatchStore)
        assert reopened.items_per_block == barray.items_per_block
        assert [batch[:] for batch in reopened] == expected
        with pytest.raises(ValueError):
            reopened.append(["nope"])
        with pytest.raises(ValueError):
            reopened[0] = ["nope"]
        with pytest.raises(ValueError):
            reopened.insert(0, ["nope"])
        with pytest.raises(ValueError):
            reopened.delete(0)
        with pytest.raises(ValueError):
            del reopened[0]
        with pytest.raises(ValueError):
            reopened.extend([["nope"]])
        with pytest.raises(ValueError):
            reopened.pop()
        with pytest.raises(ValueError):
            reopened.clear()

        reopened_rw = blosc2.open(urlpath, mode="a")
        reopened_rw[0] = ["changed", "batch", 0]
        expected[0] = ["changed", "batch", 0]
        assert [batch[:] for batch in reopened_rw] == expected

        if contiguous:
            reopened_mmap = blosc2.open(urlpath, mode="r", mmap_mode="r")
            assert isinstance(reopened_mmap, blosc2.BatchStore)
            assert [batch[:] for batch in reopened_mmap] == expected

    blosc2.remove_urlpath(urlpath)


def test_batchstore_arrow_ipc_roundtrip():
    pa = pytest.importorskip("pyarrow")
    urlpath = "test_batchstore_arrow_ipc.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(storage=_storage(True, urlpath), serializer="arrow")
    assert barray.serializer == "arrow"
    assert barray.meta["batchstore"]["serializer"] == "arrow"

    batch1 = pa.array([[1, 2], None, [3]])
    batch2 = pa.array([[4], [5, 6]])
    barray.append(batch1)
    barray.append(batch2)

    assert barray[0][:] == [[1, 2], None, [3]]
    assert barray[1][:] == [[4], [5, 6]]
    assert barray.meta["batchstore"]["arrow_schema"] is not None

    reopened = blosc2.open(urlpath, mode="r")
    assert isinstance(reopened, blosc2.BatchStore)
    assert reopened.serializer == "arrow"
    assert reopened.meta["batchstore"]["serializer"] == "arrow"
    assert reopened[0][:] == [[1, 2], None, [3]]
    assert reopened[1][:] == [[4], [5, 6]]

    blosc2.remove_urlpath(urlpath)


def test_batchstore_inferred_layout_preserves_user_vlmeta():
    barray = blosc2.BatchStore()
    barray.vlmeta["user"] = {"x": 1}

    barray.append([1, 2, 3])

    assert barray.vlmeta["user"] == {"x": 1}


def test_batchstore_arrow_layout_persistence_preserves_user_vlmeta():
    pa = pytest.importorskip("pyarrow")

    barray = blosc2.BatchStore(serializer="arrow")
    barray.vlmeta["user"] = {"x": 1}

    barray.append(pa.array([[1], [2, 3]]))

    assert barray.vlmeta["user"] == {"x": 1}


def test_batchstore_from_cframe():
    barray = blosc2.BatchStore()
    barray.extend(BATCHES)
    barray.insert(1, ["inserted", True, None])
    del barray[3]
    expected = list(BATCHES)
    expected.insert(1, ["inserted", True, None])
    del expected[3]

    restored = blosc2.from_cframe(barray.to_cframe())
    assert isinstance(restored, blosc2.BatchStore)
    assert [batch[:] for batch in restored] == expected

    restored2 = blosc2.from_cframe(barray.to_cframe())
    assert isinstance(restored2, blosc2.BatchStore)
    assert [batch[:] for batch in restored2] == expected


def test_batchstore_msgpack_supports_blosc2_objects():
    ndarray, schunk, nested_vlarray, nested_batchstore, estore = _make_nested_blosc2_objects()

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([ndarray, schunk, nested_vlarray, nested_batchstore, estore])

    restored = barray[0][:]

    assert isinstance(restored[0], blosc2.NDArray)
    assert np.array_equal(restored[0][:], ndarray[:])

    assert isinstance(restored[1], blosc2.SChunk)
    assert restored[1].decompress_chunk(0) == schunk.decompress_chunk(0)

    assert isinstance(restored[2], blosc2.VLArray)
    assert list(restored[2]) == list(nested_vlarray)

    assert isinstance(restored[3], blosc2.BatchStore)
    assert [batch[:] for batch in restored[3]] == [batch[:] for batch in nested_batchstore]

    assert isinstance(restored[4], blosc2.EmbedStore)
    assert list(restored[4].keys()) == ["/node"]
    assert np.array_equal(restored[4]["/node"][:], estore["/node"][:])


def test_msgpack_supports_c2array(monkeypatch):
    c2array = _make_c2array(monkeypatch)

    payload = msgpack_packb({"remote": c2array})
    restored = msgpack_unpackb(payload)

    assert isinstance(restored["remote"], blosc2.C2Array)
    assert restored["remote"].path == c2array.path
    assert restored["remote"].urlbase == c2array.urlbase
    assert restored["remote"].auth_token is None


def test_msgpack_supports_ref(tmp_path):
    ref, expected = _make_persistent_ref(tmp_path)

    restored = msgpack_unpackb(msgpack_packb({"ref": ref}))["ref"]

    assert isinstance(restored, blosc2.Ref)
    assert restored == ref
    np.testing.assert_array_equal(restored.open()[:], expected)


def test_batchstore_msgpack_supports_c2array(monkeypatch):
    c2array = _make_c2array(monkeypatch)

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([c2array])

    restored = barray[0][0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == c2array.path
    assert restored.urlbase == c2array.urlbase
    assert restored.auth_token is None


def test_msgpack_supports_lazyexpr(tmp_path):
    expr, expected = _make_persistent_lazyexpr(tmp_path)

    payload = msgpack_packb({"expr": expr})
    restored = msgpack_unpackb(payload)["expr"]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


def test_batchstore_msgpack_supports_lazyexpr(tmp_path):
    expr, expected = _make_persistent_lazyexpr(tmp_path)

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([expr])

    restored = barray[0][0]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


def test_msgpack_supports_lazyudf_dslkernel(tmp_path):
    udf, expected = _make_persistent_lazyudf(tmp_path)

    restored = msgpack_unpackb(msgpack_packb({"udf": udf}))["udf"]

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], expected)


def test_batchstore_msgpack_supports_lazyudf_dslkernel(tmp_path):
    udf, expected = _make_persistent_lazyudf(tmp_path)

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([udf])
    restored = barray[0][0]

    assert isinstance(restored, blosc2.LazyUDF)
    np.testing.assert_allclose(restored[:], expected)


def test_msgpack_rejects_plain_python_lazyudf(tmp_path):
    udf = _make_persistent_python_lazyudf(tmp_path)

    with pytest.raises(TypeError, match="DSLKernel"):
        msgpack_packb({"udf": udf})


def test_msgpack_rejects_lazyexpr_with_in_memory_operands():
    expr = _make_in_memory_lazyexpr()

    with pytest.raises(ValueError, match="stored on disk/network"):
        msgpack_packb({"expr": expr})


def test_batchstore_msgpack_rejects_lazyexpr_with_in_memory_operands():
    expr = _make_in_memory_lazyexpr()

    barray = blosc2.BatchStore(items_per_block=2)
    with pytest.raises(ValueError, match="stored on disk/network"):
        barray.append([expr])


@pytest.mark.network
def test_msgpack_supports_lazyexpr_with_c2array_operand(cat2_context, tmp_path):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    a = blosc2.C2Array(path)
    a_values = np.asarray(a[:])
    b = blosc2.asarray(a_values * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})

    restored = msgpack_unpackb(msgpack_packb({"expr": expr}))["expr"]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_allclose(restored[:], a_values + b[:])


@pytest.mark.network
def test_batchstore_msgpack_supports_lazyexpr_with_c2array_operand(cat2_context, tmp_path):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    a = blosc2.C2Array(path)
    a_values = np.asarray(a[:])
    b = blosc2.asarray(a_values * 2, urlpath=tmp_path / "b.b2nd", mode="w")
    expr = blosc2.lazyexpr("a + b", operands={"a": a, "b": b})

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([expr])
    restored = barray[0][0]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_allclose(restored[:], a_values + b[:])


@pytest.mark.parametrize("suffix", [".b2d", ".b2z"])
def test_msgpack_lazyexpr_with_dictstore_operands(tmp_path, suffix):
    store_path = tmp_path / f"operands{suffix}"
    ext_a = tmp_path / "a.b2nd"
    ext_b = tmp_path / "b.b2nd"
    expected = np.arange(5, dtype=np.int64) * 3

    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=str(ext_a), mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2, urlpath=str(ext_b), mode="w")
    with blosc2.DictStore(str(store_path), mode="w", threshold=None) as dstore:
        dstore["/a"] = a
        dstore["/b"] = b

    with blosc2.DictStore(str(store_path), mode="r") as dstore:
        expr = blosc2.lazyexpr("a + b", operands={"a": dstore["/a"], "b": dstore["/b"]})
        restored = msgpack_unpackb(msgpack_packb({"expr": expr}))["expr"]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


def test_batchstore_msgpack_lazyexpr_with_dictstore_operands(tmp_path):
    store_path = tmp_path / "operands.b2z"
    ext_a = tmp_path / "a.b2nd"
    ext_b = tmp_path / "b.b2nd"
    expected = np.arange(5, dtype=np.int64) * 3

    a = blosc2.asarray(np.arange(5, dtype=np.int64), urlpath=str(ext_a), mode="w")
    b = blosc2.asarray(np.arange(5, dtype=np.int64) * 2, urlpath=str(ext_b), mode="w")
    with blosc2.DictStore(str(store_path), mode="w", threshold=None) as dstore:
        dstore["/a"] = a
        dstore["/b"] = b

    with blosc2.DictStore(str(store_path), mode="r") as dstore:
        expr = blosc2.lazyexpr("a + b", operands={"a": dstore["/a"], "b": dstore["/b"]})
        barray = blosc2.BatchStore(items_per_block=2)
        barray.append([expr])
        restored = barray[0][0]

    assert isinstance(restored, blosc2.LazyExpr)
    np.testing.assert_array_equal(restored[:], expected)


@pytest.mark.network
def test_msgpack_roundtrip_c2array_network(cat2_context):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    original = blosc2.C2Array(path)

    payload = msgpack_packb({"remote": original})
    restored = msgpack_unpackb(payload)["remote"]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    np.testing.assert_allclose(restored[:], original[:])


@pytest.mark.network
def test_batchstore_msgpack_roundtrip_c2array_network(cat2_context):
    path = "@public/expr/ds-1-2-linspace-float64-b2-(5,)d.b2nd"
    original = blosc2.C2Array(path)

    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([original])

    restored = barray[0][0]

    assert isinstance(restored, blosc2.C2Array)
    assert restored.path == original.path
    assert restored.urlbase == original.urlbase
    np.testing.assert_allclose(restored[:], original[:])


def test_batchstore_info():
    barray = blosc2.BatchStore()
    barray.extend(BATCHES)

    assert barray.typesize == 1
    assert barray.contiguous == barray.schunk.contiguous
    assert barray.urlpath == barray.schunk.urlpath

    items = dict(barray.info_items)
    assert items["type"] == "BatchStore"
    assert items["serializer"] == "msgpack"
    assert items["nbatches"].startswith(f"{len(BATCHES)} (items per batch: mean=")
    assert items["nblocks"].startswith(str(len(BATCHES)))
    assert items["nitems"] == sum(len(batch) for batch in BATCHES)
    assert "urlpath" not in items
    assert "contiguous" not in items
    assert "typesize" not in items
    assert "(" in items["nbytes"]
    assert "(" in items["cbytes"]
    assert "B)" in items["nbytes"] or "KiB)" in items["nbytes"] or "MiB)" in items["nbytes"]

    text = repr(barray.info)
    assert "type" in text
    assert "serializer" in text
    assert "BatchStore" in text
    assert "items per batch" in text
    assert "items per block" in text


def test_batchstore_info_uses_persisted_batch_lengths():
    barray = blosc2.BatchStore()
    barray.extend(BATCHES)

    assert barray.vlmeta["_batch_store_metadata"]["batch_lengths"] == [len(batch) for batch in BATCHES]

    def fail_decode(*args, **kwargs):
        raise AssertionError(
            "info() should not deserialize batches when batch_lengths metadata is available"
        )

    original_decode_blocks = barray._decode_blocks
    barray._decode_blocks = fail_decode
    try:
        items = dict(barray.info_items)
    finally:
        barray._decode_blocks = original_decode_blocks

    assert items["nitems"] == sum(len(batch) for batch in BATCHES)
    assert "items per batch: mean=" in items["nbatches"]


def test_batchstore_info_reports_exact_block_stats_from_lazy_chunks():
    barray = blosc2.BatchStore(items_per_block=2)
    barray.extend([[1, 2, 3, 4, 5], [6, 7], [8]])

    items = dict(barray.info_items)
    assert items["nblocks"] == "5 (items per block: mean=1.60, max=2, min=1)"


def test_batchstore_pop_keeps_batch_lengths_metadata_in_sync():
    barray = blosc2.BatchStore(items_per_block=2)
    barray.extend([[1, 2, 3], [4, 5], [6]])

    removed = barray.pop(1)

    assert removed == [4, 5]
    assert [batch[:] for batch in barray] == [[1, 2, 3], [6]]
    assert barray.vlmeta["_batch_store_metadata"]["batch_lengths"] == [3, 1]
    items = dict(barray.info_items)
    assert items["nbatches"].startswith("2 (items per batch: mean=2.00")


def test_batchstore_clear_keeps_empty_store_vlmeta_readable():
    urlpath = "test_batchstore_clear_empty_vlmeta.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(urlpath=urlpath, mode="w", contiguous=True)
    barray.append([1, 2, 3])
    barray.clear()

    assert barray.vlmeta.getall() == {}

    reopened = blosc2.open(urlpath, mode="r")
    assert reopened.vlmeta.getall() == {}

    blosc2.remove_urlpath(urlpath)


def test_batchstore_delete_last_keeps_empty_store_vlmeta_readable():
    urlpath = "test_batchstore_delete_last_empty_vlmeta.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(urlpath=urlpath, mode="w", contiguous=True)
    barray.append([1, 2, 3])
    barray.delete(0)

    assert barray.vlmeta.getall() == {}

    reopened = blosc2.open(urlpath, mode="r")
    assert reopened.vlmeta.getall() == {}

    blosc2.remove_urlpath(urlpath)


def test_batchstore_zstd_does_not_use_dict_by_default():
    barray = blosc2.BatchStore()
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False


def test_batchstore_explicit_items_per_block():
    barray = blosc2.BatchStore(items_per_block=2)
    assert barray.items_per_block == 2
    barray.append([1, 2, 3])
    barray.append([4])
    assert [batch[:] for batch in barray] == [[1, 2, 3], [4]]


def test_batchstore_get_vlblock_and_scalar_access():
    urlpath = "test_batchstore_vlblock.b2b"
    blosc2.remove_urlpath(urlpath)

    batch = [0, 1, 2, 3, 4]
    barray = blosc2.BatchStore(storage=_storage(True, urlpath), items_per_block=2)
    barray.append(batch)

    assert barray.items_per_block == 2
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 0)) == batch[:2]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 1)) == batch[2:4]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 2)) == batch[4:]

    assert barray[0][0] == 0
    assert barray[0][2] == 2
    assert barray[0][4] == 4

    reopened = blosc2.open(urlpath, mode="r")
    assert isinstance(reopened, blosc2.BatchStore)
    assert reopened.items_per_block == 2
    assert reopened[0][0] == 0
    assert reopened[0][2] == 2
    assert reopened[0][4] == 4
    assert msgpack_unpackb(reopened.schunk.get_vlblock(0, 1)) == batch[2:4]

    blosc2.remove_urlpath(urlpath)


def test_batchstore_scalar_reads_cache_vlblocks():
    barray = blosc2.BatchStore(items_per_block=2)
    barray.append([0, 1, 2, 3, 4])

    batch = barray[0]
    original_get_vlblock = barray.schunk.get_vlblock
    calls = []

    def wrapped_get_vlblock(nchunk, nblock):
        calls.append((nchunk, nblock))
        return original_get_vlblock(nchunk, nblock)

    barray.schunk.get_vlblock = wrapped_get_vlblock
    try:
        assert batch[0] == 0
        assert batch[1] == 1
        assert batch[0] == 0
        assert batch[2] == 2
        assert batch[3] == 3
        assert calls == [(0, 0), (0, 1)]
    finally:
        barray.schunk.get_vlblock = original_get_vlblock


def test_batchstore_iter_items():
    barray = blosc2.BatchStore(items_per_block=2)
    batches = [[1, 2, 3], [4], [5, 6]]
    barray.extend(batches)

    assert [batch[:] for batch in barray] == batches
    assert list(barray.iter_items()) == [1, 2, 3, 4, 5, 6]


def test_batchstore_respects_explicit_use_dict_and_non_zstd():
    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
    assert barray.cparams.codec == blosc2.Codec.LZ4
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1, "use_dict": True})
    assert barray.cparams.codec == blosc2.Codec.LZ4HC
    assert barray.cparams.use_dict is True

    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 0})
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "use_dict": False})
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, use_dict=False))
    assert barray.cparams.use_dict is False


def test_batchstore_guess_items_per_block_uses_l2_for_clevel_5(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 100)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 1000)
    barray = blosc2.BatchStore(cparams={"clevel": 5})
    assert barray._guess_blocksize([30, 30, 30, 30]) == 4


def test_batchstore_guess_items_per_block_uses_l2_for_mid_clevel(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 100)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 150)
    barray = blosc2.BatchStore(cparams={"clevel": 6})
    assert barray._guess_blocksize([60, 60, 60, 60]) == 2


def test_batchstore_guess_items_per_block_uses_full_batch_for_clevel_9(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 1)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 1)
    barray = blosc2.BatchStore(cparams={"clevel": 9})
    assert barray._guess_blocksize([100, 100, 100, 100]) == 4


def test_vlcompress_small_blocks_roundtrip():
    values = [
        {"value": None},
        {"value": []},
        {"value": []},
        {"value": ["en:salt"]},
        {"value": []},
        {"value": ["en:sugar", "en:flour"]},
        {"value": None},
        {"value": []},
        {"value": ["en:water", "en:yeast", "en:oil"]},
        {"value": []},
        {"value": []},
        {"value": ["en:acid", "en:color", "en:preservative", "en:spice"]},
        {"value": None},
        {"value": []},
        {"value": ["en:a", "en:b", "en:c", "en:d", "en:e", "en:f"]},
        {"value": []},
        {"value": []},
        {"value": None},
        {"value": ["en:x"]},
        {"value": []},
    ]
    payloads = [msgpack_packb(value) for value in values]

    batch_payload = blosc2.blosc2_ext.vlcompress(
        payloads,
        codec=blosc2.Codec.ZSTD,
        clevel=5,
        typesize=1,
        nthreads=1,
    )
    out = blosc2.blosc2_ext.vldecompress(batch_payload, nthreads=1)

    assert out == payloads


def test_batchstore_constructor_kwargs():
    urlpath = "test_batchstore_kwargs.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(urlpath=urlpath, mode="w", contiguous=True)
    barray.extend(BATCHES)

    reopened = blosc2.BatchStore(urlpath=urlpath, mode="r", contiguous=True, mmap_mode="r")
    assert [batch[:] for batch in reopened] == BATCHES

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batchstore_list_ops.b2b"),
        (False, "test_batchstore_list_ops_s.b2b"),
    ],
)
def test_batchstore_list_like_ops(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchStore(storage=_storage(contiguous, urlpath))
    barray.extend([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert [batch[:] for batch in barray] == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert barray.pop() == [7, 8, 9]
    assert barray.pop(0) == [1, 2, 3]
    assert [batch[:] for batch in barray] == [[4, 5, 6]]

    barray.clear()
    assert len(barray) == 0
    assert [batch[:] for batch in barray] == []

    barray.extend([["a", "b", "c"], ["d", "e", "f"]])
    assert [batch[:] for batch in barray] == [["a", "b", "c"], ["d", "e", "f"]]

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert [batch[:] for batch in reopened] == [["a", "b", "c"], ["d", "e", "f"]]

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batchstore_slices.b2b"),
        (False, "test_batchstore_slices_s.b2b"),
    ],
)
def test_batchstore_slices(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    expected = [[i, i + 100, i + 200] for i in range(8)]
    barray = blosc2.BatchStore(storage=_storage(contiguous, urlpath))
    barray.extend(expected)

    assert [batch[:] for batch in barray[1:6:2]] == expected[1:6:2]
    assert [batch[:] for batch in barray[::-2]] == expected[::-2]

    barray[2:5] = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
    expected[2:5] = [["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]]
    assert [batch[:] for batch in barray] == expected

    barray[1:6:2] = [[100, 101, 102], [103, 104, 105], [106, 107, 108]]
    expected[1:6:2] = [[100, 101, 102], [103, 104, 105], [106, 107, 108]]
    assert [batch[:] for batch in barray] == expected

    del barray[::3]
    del expected[::3]
    assert [batch[:] for batch in barray] == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert [batch[:] for batch in reopened[::2]] == expected[::2]
        with pytest.raises(ValueError):
            reopened[1:3] = [[9]]
        with pytest.raises(ValueError):
            del reopened[::2]

    blosc2.remove_urlpath(urlpath)


def test_batchstore_slice_errors():
    barray = blosc2.BatchStore()
    barray.extend([[0], [1], [2], [3]])

    with pytest.raises(ValueError, match="extended slice"):
        barray[::2] = [[9]]
    with pytest.raises(TypeError):
        barray[1:2] = 3
    with pytest.raises(ValueError):
        _ = barray[::0]


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batchstore_items.b2b"),
        (False, "test_batchstore_items_s.b2b"),
    ],
)
def test_batchstore_items_accessor(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    batches = [["a", "b"], [10, 11, 12], [{"x": 1}], [None, True]]
    flat = [item for batch in batches for item in batch]
    barray = blosc2.BatchStore(storage=_storage(contiguous, urlpath), items_per_block=2)
    barray.extend(batches)

    assert len(barray.items) == len(flat)
    assert barray.items[0] == flat[0]
    assert barray.items[3] == flat[3]
    assert barray.items[-1] == flat[-1]
    assert barray.items[1:6] == flat[1:6]
    assert barray.items[::-2] == flat[::-2]

    barray.append(["tail0", "tail1"])
    flat.extend(["tail0", "tail1"])
    assert len(barray.items) == len(flat)
    assert barray.items[-2:] == flat[-2:]

    barray.insert(1, ["mid0", "mid1"])
    flat[2:2] = ["mid0", "mid1"]
    assert barray.items[:] == flat

    barray[2] = ["replaced"]
    batch_start = len(batches[0]) + 2
    flat[batch_start : batch_start + 3] = ["replaced"]
    assert barray.items[:] == flat

    del barray[0]
    del flat[:2]
    assert barray.items[:] == flat

    with pytest.raises(IndexError, match="item index out of range"):
        _ = barray.items[len(flat)]
    with pytest.raises(TypeError, match="item indices must be integers"):
        _ = barray.items[1.5]
    with pytest.raises(ValueError):
        _ = barray.items[::0]

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert reopened.items[:] == flat
        assert reopened.items[2] == flat[2]

    blosc2.remove_urlpath(urlpath)


def test_batchstore_copy():
    urlpath = "test_batchstore_copy.b2b"
    copy_path = "test_batchstore_copy_out.b2b"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.BatchStore(urlpath=urlpath, mode="w", contiguous=True)
    original.extend(BATCHES)
    original.insert(1, ["copy", True, 123])

    copied = original.copy(
        urlpath=copy_path, contiguous=False, cparams={"codec": blosc2.Codec.LZ4, "clevel": 5}
    )
    assert [batch[:] for batch in copied] == [batch[:] for batch in original]
    assert copied.urlpath == copy_path
    assert copied.schunk.contiguous is False
    assert copied.cparams.codec == blosc2.Codec.LZ4
    assert copied.cparams.clevel == 5

    inmem = original.copy()
    assert [batch[:] for batch in inmem] == [batch[:] for batch in original]
    assert inmem.urlpath is None

    with pytest.raises(ValueError, match="meta should not be passed to copy"):
        original.copy(meta={})

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)


def test_batchstore_copy_with_storage_preserves_user_metadata():
    urlpath = "test_batchstore_copy_storage.b2b"
    copy_path = "test_batchstore_copy_storage_out.b2b"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.BatchStore(urlpath=urlpath, mode="w", contiguous=True, meta={"user_meta": {"a": 1}})
    original.vlmeta["user_vlmeta"] = {"b": 2}
    original.extend(BATCHES)

    copied = original.copy(storage=blosc2.Storage(contiguous=False, urlpath=copy_path, mode="w"))

    assert [batch[:] for batch in copied] == [batch[:] for batch in original]
    assert copied.meta["user_meta"] == {"a": 1}
    assert copied.vlmeta["user_vlmeta"] == {"b": 2}

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)


@pytest.mark.parametrize(("contiguous", "nthreads"), [(False, 2), (True, 4)])
def test_batchstore_multithreaded_inner_vl(contiguous, nthreads):
    batches = []
    for batch_id in range(24):
        batch = []
        for obj_id, size in enumerate(
            (13, 1024 + batch_id * 17, 70_000 + batch_id * 13, 250_000 + batch_id * 101)
        ):
            batch.append(
                {
                    "batch": batch_id,
                    "obj": obj_id,
                    "size": size,
                    "payload": _make_payload(batch_id + obj_id, size),
                }
            )
        batches.append(batch)

    barray = blosc2.BatchStore(
        storage=blosc2.Storage(contiguous=contiguous),
        cparams=blosc2.CParams(typesize=1, nthreads=nthreads, codec=blosc2.Codec.ZSTD, clevel=5),
        dparams=blosc2.DParams(nthreads=nthreads),
    )
    barray.extend(batches)

    assert [batch[:] for batch in barray] == batches
    assert [barray[i][:] for i in range(len(barray))] == batches


def test_batchstore_validation_errors():
    barray = blosc2.BatchStore()

    with pytest.raises(TypeError):
        barray.append("value")
    with pytest.raises(ValueError):
        barray.append([])
    with pytest.raises(TypeError):
        barray.insert("0", ["bad"])
    with pytest.raises(IndexError):
        barray.delete(3)
    with pytest.raises(IndexError):
        blosc2.BatchStore().pop()
    barray.extend([[1, 2, 3]])
    assert barray.append([2, 3]) == 2
    assert [batch[:] for batch in barray] == [[1, 2, 3], [2, 3]]
    with pytest.raises(NotImplementedError):
        barray.pop(slice(0, 1))


def test_batchstore_in_embed_store():
    estore = blosc2.EmbedStore()
    barray = blosc2.BatchStore()
    barray.extend(BATCHES)

    estore["/batch"] = barray
    restored = estore["/batch"]
    assert isinstance(restored, blosc2.BatchStore)
    assert [batch[:] for batch in restored] == BATCHES


def test_batchstore_in_dict_store():
    path = "test_batchstore_store.b2z"
    blosc2.remove_urlpath(path)

    with blosc2.DictStore(path, mode="w", threshold=1) as dstore:
        barray = blosc2.BatchStore()
        barray.extend(BATCHES)
        dstore["/batch"] = barray

    with blosc2.DictStore(path, mode="r") as dstore:
        restored = dstore["/batch"]
        assert isinstance(restored, blosc2.BatchStore)
        assert [batch[:] for batch in restored] == BATCHES

    blosc2.remove_urlpath(path)
