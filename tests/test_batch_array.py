#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest

import blosc2
from blosc2.msgpack_utils import msgpack_packb, msgpack_unpackb

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


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batcharray.b2b"),
        (False, "test_batcharray_s.b2b"),
    ],
)
def test_batcharray_roundtrip(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
    assert barray.meta["batcharray"]["serializer"] == "msgpack"

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
        assert isinstance(reopened, blosc2.BatchArray)
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
            assert isinstance(reopened_mmap, blosc2.BatchArray)
            assert [batch[:] for batch in reopened_mmap] == expected

    blosc2.remove_urlpath(urlpath)


def test_batcharray_arrow_ipc_roundtrip():
    pa = pytest.importorskip("pyarrow")
    urlpath = "test_batcharray_arrow_ipc.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(storage=_storage(True, urlpath), serializer="arrow")
    assert barray.serializer == "arrow"
    assert barray.meta["batcharray"]["serializer"] == "arrow"

    batch1 = pa.array([[1, 2], None, [3]])
    batch2 = pa.array([[4], [5, 6]])
    barray.append(batch1)
    barray.append(batch2)

    assert barray[0][:] == [[1, 2], None, [3]]
    assert barray[1][:] == [[4], [5, 6]]
    assert barray.meta["batcharray"]["arrow_schema"] is not None

    reopened = blosc2.open(urlpath, mode="r")
    assert isinstance(reopened, blosc2.BatchArray)
    assert reopened.serializer == "arrow"
    assert reopened.meta["batcharray"]["serializer"] == "arrow"
    assert reopened[0][:] == [[1, 2], None, [3]]
    assert reopened[1][:] == [[4], [5, 6]]

    blosc2.remove_urlpath(urlpath)


def test_batcharray_inferred_layout_preserves_user_vlmeta():
    barray = blosc2.BatchArray()
    barray.vlmeta["user"] = {"x": 1}

    barray.append([1, 2, 3])

    assert barray.vlmeta["user"] == {"x": 1}


def test_batcharray_arrow_layout_persistence_preserves_user_vlmeta():
    pa = pytest.importorskip("pyarrow")

    barray = blosc2.BatchArray(serializer="arrow")
    barray.vlmeta["user"] = {"x": 1}

    barray.append(pa.array([[1], [2, 3]]))

    assert barray.vlmeta["user"] == {"x": 1}


def test_batcharray_from_cframe():
    barray = blosc2.BatchArray()
    barray.extend(BATCHES)
    barray.insert(1, ["inserted", True, None])
    del barray[3]
    expected = list(BATCHES)
    expected.insert(1, ["inserted", True, None])
    del expected[3]

    restored = blosc2.from_cframe(barray.to_cframe())
    assert isinstance(restored, blosc2.BatchArray)
    assert [batch[:] for batch in restored] == expected

    restored2 = blosc2.from_cframe(barray.to_cframe())
    assert isinstance(restored2, blosc2.BatchArray)
    assert [batch[:] for batch in restored2] == expected


def test_batcharray_info():
    barray = blosc2.BatchArray()
    barray.extend(BATCHES)

    assert barray.typesize == 1
    assert barray.contiguous == barray.schunk.contiguous
    assert barray.urlpath == barray.schunk.urlpath

    items = dict(barray.info_items)
    assert items["type"] == "BatchArray"
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
    assert "BatchArray" in text
    assert "items per batch" in text
    assert "items per block" in text


def test_batcharray_info_uses_persisted_batch_lengths():
    barray = blosc2.BatchArray()
    barray.extend(BATCHES)

    assert barray.vlmeta["_batch_array_metadata"]["batch_lengths"] == [len(batch) for batch in BATCHES]

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


def test_batcharray_info_reports_exact_block_stats_from_lazy_chunks():
    barray = blosc2.BatchArray(items_per_block=2)
    barray.extend([[1, 2, 3, 4, 5], [6, 7], [8]])

    items = dict(barray.info_items)
    assert items["nblocks"] == "5 (items per block: mean=1.60, max=2, min=1)"


def test_batcharray_pop_keeps_batch_lengths_metadata_in_sync():
    barray = blosc2.BatchArray(items_per_block=2)
    barray.extend([[1, 2, 3], [4, 5], [6]])

    removed = barray.pop(1)

    assert removed == [4, 5]
    assert [batch[:] for batch in barray] == [[1, 2, 3], [6]]
    assert barray.vlmeta["_batch_array_metadata"]["batch_lengths"] == [3, 1]
    items = dict(barray.info_items)
    assert items["nbatches"].startswith("2 (items per batch: mean=2.00")


def test_batcharray_clear_keeps_empty_store_vlmeta_readable():
    urlpath = "test_batcharray_clear_empty_vlmeta.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True)
    barray.append([1, 2, 3])
    barray.clear()

    assert barray.vlmeta.getall() == {}

    reopened = blosc2.open(urlpath, mode="r")
    assert reopened.vlmeta.getall() == {}

    blosc2.remove_urlpath(urlpath)


def test_batcharray_delete_last_keeps_empty_store_vlmeta_readable():
    urlpath = "test_batcharray_delete_last_empty_vlmeta.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True)
    barray.append([1, 2, 3])
    barray.delete(0)

    assert barray.vlmeta.getall() == {}

    reopened = blosc2.open(urlpath, mode="r")
    assert reopened.vlmeta.getall() == {}

    blosc2.remove_urlpath(urlpath)


def test_batcharray_zstd_does_not_use_dict_by_default():
    barray = blosc2.BatchArray()
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False


def test_batcharray_explicit_items_per_block():
    barray = blosc2.BatchArray(items_per_block=2)
    assert barray.items_per_block == 2
    barray.append([1, 2, 3])
    barray.append([4])
    assert [batch[:] for batch in barray] == [[1, 2, 3], [4]]


def test_batcharray_get_vlblock_and_scalar_access():
    urlpath = "test_batcharray_vlblock.b2b"
    blosc2.remove_urlpath(urlpath)

    batch = [0, 1, 2, 3, 4]
    barray = blosc2.BatchArray(storage=_storage(True, urlpath), items_per_block=2)
    barray.append(batch)

    assert barray.items_per_block == 2
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 0)) == batch[:2]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 1)) == batch[2:4]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 2)) == batch[4:]

    assert barray[0][0] == 0
    assert barray[0][2] == 2
    assert barray[0][4] == 4

    reopened = blosc2.open(urlpath, mode="r")
    assert isinstance(reopened, blosc2.BatchArray)
    assert reopened.items_per_block == 2
    assert reopened[0][0] == 0
    assert reopened[0][2] == 2
    assert reopened[0][4] == 4
    assert msgpack_unpackb(reopened.schunk.get_vlblock(0, 1)) == batch[2:4]

    blosc2.remove_urlpath(urlpath)


def test_batcharray_scalar_reads_cache_vlblocks():
    barray = blosc2.BatchArray(items_per_block=2)
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


def test_batcharray_iter_items():
    barray = blosc2.BatchArray(items_per_block=2)
    batches = [[1, 2, 3], [4], [5, 6]]
    barray.extend(batches)

    assert [batch[:] for batch in barray] == batches
    assert list(barray.iter_items()) == [1, 2, 3, 4, 5, 6]


def test_batcharray_respects_explicit_use_dict_and_non_zstd():
    barray = blosc2.BatchArray(cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
    assert barray.cparams.codec == blosc2.Codec.LZ4
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchArray(cparams={"codec": blosc2.Codec.LZ4HC, "clevel": 1, "use_dict": True})
    assert barray.cparams.codec == blosc2.Codec.LZ4HC
    assert barray.cparams.use_dict is True

    barray = blosc2.BatchArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 0})
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchArray(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "use_dict": False})
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchArray(cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, use_dict=False))
    assert barray.cparams.use_dict is False


def test_batcharray_guess_items_per_block_uses_l2_for_clevel_5(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 100)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 1000)
    barray = blosc2.BatchArray(cparams={"clevel": 5})
    assert barray._guess_blocksize([30, 30, 30, 30]) == 4


def test_batcharray_guess_items_per_block_uses_l2_for_mid_clevel(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 100)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 150)
    barray = blosc2.BatchArray(cparams={"clevel": 6})
    assert barray._guess_blocksize([60, 60, 60, 60]) == 2


def test_batcharray_guess_items_per_block_uses_full_batch_for_clevel_9(monkeypatch):
    monkeypatch.setitem(blosc2.cpu_info, "l1_data_cache_size", 1)
    monkeypatch.setitem(blosc2.cpu_info, "l2_cache_size", 1)
    barray = blosc2.BatchArray(cparams={"clevel": 9})
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


def test_batcharray_constructor_kwargs():
    urlpath = "test_batcharray_kwargs.b2b"
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True)
    barray.extend(BATCHES)

    reopened = blosc2.BatchArray(urlpath=urlpath, mode="r", contiguous=True, mmap_mode="r")
    assert [batch[:] for batch in reopened] == BATCHES

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batcharray_list_ops.b2b"),
        (False, "test_batcharray_list_ops_s.b2b"),
    ],
)
def test_batcharray_list_like_ops(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
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
        (True, "test_batcharray_slices.b2b"),
        (False, "test_batcharray_slices_s.b2b"),
    ],
)
def test_batcharray_slices(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    expected = [[i, i + 100, i + 200] for i in range(8)]
    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
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


def test_batcharray_slice_errors():
    barray = blosc2.BatchArray()
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
        (True, "test_batcharray_items.b2b"),
        (False, "test_batcharray_items_s.b2b"),
    ],
)
def test_batcharray_items_accessor(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    batches = [["a", "b"], [10, 11, 12], [{"x": 1}], [None, True]]
    flat = [item for batch in batches for item in batch]
    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath), items_per_block=2)
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


def test_batcharray_copy():
    urlpath = "test_batcharray_copy.b2b"
    copy_path = "test_batcharray_copy_out.b2b"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True)
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


def test_batcharray_copy_with_storage_preserves_user_metadata():
    urlpath = "test_batcharray_copy_storage.b2b"
    copy_path = "test_batcharray_copy_storage_out.b2b"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True, meta={"user_meta": {"a": 1}})
    original.vlmeta["user_vlmeta"] = {"b": 2}
    original.extend(BATCHES)

    copied = original.copy(storage=blosc2.Storage(contiguous=False, urlpath=copy_path, mode="w"))

    assert [batch[:] for batch in copied] == [batch[:] for batch in original]
    assert copied.meta["user_meta"] == {"a": 1}
    assert copied.vlmeta["user_vlmeta"] == {"b": 2}

    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)


@pytest.mark.parametrize(("contiguous", "nthreads"), [(False, 2), (True, 4)])
def test_batcharray_multithreaded_inner_vl(contiguous, nthreads):
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

    barray = blosc2.BatchArray(
        storage=blosc2.Storage(contiguous=contiguous),
        cparams=blosc2.CParams(typesize=1, nthreads=nthreads, codec=blosc2.Codec.ZSTD, clevel=5),
        dparams=blosc2.DParams(nthreads=nthreads),
    )
    barray.extend(batches)

    assert [batch[:] for batch in barray] == batches
    assert [barray[i][:] for i in range(len(barray))] == batches


def test_batcharray_validation_errors():
    barray = blosc2.BatchArray()

    with pytest.raises(TypeError):
        barray.append("value")
    with pytest.raises(ValueError):
        barray.append([])
    with pytest.raises(TypeError):
        barray.insert("0", ["bad"])
    with pytest.raises(IndexError):
        barray.delete(3)
    with pytest.raises(IndexError):
        blosc2.BatchArray().pop()
    barray.extend([[1, 2, 3]])
    assert barray.append([2, 3]) == 2
    assert [batch[:] for batch in barray] == [[1, 2, 3], [2, 3]]
    with pytest.raises(NotImplementedError):
        barray.pop(slice(0, 1))


def test_batcharray_in_embed_store():
    estore = blosc2.EmbedStore()
    barray = blosc2.BatchArray()
    barray.extend(BATCHES)

    estore["/batch"] = barray
    restored = estore["/batch"]
    assert isinstance(restored, blosc2.BatchArray)
    assert [batch[:] for batch in restored] == BATCHES


def test_batcharray_in_dict_store():
    path = "test_batcharray_store.b2z"
    blosc2.remove_urlpath(path)

    with blosc2.DictStore(path, mode="w", threshold=1) as dstore:
        barray = blosc2.BatchArray()
        barray.extend(BATCHES)
        dstore["/batch"] = barray

    with blosc2.DictStore(path, mode="r") as dstore:
        restored = dstore["/batch"]
        assert isinstance(restored, blosc2.BatchArray)
        assert [batch[:] for batch in restored] == BATCHES

    blosc2.remove_urlpath(path)
