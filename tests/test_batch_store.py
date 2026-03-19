#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest

import blosc2
from blosc2._msgpack_utils import msgpack_packb, msgpack_unpackb

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
    assert barray.blocksize_max is not None
    assert 1 <= barray.blocksize_max <= len(BATCHES[0])
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
        assert reopened.blocksize_max is None
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


def test_batchstore_info():
    barray = blosc2.BatchStore()
    barray.extend(BATCHES)

    assert barray.typesize == 1
    assert barray.contiguous == barray.schunk.contiguous
    assert barray.urlpath == barray.schunk.urlpath

    items = dict(barray.info_items)
    assert items["type"] == "BatchStore"
    assert items["nbatches"] == len(BATCHES)
    assert items["batch stats"].startswith("mean=")
    assert items["blocksize_max"] == barray.blocksize_max
    assert items["nitems"] == sum(len(batch) for batch in BATCHES)
    assert "urlpath" not in items
    assert "contiguous" not in items
    assert "typesize" not in items
    assert "(" in items["nbytes"]
    assert "(" in items["cbytes"]
    assert "B)" in items["nbytes"] or "KiB)" in items["nbytes"] or "MiB)" in items["nbytes"]

    text = repr(barray.info)
    assert "type" in text
    assert "BatchStore" in text
    assert "batch stats" in text
    assert "blocksize_max" in text


def test_batchstore_zstd_does_not_use_dict_by_default():
    barray = blosc2.BatchStore()
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False


def test_batchstore_explicit_blocksize_max():
    barray = blosc2.BatchStore(blocksize_max=2)
    assert barray.blocksize_max == 2
    barray.append([1, 2, 3])
    barray.append([4])
    assert [batch[:] for batch in barray] == [[1, 2, 3], [4]]


def test_batchstore_get_vlblock_and_scalar_access():
    urlpath = "test_batchstore_vlblock.b2b"
    blosc2.remove_urlpath(urlpath)

    batch = [0, 1, 2, 3, 4]
    barray = blosc2.BatchStore(storage=_storage(True, urlpath), blocksize_max=2)
    barray.append(batch)

    assert barray.blocksize_max == 2
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 0)) == batch[:2]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 1)) == batch[2:4]
    assert msgpack_unpackb(barray.schunk.get_vlblock(0, 2)) == batch[4:]

    assert barray[0][0] == 0
    assert barray[0][2] == 2
    assert barray[0][4] == 4

    reopened = blosc2.open(urlpath, mode="r")
    assert isinstance(reopened, blosc2.BatchStore)
    assert reopened[0][0] == 0
    assert reopened[0][2] == 2
    assert reopened[0][4] == 4
    assert msgpack_unpackb(reopened.schunk.get_vlblock(0, 1)) == batch[2:4]

    blosc2.remove_urlpath(urlpath)


def test_batchstore_scalar_reads_cache_vlblocks():
    barray = blosc2.BatchStore(blocksize_max=2)
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


def test_batchstore_respects_explicit_use_dict_and_non_zstd():
    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.LZ4, "clevel": 5})
    assert barray.cparams.codec == blosc2.Codec.LZ4
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 0})
    assert barray.cparams.codec == blosc2.Codec.ZSTD
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams={"codec": blosc2.Codec.ZSTD, "clevel": 5, "use_dict": False})
    assert barray.cparams.use_dict is False

    barray = blosc2.BatchStore(cparams=blosc2.CParams(codec=blosc2.Codec.ZSTD, clevel=5, use_dict=False))
    assert barray.cparams.use_dict is False


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
