#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

import pytest

import blosc2
from blosc2._msgpack_utils import msgpack_packb

BATCHES = [
    [b"bytes\x00payload", "plain text", 42],
    [{"nested": [1, 2]}, None],
    [(1, 2, "three"), 3.5, True, {"rows": [[], ["nested"]]}],
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
        (True, "test_batcharray.b2frame"),
        (False, "test_batcharray_s.b2frame"),
    ],
)
def test_batcharray_roundtrip(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
    assert barray.meta["batcharray"]["serializer"] == "msgpack"

    for i, batch in enumerate(BATCHES, start=1):
        assert barray.append(batch) == i

    assert len(barray) == len(BATCHES)
    assert [batch[:] for batch in barray] == BATCHES

    batch0 = barray[0]
    assert isinstance(batch0, blosc2.Batch)
    assert len(batch0) == len(BATCHES[0])
    assert batch0[1] == BATCHES[0][1]
    assert batch0[:] == BATCHES[0]
    assert isinstance(batch0.lazychunk, bytes)
    assert batch0.nbytes > 0
    assert batch0.cbytes > 0
    assert batch0.cratio > 0

    expected = list(BATCHES)
    expected[1] = ["updated", {"tuple": (7, 8)}]
    expected[-1] = ["tiny"]
    barray[1] = expected[1]
    barray[-1] = expected[-1]
    assert barray.insert(0, ["head", 0]) == len(expected) + 1
    expected.insert(0, ["head", 0])
    assert barray.insert(-1, ["between", {"k": 5}]) == len(expected) + 1
    expected.insert(-1, ["between", {"k": 5}])
    assert barray.insert(999, ["tail"]) == len(expected) + 1
    expected.insert(999, ["tail"])
    assert barray.delete(2) == len(expected) - 1
    del expected[2]
    del barray[-2]
    del expected[-2]
    assert [batch[:] for batch in barray] == expected

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert isinstance(reopened, blosc2.BatchArray)
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
        reopened_rw[0] = ["changed"]
        expected[0] = ["changed"]
        assert [batch[:] for batch in reopened_rw] == expected

        if contiguous:
            reopened_mmap = blosc2.open(urlpath, mode="r", mmap_mode="r")
            assert isinstance(reopened_mmap, blosc2.BatchArray)
            assert [batch[:] for batch in reopened_mmap] == expected

    blosc2.remove_urlpath(urlpath)


def test_batcharray_from_cframe():
    barray = blosc2.BatchArray()
    barray.extend(BATCHES)
    barray.insert(1, ["inserted", True])
    del barray[3]
    expected = list(BATCHES)
    expected.insert(1, ["inserted", True])
    del expected[3]

    restored = blosc2.from_cframe(barray.to_cframe())
    assert isinstance(restored, blosc2.BatchArray)
    assert [batch[:] for batch in restored] == expected

    restored2 = blosc2.batcharray_from_cframe(barray.to_cframe())
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
    assert items["nbatches"] == len(BATCHES)
    assert items["nitems"] == sum(len(batch) for batch in BATCHES)
    assert items["batch_len_min"] == 2
    assert items["batch_len_max"] == 4
    assert items["batch_len_avg"] == "3.00"
    assert "urlpath" not in items
    assert "contiguous" not in items
    assert "typesize" not in items
    assert "(" in items["nbytes"]
    assert "(" in items["cbytes"]
    assert "B)" in items["nbytes"] or "KiB)" in items["nbytes"] or "MiB)" in items["nbytes"]

    text = repr(barray.info)
    assert "type" in text
    assert "BatchArray" in text
    assert "batch_len_avg" in text


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

    chunk = blosc2.blosc2_ext.vlcompress(
        payloads,
        codec=blosc2.Codec.ZSTD,
        clevel=5,
        typesize=1,
        nthreads=1,
    )
    out = blosc2.blosc2_ext.vldecompress(chunk, nthreads=1)

    assert out == payloads


def test_batcharray_constructor_kwargs():
    urlpath = "test_batcharray_kwargs.b2frame"
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
        (True, "test_batcharray_list_ops.b2frame"),
        (False, "test_batcharray_list_ops_s.b2frame"),
    ],
)
def test_batcharray_list_like_ops(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
    barray.extend([[1, 2], [3], [4, 5, 6]])
    assert [batch[:] for batch in barray] == [[1, 2], [3], [4, 5, 6]]
    assert barray.pop() == [4, 5, 6]
    assert barray.pop(0) == [1, 2]
    assert [batch[:] for batch in barray] == [[3]]

    barray.clear()
    assert len(barray) == 0
    assert [batch[:] for batch in barray] == []

    barray.extend([["a"], ["b", "c"]])
    assert [batch[:] for batch in barray] == [["a"], ["b", "c"]]

    if urlpath is not None:
        reopened = blosc2.open(urlpath, mode="r")
        assert [batch[:] for batch in reopened] == [["a"], ["b", "c"]]

    blosc2.remove_urlpath(urlpath)


@pytest.mark.parametrize(
    ("contiguous", "urlpath"),
    [
        (False, None),
        (True, None),
        (True, "test_batcharray_slices.b2frame"),
        (False, "test_batcharray_slices_s.b2frame"),
    ],
)
def test_batcharray_slices(contiguous, urlpath):
    blosc2.remove_urlpath(urlpath)

    expected = [[i, i + 100] for i in range(8)]
    barray = blosc2.BatchArray(storage=_storage(contiguous, urlpath))
    barray.extend(expected)

    assert [batch[:] for batch in barray[1:6:2]] == expected[1:6:2]
    assert [batch[:] for batch in barray[::-2]] == expected[::-2]

    barray[2:5] = [["a"], ["b", "c"]]
    expected[2:5] = [["a"], ["b", "c"]]
    assert [batch[:] for batch in barray] == expected

    barray[1:6:2] = [[100], [101], [102]]
    expected[1:6:2] = [[100], [101], [102]]
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


def test_batcharray_copy():
    urlpath = "test_batcharray_copy.b2frame"
    copy_path = "test_batcharray_copy_out.b2frame"
    blosc2.remove_urlpath(urlpath)
    blosc2.remove_urlpath(copy_path)

    original = blosc2.BatchArray(urlpath=urlpath, mode="w", contiguous=True)
    original.extend(BATCHES)
    original.insert(1, ["copy", True])

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
    barray.extend([[1]])
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
