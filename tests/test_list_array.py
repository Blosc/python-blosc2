#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

from dataclasses import dataclass

import pytest

import blosc2


@pytest.mark.parametrize("storage", ["vl", "batch"])
def test_listarray_append_extend_and_replace(storage, tmp_path):
    urlpath = tmp_path / f"values-{storage}.b2b"
    arr = blosc2.ListArray(
        item_spec=blosc2.string(max_length=16),
        nullable=True,
        storage=storage,
        batch_rows=2,
        urlpath=str(urlpath),
        mode="w",
    )
    arr.append(["a", "b"])
    arr.append([])
    arr.append(None)
    arr.extend([["c"], ["d", "e"]])

    assert len(arr) == 5
    assert arr[0] == ["a", "b"]
    assert arr[1] == []
    assert arr[2] is None
    assert arr[1:4] == [[], None, ["c"]]
    assert arr[[0, 2, 4]] == [["a", "b"], None, ["d", "e"]]

    arr[3] = ["x", "y"]
    assert arr[3] == ["x", "y"]

    arr.flush()
    reopened = blosc2.open(str(urlpath), mode="r")
    assert isinstance(reopened, blosc2.ListArray)
    assert reopened[:] == [["a", "b"], [], None, ["x", "y"], ["d", "e"]]

    restored = blosc2.from_cframe(arr.to_cframe())
    assert isinstance(restored, blosc2.ListArray)
    assert restored[:] == reopened[:]


def test_listarray_batch_pending_rows_visible_before_flush():
    arr = blosc2.ListArray(item_spec=blosc2.int32(), storage="batch", batch_rows=4)
    arr.append([1, 2])
    arr.append([])
    arr.append([3])

    assert len(arr) == 3
    assert arr[:] == [[1, 2], [], [3]]


def test_listarray_rejects_invalid_cells():
    arr = blosc2.ListArray(item_spec=blosc2.int32(), nullable=False)
    with pytest.raises(ValueError):
        arr.append(None)
    with pytest.raises(TypeError):
        arr.append("abc")
    with pytest.raises(ValueError):
        arr.append([1, None])


def test_listarray_boolean_fancy_indexing():
    arr = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="batch", batch_rows=2)
    arr.extend([[1], None, [], [2, 3]])
    assert arr[[3, 0]] == [[2, 3], [1]]
    assert arr[blosc2.asarray([True, False, True, False])[:]] == [[1], []]


def test_listarray_arrow_roundtrip():
    pa = pytest.importorskip("pyarrow")

    values = pa.array([["a"], None, ["b", "c"]])
    arr = blosc2.ListArray.from_arrow(values, item_spec=blosc2.string(), nullable=True)
    assert arr[:] == [["a"], None, ["b", "c"]]
    assert arr.to_arrow().to_pylist() == [["a"], None, ["b", "c"]]


def test_listarray_extend_validate_false_preserves_none():
    arr = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="batch", batch_rows=2)
    arr.extend([[1], None, [2, 3]], validate=False)
    assert arr[:] == [[1], None, [2, 3]]


# ---------------------------------------------------------------------------
# ListArray.copy() fast-path tests
# ---------------------------------------------------------------------------

ROWS = [[1, 2, 3], [], [4], None, [5, 6]]


def _make_batch_array(rows=None, **kwargs):
    arr = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="batch", **kwargs)
    arr.extend(rows or ROWS)
    arr.flush()
    return arr


def test_listarray_copy_fast_path_inmemory():
    src = _make_batch_array()
    dst = src.copy()
    assert dst[:] == ROWS
    assert dst._pending_cells == []


def test_listarray_copy_fast_path_persistent(tmp_path):
    src = _make_batch_array()
    dst_path = str(tmp_path / "copy.b2b")
    dst = src.copy(urlpath=dst_path, mode="w")
    assert dst[:] == ROWS
    assert dst.urlpath == dst_path

    reopened = blosc2.open(dst_path)
    assert reopened[:] == ROWS


def test_listarray_copy_fast_path_preserves_nulls():
    rows = [None, [1], None, [], None]
    src = _make_batch_array(rows=rows)
    dst = src.copy()
    assert dst[:] == rows


def test_listarray_copy_fast_path_empty():
    src = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="batch")
    src.flush()
    dst = src.copy()
    assert len(dst) == 0
    assert dst[:] == []


def test_listarray_copy_cparams_override_uses_slow_path():
    # Supplying cparams must bypass chunk_copy and still produce correct data.
    src = _make_batch_array()
    dst = src.copy(cparams={"codec": blosc2.Codec.LZ4, "clevel": 1})
    assert dst[:] == ROWS
    assert dst._backend.cparams.codec == blosc2.Codec.LZ4


def test_listarray_copy_pending_cells_uses_slow_path():
    # A ListArray with unflushed pending cells must fall back to extend().
    src = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="batch", batch_rows=10)
    src.extend(ROWS)
    # Do NOT flush — _pending_cells is non-empty.
    assert src._pending_cells

    dst = src.copy()
    assert dst[:] == ROWS


def test_listarray_copy_vl_storage_uses_slow_path():
    src = blosc2.ListArray(item_spec=blosc2.int32(), nullable=True, storage="vl")
    src.extend(ROWS)
    dst = src.copy()
    assert dst[:] == ROWS
    assert dst.spec.storage == "vl"


def test_listarray_copy_large_batch(tmp_path):
    # Many rows with multiple chunks to exercise batch_lengths persistence.
    rows = [[i, i + 1] for i in range(0, 10000, 2)]
    src = blosc2.ListArray(item_spec=blosc2.int32(), storage="batch", batch_rows=500)
    src.extend(rows)
    src.flush()

    dst_path = str(tmp_path / "large.b2b")
    dst = src.copy(urlpath=dst_path, mode="w")
    assert len(dst) == len(rows)
    assert dst[0] == rows[0]
    assert dst[-1] == rows[-1]

    reopened = blosc2.open(dst_path)
    assert len(reopened) == len(rows)


# ---------------------------------------------------------------------------
# CTable integration: copy with list column uses fast path
# ---------------------------------------------------------------------------


@dataclass
class _RowWithList:
    tags: list[int] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.int32(), nullable=True, batch_rows=4)
    )
    value: float = blosc2.field(blosc2.float64())


_LIST_DATA = [
    ([1, 2], 0.5),
    (None, 1.0),
    ([3], 2.0),
    ([], 3.0),
]


def test_ctable_copy_with_list_column(tmp_path):
    t = blosc2.CTable(_RowWithList, new_data=_LIST_DATA)

    dst_path = str(tmp_path / "ctable_copy.b2z")
    copied = t.copy(urlpath=dst_path, overwrite=True)

    assert copied.nrows == len(_LIST_DATA)
    for i, (expected_tags, expected_val) in enumerate(_LIST_DATA):
        assert copied.tags[i] == expected_tags
        assert copied.value[i] == pytest.approx(expected_val)


def test_ctable_copy_with_list_column_correctness(tmp_path):
    # Verify chunk-level copy matches element-wise reference on a larger dataset.
    rows = [([i % 5, i % 3], float(i)) for i in range(200)]
    rows[50] = (None, 50.0)
    rows[100] = ([], 100.0)

    t = blosc2.CTable(_RowWithList, new_data=rows)
    dst_path = str(tmp_path / "correctness.b2z")
    copied = t.copy(urlpath=dst_path, overwrite=True)

    assert copied.nrows == len(rows)
    assert copied.tags[:] == [r[0] for r in rows]


def test_listarray_extend_arrow_flushes_pending_rows():
    # Regression: extend_arrow appended chunks straight to the backend,
    # reordering them ahead of unflushed pending cells.
    pa = pytest.importorskip("pyarrow")
    arr = blosc2.ListArray(
        item_spec=blosc2.schema.int64(), storage="batch", serializer="arrow", batch_rows=100
    )
    arr.append([1, 2])  # pending, not yet flushed
    arr.extend_arrow(pa.array([[3, 4], [5, 6]], type=pa.list_(pa.int64())))
    arr.flush()
    assert arr[:] == [[1, 2], [3, 4], [5, 6]]
