#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

from __future__ import annotations

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


def test_listarray_arrow_roundtrip():
    pa = pytest.importorskip("pyarrow")

    values = pa.array([["a"], None, ["b", "c"]])
    arr = blosc2.ListArray.from_arrow(values, item_spec=blosc2.string(), nullable=True)
    assert arr[:] == [["a"], None, ["b", "c"]]
    assert arr.to_arrow().to_pylist() == [["a"], None, ["b", "c"]]
