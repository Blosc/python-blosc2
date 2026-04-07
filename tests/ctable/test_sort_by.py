#######################################################################
# Copyright (c) 2019-present, Blosc Development Team <blosc@blosc.org>
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#######################################################################

"""Tests for CTable.sort_by()."""

from dataclasses import dataclass

import numpy as np
import pytest

import blosc2
from blosc2 import CTable


@dataclass
class Row:
    id: int = blosc2.field(blosc2.int64(ge=0))
    score: float = blosc2.field(blosc2.float64(ge=0, le=100), default=0.0)
    active: bool = blosc2.field(blosc2.bool(), default=True)


@dataclass
class StrRow:
    label: str = blosc2.field(blosc2.string(max_length=16))
    rank: int = blosc2.field(blosc2.int64(ge=0), default=0)


DATA = [
    (3, 80.0, True),
    (1, 50.0, False),
    (4, 90.0, True),
    (2, 50.0, True),
    (0, 70.0, False),
]


# ===========================================================================
# Single-column sort
# ===========================================================================


def test_sort_single_col_ascending():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"].to_numpy(), [0, 1, 2, 3, 4])


def test_sort_single_col_descending():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("score", ascending=False)
    np.testing.assert_array_equal(s["score"].to_numpy(), [90.0, 80.0, 70.0, 50.0, 50.0])


def test_sort_bool_column():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("active")
    # False < True → False rows first
    assert list(s["active"].to_numpy()) == [False, False, True, True, True]


def test_sort_string_column():
    t = CTable(StrRow, new_data=[("charlie", 3), ("alice", 1), ("dave", 4), ("bob", 2)])
    s = t.sort_by("label")
    assert list(s["label"].to_numpy()) == ["alice", "bob", "charlie", "dave"]


def test_sort_string_column_descending():
    t = CTable(StrRow, new_data=[("charlie", 3), ("alice", 1), ("dave", 4), ("bob", 2)])
    s = t.sort_by("label", ascending=False)
    assert list(s["label"].to_numpy()) == ["dave", "charlie", "bob", "alice"]


# ===========================================================================
# Multi-column sort
# ===========================================================================


def test_sort_multi_col_both_asc():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=[True, True])
    scores = s["score"].to_numpy()
    ids = s["id"].to_numpy()
    # score asc; tiebreak: id asc (both 50.0 rows → id 1 before id 2)
    assert scores[0] == pytest.approx(50.0)
    assert ids[0] == 1
    assert scores[1] == pytest.approx(50.0)
    assert ids[1] == 2


def test_sort_multi_col_mixed():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=[True, False])
    scores = s["score"].to_numpy()
    ids = s["id"].to_numpy()
    # score asc; tiebreak: id desc (both 50.0 rows → id 2 before id 1)
    assert scores[0] == pytest.approx(50.0)
    assert ids[0] == 2
    assert scores[1] == pytest.approx(50.0)
    assert ids[1] == 1


def test_sort_multi_col_ascending_list_notation():
    """Passing ascending=True (single bool) applies to all keys."""
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=True)
    np.testing.assert_array_equal(s["id"].to_numpy()[:2], [1, 2])


# ===========================================================================
# Non-destructive: original table is unchanged
# ===========================================================================


def test_sort_does_not_modify_original():
    t = CTable(Row, new_data=DATA)
    original_ids = t["id"].to_numpy().copy()
    _ = t.sort_by("id")
    np.testing.assert_array_equal(t["id"].to_numpy(), original_ids)


def test_sort_returns_new_table():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("id")
    assert s is not t


# ===========================================================================
# inplace=True
# ===========================================================================


def test_sort_inplace_returns_self():
    t = CTable(Row, new_data=DATA)
    result = t.sort_by("id", inplace=True)
    assert result is t


def test_sort_inplace_modifies_table():
    t = CTable(Row, new_data=DATA)
    t.sort_by("id", inplace=True)
    np.testing.assert_array_equal(t["id"].to_numpy(), [0, 1, 2, 3, 4])


def test_sort_inplace_descending():
    t = CTable(Row, new_data=DATA)
    t.sort_by("score", ascending=False, inplace=True)
    assert t["score"][0] == pytest.approx(90.0)
    assert t["score"][-1] == pytest.approx(50.0)


# ===========================================================================
# Interaction with deletions
# ===========================================================================


def test_sort_skips_deleted_rows():
    t = CTable(Row, new_data=DATA)
    t.delete([0])  # delete id=3 (first row)
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"].to_numpy(), [0, 1, 2, 4])
    assert len(s) == 4


def test_sort_inplace_skips_deleted_rows():
    t = CTable(Row, new_data=DATA)
    t.delete([0, 2])  # delete id=3 and id=4
    t.sort_by("id", inplace=True)
    np.testing.assert_array_equal(t["id"].to_numpy(), [0, 1, 2])
    assert len(t) == 3


def test_sort_all_columns_consistent():
    """All columns move together when sorted."""
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("id")
    ids = s["id"].to_numpy()
    scores = s["score"].to_numpy()
    # Original DATA: id→score mapping: 0→70, 1→50, 2→50, 3→80, 4→90
    expected = {0: 70.0, 1: 50.0, 2: 50.0, 3: 80.0, 4: 90.0}
    for i, v in zip(ids, scores, strict=True):
        assert v == pytest.approx(expected[int(i)])


# ===========================================================================
# Edge cases
# ===========================================================================


def test_sort_empty_table():
    t = CTable(Row)
    s = t.sort_by("id")
    assert len(s) == 0


def test_sort_single_row():
    t = CTable(Row, new_data=[(7, 42.0, True)])
    s = t.sort_by("id")
    assert s["id"][0] == 7


def test_sort_already_sorted():
    data = [(i, float(i * 10), True) for i in range(5)]
    t = CTable(Row, new_data=data)
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"].to_numpy(), list(range(5)))


def test_sort_reverse_sorted():
    data = [(i, float(i * 10), True) for i in range(5, 0, -1)]
    t = CTable(Row, new_data=data)
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"].to_numpy(), [1, 2, 3, 4, 5])


# ===========================================================================
# Error cases
# ===========================================================================


def test_sort_view_raises():
    t = CTable(Row, new_data=DATA)
    view = t.where(t["id"] > 2)
    with pytest.raises(ValueError, match="view"):
        view.sort_by("id")


def test_sort_unknown_column_raises():
    t = CTable(Row, new_data=DATA)
    with pytest.raises(KeyError):
        t.sort_by("nonexistent")


def test_sort_complex_column_raises():
    @dataclass
    class CRow:
        val: complex = blosc2.field(blosc2.complex128())

    t = CTable(CRow, new_data=[(1 + 2j,), (3 + 4j,)])
    with pytest.raises(TypeError, match="complex"):
        t.sort_by("val")


def test_sort_ascending_length_mismatch_raises():
    t = CTable(Row, new_data=DATA)
    with pytest.raises(ValueError, match="ascending"):
        t.sort_by(["id", "score"], ascending=[True])


def test_sort_readonly_inplace_raises():
    import os
    import shutil

    path = "saved_ctable/_sort_ro_test"
    os.makedirs(path, exist_ok=True)
    try:
        t = CTable(Row, urlpath=path, mode="w", new_data=DATA)
        del t
        t_ro = CTable.open(path, mode="r")
        with pytest.raises(ValueError, match="read-only"):
            t_ro.sort_by("id", inplace=True)
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
