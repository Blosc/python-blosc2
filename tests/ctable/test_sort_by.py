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


@dataclass
class DictSortRow:
    value: int = blosc2.field(blosc2.int64(ge=0))
    sort_key: int = blosc2.field(blosc2.int64(ge=0))
    label: str = blosc2.field(blosc2.dictionary())


@dataclass
class ListRow:
    id: int = blosc2.field(blosc2.int64(ge=0))
    tags: list[str] = blosc2.field(  # noqa: RUF009
        blosc2.list(blosc2.string(max_length=16), nullable=True, batch_rows=2)
    )


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
    np.testing.assert_array_equal(s["id"][:], [0, 1, 2, 3, 4])


def test_sort_accepts_column_selector():
    t = CTable(Row, new_data=DATA)

    s = t.sort_by(t.id)

    np.testing.assert_array_equal(s["id"][:], [0, 1, 2, 3, 4])


def test_sort_accepts_nested_column_selector_from_view():
    t = CTable(Row, new_data=DATA)
    t.rename_column("id", "trip.sec")

    view = t.where(t["trip.sec"] > 0)
    s = view.sort_by(t.trip.sec)

    np.testing.assert_array_equal(s["trip.sec"][:], [1, 2, 3, 4])


def test_sort_projected_view_with_dictionary_column_above_default_capacity():
    n = 5000
    data = [(i, n - i, f"label-{i % 7}") for i in range(n)]
    t = CTable(DictSortRow, new_data=data)

    view = t.where(t.value >= 0, columns=["value", "sort_key", "label"])
    sorted_view = view.sort_by(t.sort_key)

    assert sorted_view.nrows == n
    assert len(sorted_view._cols["label"].codes) >= n
    np.testing.assert_array_equal(sorted_view["sort_key"][:5], [1, 2, 3, 4, 5])
    # Regression check: rendering used to index dictionary codes beyond their capacity.
    assert "label" in str(sorted_view)


def test_sort_accepts_column_selectors_in_multi_key_list():
    t = CTable(Row, new_data=DATA)

    s = t.sort_by([t.score, t.id], ascending=[True, False])

    np.testing.assert_array_equal(s["id"][:][:2], [2, 1])


def test_sort_single_col_descending():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("score", ascending=False)
    np.testing.assert_array_equal(s["score"][:], [90.0, 80.0, 70.0, 50.0, 50.0])


def test_sort_bool_column():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("active")
    # False < True → False rows first
    assert list(s["active"][:]) == [False, False, True, True, True]


def test_sort_string_column():
    t = CTable(StrRow, new_data=[("charlie", 3), ("alice", 1), ("dave", 4), ("bob", 2)])
    s = t.sort_by("label")
    assert list(s["label"][:]) == ["alice", "bob", "charlie", "dave"]


def test_sort_string_column_descending():
    t = CTable(StrRow, new_data=[("charlie", 3), ("alice", 1), ("dave", 4), ("bob", 2)])
    s = t.sort_by("label", ascending=False)
    assert list(s["label"][:]) == ["dave", "charlie", "bob", "alice"]


# ===========================================================================
# Multi-column sort
# ===========================================================================


def test_sort_multi_col_both_asc():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=[True, True])
    scores = s["score"][:]
    ids = s["id"][:]
    # score asc; tiebreak: id asc (both 50.0 rows → id 1 before id 2)
    assert scores[0] == pytest.approx(50.0)
    assert ids[0] == 1
    assert scores[1] == pytest.approx(50.0)
    assert ids[1] == 2


def test_sort_multi_col_mixed():
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=[True, False])
    scores = s["score"][:]
    ids = s["id"][:]
    # score asc; tiebreak: id desc (both 50.0 rows → id 2 before id 1)
    assert scores[0] == pytest.approx(50.0)
    assert ids[0] == 2
    assert scores[1] == pytest.approx(50.0)
    assert ids[1] == 1


def test_sort_multi_col_ascending_list_notation():
    """Passing ascending=True (single bool) applies to all keys."""
    t = CTable(Row, new_data=DATA)
    s = t.sort_by(["score", "id"], ascending=True)
    np.testing.assert_array_equal(s["id"][:][:2], [1, 2])


# ===========================================================================
# Non-destructive: original table is unchanged
# ===========================================================================


def test_sort_does_not_modify_original():
    t = CTable(Row, new_data=DATA)
    original_ids = t["id"][:].copy()
    _ = t.sort_by("id")
    np.testing.assert_array_equal(t["id"][:], original_ids)


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
    np.testing.assert_array_equal(t["id"][:], [0, 1, 2, 3, 4])


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
    np.testing.assert_array_equal(s["id"][:], [0, 1, 2, 4])
    assert len(s) == 4


def test_sort_inplace_skips_deleted_rows():
    t = CTable(Row, new_data=DATA)
    t.delete([0, 2])  # delete id=3 and id=4
    t.sort_by("id", inplace=True)
    np.testing.assert_array_equal(t["id"][:], [0, 1, 2])
    assert len(t) == 3


def test_sort_all_columns_consistent():
    """All columns move together when sorted."""
    t = CTable(Row, new_data=DATA)
    s = t.sort_by("id")
    ids = s["id"][:]
    scores = s["score"][:]
    # Original DATA: id→score mapping: 0→70, 1→50, 2→50, 3→80, 4→90
    expected = {0: 70.0, 1: 50.0, 2: 50.0, 3: 80.0, 4: 90.0}
    for i, v in zip(ids, scores, strict=True):
        assert v == pytest.approx(expected[int(i)])


def test_sort_copy_keeps_list_columns_aligned():
    data = [(3, ["c"]), (1, ["a", "one"]), (4, ["d"]), (2, None), (0, [])]
    t = CTable(ListRow, new_data=data)

    s = t.sort_by("id")

    assert list(s["id"][:]) == [0, 1, 2, 3, 4]
    assert s["tags"][:] == [[], ["a", "one"], None, ["c"], ["d"]]
    assert t["tags"][:] == [["c"], ["a", "one"], ["d"], None, []]


def test_sort_inplace_keeps_list_columns_aligned():
    data = [(3, ["c"]), (1, ["a", "one"]), (4, ["d"]), (2, None), (0, [])]
    t = CTable(ListRow, new_data=data)

    result = t.sort_by("id", inplace=True)

    assert result is t
    assert list(t["id"][:]) == [0, 1, 2, 3, 4]
    assert t["tags"][:] == [[], ["a", "one"], None, ["c"], ["d"]]


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
    np.testing.assert_array_equal(s["id"][:], list(range(5)))


def test_sort_reverse_sorted():
    data = [(i, float(i * 10), True) for i in range(5, 0, -1)]
    t = CTable(Row, new_data=data)
    s = t.sort_by("id")
    np.testing.assert_array_equal(s["id"][:], [1, 2, 3, 4, 5])


# ===========================================================================
# Error cases
# ===========================================================================


def test_sort_view_inplace_raises():
    t = CTable(Row, new_data=DATA)
    view = t.where(t["id"] > 2)
    with pytest.raises(ValueError, match="inplace"):
        view.sort_by("id", inplace=True)


def test_sort_view_copy_works():
    t = CTable(Row, new_data=DATA)
    view = t.where(t["id"] > 2)
    sorted_view = view.sort_by("id", ascending=False)
    ids = [sorted_view["id"][i] for i in range(len(sorted_view))]
    assert ids == sorted(ids, reverse=True)


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
    import pathlib
    import shutil

    path_obj = pathlib.Path(__file__).parent / "saved_ctable" / "_sort_ro_test.b2d"
    path = str(path_obj)
    os.makedirs(path_obj.parent, exist_ok=True)
    try:
        t = CTable(Row, urlpath=path, mode="w", new_data=DATA)
        t.close()
        t_ro = CTable.open(path, mode="r")
        with pytest.raises(ValueError, match="read-only"):
            t_ro.sort_by("id", inplace=True)
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ===========================================================================
# Regression: sort_by on an unprojected view must not gather all columns
# ===========================================================================


@dataclass
class WideSortRow:
    a: int = blosc2.field(blosc2.int64(), default=0)
    b: float = blosc2.field(blosc2.float64(), default=0.0)
    c: float = blosc2.field(blosc2.float64(), default=0.0)
    d: str = ""
    e: int = blosc2.field(blosc2.int64(), default=0)


def _loaded_columns(table) -> set[str]:
    """Columns whose payload has actually been opened.

    ``_cols`` is a ``_LazyColumnDict``; bypassing its ``__contains__`` with
    ``dict.__contains__`` reveals what was loaded without forcing a load.
    """
    return {name for name in table.col_names if dict.__contains__(table._cols, name)}


def test_sort_unprojected_view_opens_only_needed_columns(tmp_path):
    """``where(cond).sort_by(key)`` without ``columns=`` used to gather every
    column of the view (~30x slower than projecting first).  It must open only
    the condition and sort-key columns, deferring the rest until read."""
    n = 1000
    i = np.arange(n)
    data = np.empty(n, dtype=[("a", "<i8"), ("b", "<f8"), ("c", "<f8"), ("d", "U10"), ("e", "<i8")])
    data["a"] = i
    data["b"] = (i * 7919) % n  # a permutation: distinct values, no sort ties
    data["c"] = i * 0.5
    data["d"] = np.char.add("s", i.astype("U6"))
    data["e"] = i * 3

    urlpath = str(tmp_path / "wide-sort.b2z")
    t = CTable(WideSortRow, urlpath=urlpath, mode="w", expected_size=n)
    t.extend(data, validate=False)
    t.close()

    t = blosc2.open(urlpath)
    try:
        assert _loaded_columns(t) == set()

        res = t.where(t.a >= n - 100).sort_by("b")
        assert _loaded_columns(t) <= {"a", "b"}
        assert _loaded_columns(res) <= {"a", "b"}

        # Deferred columns are still served correctly, on demand only
        mask = data["a"] >= n - 100
        order = np.argsort(data["b"][mask], kind="stable")
        np.testing.assert_array_equal(res["e"][:], data["e"][mask][order])
        loaded = _loaded_columns(res)
        assert "c" not in loaded
        assert "d" not in loaded
    finally:
        t.close()


if __name__ == "__main__":
    pytest.main(["-v", __file__])
